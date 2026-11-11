// Playwright screen check. Run before every frontend commit.
//
//   node scripts/screen_check.js                # 1366x768 + 1920x1080, dark
//   node scripts/screen_check.js --all-sizes    # every s4b-B target incl. 125% / 150% zoom equivalents
//   node scripts/screen_check.js --theme light
//
// Starts the app on a free port, logs in once per viewport, opens every page via the sidebar
// (client-side navigation keeps the session), screenshots to outputs/private/screens_check/ and
// FAILS if documentElement.scrollWidth > innerWidth or any visible element's right edge passes
// the viewport (checked at >= 1024 px; below that the sidebar becomes an overlay drawer).
const { chromium } = require('playwright');
const { spawn } = require('child_process');
const net = require('net');
const fs = require('fs');
const path = require('path');

const REPO = path.resolve(__dirname, '..');
const OUT = path.join(REPO, 'outputs', 'private', 'screens_check');
const DEFAULT_SIZES = [[1366, 768], [1920, 1080]];
const ALL_SIZES = [[1280, 720], [1366, 768], [1440, 900], [1536, 864], [1920, 1080],
                   [1093, 614], [911, 512]];   // 1366x768 at 125% / 150% browser zoom (1920@125% = 1536x864)
const PASSWORD = 'screen-check-only';
const args = process.argv.slice(2);
const sizes = args.includes('--all-sizes') ? ALL_SIZES : DEFAULT_SIZES;
const scheme = args.includes('--theme') ? args[args.indexOf('--theme') + 1] : 'dark';

function freePort() {
  return new Promise((res) => { const s = net.createServer(); s.listen(0, () => { const p = s.address().port; s.close(() => res(p)); }); });
}
async function waitPort(port, tries = 240) {
  for (let i = 0; i < tries; i++) {
    const ok = await new Promise((res) => { const c = net.connect(port, '127.0.0.1', () => { c.end(); res(true); }); c.on('error', () => res(false)); });
    if (ok) return; await new Promise((r) => setTimeout(r, 500));
  }
  throw new Error('streamlit did not start');
}
async function idle(page) {
  await page.waitForTimeout(700);
  try { await page.waitForSelector('[data-testid="stStatusWidget"]', { state: 'detached', timeout: 20000 }); } catch (e) {}
  await page.waitForTimeout(900);
}
const overflowJs = () => {
  const vw = window.innerWidth; const doc = document.documentElement.scrollWidth; const bad = [];
  for (const el of document.querySelectorAll('body *')) {
    const r = el.getBoundingClientRect();
    if (r.width === 0 || r.height === 0) continue;
    const st = getComputedStyle(el);
    if (st.visibility === 'hidden' || st.display === 'none' || st.position === 'fixed') continue;
    let p = el.parentElement, inScroller = false;
    while (p) { const ps = getComputedStyle(p);
      if (['auto', 'scroll', 'hidden'].includes(ps.overflowX) && p !== document.body && p !== document.documentElement
          && p.scrollWidth > p.clientWidth + 1) { inScroller = true; break; }
      p = p.parentElement; }
    if (inScroller) continue;
    if (r.right > vw + 1) bad.push(String(el.getAttribute('data-testid') || el.className || el.tagName).slice(0, 70) + ' right=' + Math.round(r.right));
  }
  return { vw, doc, bad: bad.slice(0, 6) };
};

(async () => {
  fs.mkdirSync(OUT, { recursive: true });
  const port = await freePort();
  const app = spawn('python3', ['-m', 'streamlit', 'run', 'app/Home.py', '--server.port', String(port), '--server.headless', 'true'],
                    { cwd: REPO, env: { ...process.env, APP_PASSWORD: PASSWORD }, stdio: 'ignore' });
  let failures = 0;
  try {
    await waitPort(port);
    const browser = await chromium.launch();
    for (const [w, h] of sizes) {
      const ctx = await browser.newContext({ viewport: { width: w, height: h }, deviceScaleFactor: 1, colorScheme: scheme });
      const page = await ctx.newPage();
      await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: 'networkidle' });
      await page.getByLabel('Password', { exact: true }).fill(PASSWORD);
      await page.getByRole('button', { name: 'Log in' }).click();
      await page.waitForSelector('text=WP model Brier', { timeout: 60000 });
      await idle(page);
      const nav = page.locator('section[data-testid="stSidebar"] [data-testid="stPageLink"] a');
      let titles = [];
      for (let i = 0; i < await nav.count(); i++) titles.push((await nav.nth(i).innerText()).trim().split('\n').pop().trim());
      if (!titles.length) titles = ['Home'];
      for (const title of titles) {
        const link = page.locator('section[data-testid="stSidebar"] [data-testid="stPageLink"] a', { hasText: title }).first();
        if (await link.count() && await link.isVisible()) await link.click();
        await idle(page);
        const slug = title.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/_+$/, '');
        await page.screenshot({ path: path.join(OUT, `${slug}_${w}x${h}_${scheme}.png`) });
        const res = await page.evaluate(overflowJs);
        const errs = await page.locator('[data-testid="stException"]').count();
        const ok = (w < 1024 || (res.doc <= res.vw && res.bad.length === 0)) && errs === 0;
        if (!ok) failures++;
        console.log(`${ok ? 'ok  ' : 'FAIL'} ${w}x${h} ${scheme} ${title.padEnd(26)} scrollWidth=${res.doc} vw=${res.vw}`
                    + (errs ? ` exceptions=${errs}` : '') + (res.bad.length ? ` offenders=${JSON.stringify(res.bad)}` : ''));
      }
      await ctx.close();
    }
    await browser.close();
  } finally { app.kill(); }
  console.log(`\n${failures} failure(s); screenshots in ${OUT}`);
  process.exit(failures ? 1 : 0);
})().catch((e) => { console.error(e); process.exit(2); });
