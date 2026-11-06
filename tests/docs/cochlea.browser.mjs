import assert from 'node:assert/strict';
import { simulate, stimulus, SAMPLE_RATE } from '../../docs/cochlea/simulation.mjs';

export async function checkCochlea(page, url) {
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto(url, { waitUntil: 'networkidle' });
  const widget = page.locator('.cochlea-experiment');
  await widget.waitFor();
  assert.equal(await widget.locator('.oscillators button').count(), 16);
  const value = async name => Number(await widget.locator(`[data-value="${name}"]`).innerText());
  const verify = async (frequency, qualityFactor, correctionRate, burst) => {
    const n = Number(await widget.getAttribute('data-sample'));
    const trace = simulate(stimulus(frequency, burst), { qualityFactor, correctionRate });
    for (const [name, expected] of [['input', trace.raw[n]], ['prediction', trace.prediction[n]], ['error', trace.error[n]]]) {
      assert.ok(Math.abs(await value(name) - expected) <= 0.000051, `${name} does not match sample ${n}`);
    }
  };
  await verify(371, 16, .5, true);
  const prediction = await value('prediction');
  await widget.getByRole('button', { name: 'Oscillator 12, 2343 Hz', exact: true }).click();
  assert.equal(await value('prediction'), prediction, 'Selecting a view must not change inference');
  assert.match(await widget.locator('.node-detail').innerText(), /Node 12/);
  await widget.getByRole('button', { name: 'Tune input to node 12', exact: true }).click();
  await verify(2343, 16, .5, true);

  const q = widget.locator('#cochlea-q');
  await q.focus();
  await q.press('End');
  await verify(2343, 32, .5, true);
  await q.press('Home');
  await verify(2343, 2, .5, true);
  const correction = widget.locator('#cochlea-correction');
  await correction.focus();
  await correction.press('Home');
  await verify(2343, 2, 0, true);
  assert.equal(await value('prediction'), 0);
  assert.equal(await value('error'), await value('input'));
  assert.match(await widget.locator('.explanation').innerText(), /stays at rest/);
  await correction.press('End');
  await verify(2343, 2, 1, true);
  await widget.getByRole('button', { name: 'After sound', exact: true }).click();
  assert.equal(await value('input'), 0);
  assert.match(await widget.locator('.explanation').innerText(), /not free decay/);
  await verify(2343, 2, 1, true);
  await widget.locator('#cochlea-input').selectOption('tone');
  await verify(2343, 2, 1, false);

  const frequency = widget.locator('#cochlea-frequency');
  await frequency.focus();
  await frequency.press('End');
  await verify(8000, 2, 1, false);
  await frequency.press('Home');
  await verify(80, 2, 1, false);
  await widget.getByRole('button', { name: 'Start', exact: true }).click();
  assert.equal(await widget.getAttribute('data-sample'), '0');
  await widget.getByRole('button', { name: 'Play 1000x slower', exact: true }).click();
  await page.waitForFunction(() => Number(document.querySelector('.cochlea-experiment').dataset.sample) > 2);
  await widget.getByRole('button', { name: 'Pause', exact: true }).click();
  const paused = await widget.getAttribute('data-sample');
  await page.waitForTimeout(100);
  assert.equal(await widget.getAttribute('data-sample'), paused);
  await verify(80, 2, 1, false);

  const time = widget.locator('#cochlea-time');
  await time.focus();
  await time.press('End');
  assert.equal(Number(await widget.getAttribute('data-sample')), Math.round(.16 * SAMPLE_RATE) - 1);
  await verify(80, 2, 1, false);
  await page.setViewportSize({ width: 390, height: 900 });
  assert.ok(await widget.evaluate(el => el.scrollWidth <= el.clientWidth + 1), 'Widget overflows on mobile');
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'Page overflows on mobile');
  const viewport = widget.locator('.plot-viewport').first();
  await viewport.evaluate(el => { el.scrollLeft = el.scrollWidth - el.clientWidth; });
  assert.ok(await viewport.evaluate(el => el.scrollLeft > 0), 'Mobile plots cannot be panned');
  assert.deepEqual(errors, []);
}
