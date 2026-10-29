import { test, expect, type Page } from '@playwright/test'

async function open(page: Page, topic: string) {
  await page.goto('/')
  await page.getByRole('button', { name: 'Open model wiki', exact: true }).click()
  const wiki = page.getByRole('dialog', { name: 'The model wiki', exact: true })
  await wiki
    .getByRole('navigation', { name: 'Wiki topics' })
    .getByRole('button', { name: topic, exact: true })
    .click()
  await wiki.getByRole('tab', { name: 'Interactive lab', exact: true }).click()
  return wiki
}
test('a local learning trajectory exposes gradients, simultaneous updates and overshooting', async ({
  page,
}) => {
  const wiki = await open(page, 'How a model learns: gradients and updates')
  await expect(
    wiki.locator('.wiki-calculation').filter({ hasText: 'Backward gradients' }),
  ).toContainText('dL/dw = -6; dL/db = -3')
  await wiki.getByRole('button', { name: 'Take one SGD step', exact: true }).click()
  await expect(
    wiki.locator('.wiki-calculation').filter({ hasText: 'Current parameters' }),
  ).toContainText('w = 0.6, b = 0.3')
  await expect(wiki.locator('.wiki-lab-history')).toContainText('4.5 → 1.125')
  await wiki.getByLabel('Set SGD learning rate', { exact: true }).fill('1')
  await expect(
    wiki.locator('.wiki-calculation').filter({ hasText: 'Current parameters' }),
  ).toContainText('after 0 steps')
  await expect(wiki.locator('.wiki-scratch-insight')).toContainText('overshoots')
  await page.screenshot({ path: 'docs/images/wiki-learning-loop.png' })
})
test('position methods isolate absolute and relative effects and keep the causal mask', async ({
  page,
}) => {
  const wiki = await open(page, 'Compare positional encoding methods')
  const rotary = wiki.locator('.wiki-calculation').filter({ hasText: 'RoPE pair rotation' })
  const absolute = wiki
    .locator('.wiki-calculation')
    .filter({ hasText: 'Fictional learned absolute addition' })
  const r = await rotary.textContent(),
    a = await absolute.textContent()
  await wiki.getByLabel('Set Comparison query position', { exact: true }).fill('2')
  await wiki.getByLabel('Set Comparison key position', { exact: true }).fill('1')
  await expect(rotary).toHaveText(r!)
  expect(await absolute.textContent()).not.toBe(a)
  await wiki.getByLabel('Set Comparison key position', { exact: true }).fill('3')
  await expect(wiki.locator('.wiki-scratch-insight')).toContainText('ALL methods exclude this key')
})
test('feature coordinates rotate while the paired readout remains fixed', async ({ page }) => {
  const wiki = await open(page, 'Features, neurons and superposition')
  const readout = wiki.locator('.wiki-calculation').filter({ hasText: 'F0 dot-product readout' })
  const coords = wiki
    .locator('.wiki-calculation')
    .filter({ hasText: 'Displayed hidden coordinates' })
  await expect(readout).toContainText('1.25')
  const before = await coords.textContent()
  await wiki.getByLabel('Set Feature view rotation', { exact: true }).fill('90')
  await expect(readout).toContainText('1.25')
  expect(await coords.textContent()).not.toBe(before)
  await expect(
    wiki.getByRole('img', { name: 'Synthetic feature directions and their summed hidden vector' }),
  ).toBeVisible()
})
test('ablation and patching report explicit output changes against both baselines', async ({
  page,
}) => {
  const wiki = await open(page, 'Test a mechanism: ablation and activation patching')
  const output = wiki.locator('.wiki-calculation').filter({ hasText: 'Recomputed output logit' })
  await wiki.getByRole('button', { name: 'Ablate negative path', exact: true }).click()
  await expect(output).toContainText('3')
  await wiki.getByRole('button', { name: 'Patch clean first coordinate', exact: true }).click()
  await expect(output).toContainText('1')
  await expect(
    wiki.locator('.wiki-calculation').filter({ hasText: 'change from altered baseline' }),
  ).toContainText('3')
  await wiki.getByRole('button', { name: 'Replace first coordinate', exact: true }).click()
  await wiki.getByLabel('Set Replacement first coordinate', { exact: true }).fill('-1')
  await expect(output).toContainText('-5')
})
test('preference loss changes independently from the current policy and demonstration objective', async ({
  page,
}) => {
  const wiki = await open(page, 'From pretraining to assistant behaviour')
  const probability = wiki.locator('.wiki-calculation').filter({ hasText: 'Policy probability' })
  const demonstration = wiki
    .locator('.wiki-calculation')
    .filter({ hasText: 'Demonstration negative' })
  const before = await demonstration.textContent()
  await wiki.getByLabel('Set Reference completion log-probability gap', { exact: true }).fill('2')
  await expect(probability).toContainText('0.5')
  await expect(demonstration).toHaveText(before!)
  await expect(
    wiki.locator('.wiki-calculation').filter({ hasText: 'DPO pair loss' }),
  ).toContainText('1.313262')
})
test('precision lab labels nonfinite values and demonstrates order-sensitive binary32 arithmetic', async ({
  page,
}) => {
  const wiki = await open(page, 'Floating-point precision and numerical stability')
  await expect(
    wiki.locator('.wiki-calculation').filter({ hasText: 'Naive normalized' }),
  ).toContainText('[NaN, NaN]')
  await expect(
    wiki.locator('.wiki-calculation').filter({ hasText: 'cancel first:' }),
  ).toContainText('1')
  const stable = wiki.locator('.wiki-calculation').filter({ hasText: 'Max-shifted stable' })
  const before = await stable.textContent()
  await wiki.getByLabel('Set Large softmax logit', { exact: true }).fill('0')
  await expect(stable).toHaveText(before!)
  await expect(
    wiki.locator('.wiki-calculation').filter({ hasText: 'Naive normalized' }),
  ).not.toContainText('NaN')
})
test('existing normalization and expert guides expose their new comparisons', async ({ page }) => {
  const wiki = await open(page, 'RMSNorm, gain and epsilon')
  const layer = wiki.locator('.wiki-calculation').filter({ hasText: 'LayerNorm comparison output' })
  const rms = wiki.locator('.wiki-calculation').filter({ hasText: 'RMSNorm comparison output' })
  const initialLayer = await layer.textContent(),
    initialRms = await rms.textContent()
  await wiki.getByLabel('Set Normalization common offset', { exact: true }).fill('2')
  await expect(layer).toHaveText(initialLayer!)
  expect(await rms.textContent()).not.toBe(initialRms)
  await wiki.getByLabel('Use a constant feature row').check()
  await expect(layer).toContainText('[0, 0]')
  await wiki
    .getByRole('navigation', { name: 'Wiki topics' })
    .getByRole('button', { name: 'Mixture-of-experts and routed computation', exact: true })
    .click()
  await wiki.getByRole('tab', { name: 'Interactive lab', exact: true }).click()
  await expect(wiki.locator('.wiki-expert-branches [data-selected="true"]')).toHaveCount(1)
  await wiki.getByLabel('Set Expert two router score', { exact: true }).fill('3')
  await expect(wiki.locator('.wiki-expert-branches [data-selected="true"]')).toContainText('E2')
  await wiki.getByLabel('Set Selected expert count', { exact: true }).fill('2')
  await expect(wiki.locator('.wiki-expert-branches [data-selected="true"]')).toHaveCount(2)
  await expect(
    wiki.locator('.wiki-calculation').filter({ hasText: 'Expert-only parameters' }),
  ).toContainText('stored 6; activated 4')
})
test('all new labs fit narrow light layouts, preserve scratch controls across reading depths and leave the workspace intact', async ({
  page,
}) => {
  test.setTimeout(60000)
  await page.setViewportSize({ width: 390, height: 844 })
  await page.goto('/')
  await page.getByRole('button', { name: 'Switch to light mode' }).click()
  await expect
    .poll(() =>
      page.evaluate(
        () => JSON.parse(localStorage.getItem('model-visualizer.workspace.v1') ?? '{}').ui?.theme,
      ),
    )
    .toBe('light')
  const before = await page.evaluate(() => localStorage.getItem('model-visualizer.workspace.v1'))
  await page.getByRole('button', { name: 'Open model wiki', exact: true }).click()
  const wiki = page.getByRole('dialog', { name: 'The model wiki', exact: true })
  for (const topic of [
    'How a model learns: gradients and updates',
    'Compare positional encoding methods',
    'Test a mechanism: ablation and activation patching',
    'From pretraining to assistant behaviour',
    'Floating-point precision and numerical stability',
    'RMSNorm, gain and epsilon',
    'Mixture-of-experts and routed computation',
    'Features, neurons and superposition',
  ]) {
    await wiki.getByRole('button', { name: 'Browse wiki topics' }).click()
    await wiki
      .getByRole('navigation', { name: 'Wiki topics' })
      .getByRole('button', { name: topic, exact: true })
      .click()
    await wiki.getByRole('tab', { name: 'Interactive lab', exact: true }).click()
    await wiki.getByRole('region', { name: 'Interactive worked example' }).scrollIntoViewIfNeeded()
    expect(await wiki.evaluate((e) => e.scrollWidth <= e.clientWidth), topic).toBe(true)
  }
  await wiki.getByLabel('Set Feature view rotation', { exact: true }).fill('90')
  await wiki.getByRole('tab', { name: 'Introduction', exact: true }).click()
  await expect(wiki.getByRole('tabpanel').locator('.katex:visible')).toHaveCount(0)
  await wiki.getByRole('tab', { name: 'Interactive lab', exact: true }).click()
  await expect(wiki.getByLabel('Feature view rotation', { exact: true })).toHaveValue('90')
  await wiki.getByRole('region', { name: 'Interactive worked example' }).scrollIntoViewIfNeeded()
  await page.screenshot({ path: 'docs/images/wiki-features-mobile.png' })
  await page.keyboard.press('Escape')
  expect(await page.evaluate(() => localStorage.getItem('model-visualizer.workspace.v1'))).toBe(
    before,
  )
})
