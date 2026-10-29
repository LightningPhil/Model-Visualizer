import { expect, test, type Page } from '@playwright/test'

const READING_KEY = 'model-visualizer.wiki-reading.v1'
async function openWiki(page: Page, title?: string) {
  await page.goto('/')
  await page.getByRole('button', { name: 'Open model wiki', exact: true }).click()
  const wiki = page.getByRole('dialog', { name: 'The model wiki', exact: true })
  if (title)
    await wiki
      .getByRole('navigation', { name: 'Wiki topics' })
      .getByRole('button', { name: title, exact: true })
      .click()
  return wiki
}

test('worked attention recalculates visibility, probabilities and values without changing the model', async ({
  page,
}) => {
  const wiki = await openWiki(page, 'What self-attention computes')
  await wiki.getByRole('button', { name: 'Explore a worked example', exact: true }).click()
  const example = wiki.getByRole('region', { name: 'Interactive worked example' })
  await expect(example).toContainText('SYNTHETIC')
  await example.getByLabel('Example query position', { exact: true }).fill('0')
  await expect(example.locator('tr[data-visible="false"]')).toHaveCount(2)
  await expect(example.locator('tbody tr').first()).toContainText('100%')
  await example.getByLabel('Apply the causal mask').uncheck()
  await expect(example.locator('tr[data-visible="false"]')).toHaveCount(0)
  await expect(example).toContainText('Noncausal comparison')
  await example.getByRole('button', { name: 'Reset worked example' }).click()
  await expect(example.getByLabel('Example query position', { exact: true })).toHaveValue('1')
  await expect(example.locator('tr[data-visible="false"]')).toHaveCount(1)
  const saved = await page.evaluate(() =>
    JSON.parse(localStorage.getItem('model-visualizer.workspace.v1')!),
  )
  expect(saved.config.attention).toBe('GQA')
  expect(saved.ui.prompt).toBe('The future is beautifully complex.')
  await page.screenshot({ path: 'docs/images/wiki-attention-worked.png' })
})

test('rotary geometry and reasoning questions are usable in a narrow light reader', async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await page.goto('/')
  await page.getByRole('button', { name: 'Switch to light mode' }).click()
  await page.getByRole('button', { name: 'Open model wiki', exact: true }).click()
  const wiki = page.getByRole('dialog', { name: 'The model wiki', exact: true })
  await wiki.getByRole('button', { name: 'Browse wiki topics' }).click()
  await wiki
    .getByRole('navigation', { name: 'Wiki topics' })
    .getByRole('button', { name: 'RoPE: position as a rotation', exact: true })
    .click()
  await wiki.getByRole('button', { name: 'Explore a worked example', exact: true }).click()
  const example = wiki.getByRole('region', { name: 'Interactive worked example' })
  await example.getByLabel('Example absolute position', { exact: true }).fill('0')
  await expect(example.getByRole('img')).toHaveAttribute('aria-label', /rotated vector \[1, 0.5\]/)
  await example.getByLabel('Example absolute position', { exact: true }).fill('8')
  await example.getByLabel('Example RoPE pair index', { exact: true }).fill('1')
  await expect(example).toContainText('0.08 radians')
  expect(await wiki.evaluate((el) => el.scrollWidth <= el.clientWidth)).toBe(true)
  await page.screenshot({ path: 'docs/images/wiki-rotation-mobile.png' })
  await wiki.locator('.wiki-reasoning summary').first().click()
  await expect(wiki.locator('.wiki-reasoning details[open]')).toContainText(
    'absolute sequence position',
  )
  await expect(wiki.locator('.katex-error')).toHaveCount(0)
})

test('glossary exposes active symbols and typo suggestions remain explicit', async ({ page }) => {
  const wiki = await openWiki(page)
  await wiki.getByRole('button', { name: 'Glossary & current symbols', exact: true }).click()
  await wiki.getByLabel('Search glossary').fill('Hkv')
  await expect(wiki.locator('.wiki-glossary-list > div')).toHaveCount(1)
  await expect(wiki.locator('.wiki-glossary-list')).toContainText('Current study: 2')
  await wiki
    .locator('.wiki-glossary-list')
    .getByRole('button', { name: 'Read the mechanism', exact: true })
    .click()
  await expect(wiki.locator('.wiki-article h3')).toHaveText('MHA, GQA and MQA')
  await wiki.getByRole('textbox', { name: 'Search the model wiki' }).fill('softmx')
  await expect(wiki.locator('.wiki-search-suggestions')).toContainText('softmax')
  await wiki
    .locator('.wiki-search-suggestions')
    .getByRole('button', { name: 'softmax', exact: true })
    .click()
  await expect(wiki.locator('.wiki-results')).toContainText(
    'Softmax and stable probability calculations',
  )
  await wiki.getByLabel('Wiki collection').selectOption('Glossary')
  await expect(wiki.locator('.wiki-results > button').first()).toContainText('GLOSSARY')
})

test('bookmarks and reading positions survive closing and reload, separately from workspace data', async ({
  page,
}) => {
  const wiki = await openWiki(page, 'RoPE: position as a rotation')
  await wiki.getByRole('button', { name: 'Save article', exact: true }).click()
  await expect(wiki.getByRole('button', { name: 'Unsave article', exact: true })).toHaveAttribute(
    'aria-pressed',
    'true',
  )
  await wiki.getByRole('button', { name: 'Explore a worked example', exact: true }).click()
  await expect
    .poll(() =>
      page.evaluate(
        (key) => JSON.parse(localStorage.getItem(key)!).bookmarks[0].scroll,
        READING_KEY,
      ),
    )
    .toBeGreaterThan(200)
  await page.keyboard.press('Escape')
  await page.reload()
  await page.getByRole('button', { name: 'Open model wiki', exact: true }).click()
  await wiki.getByRole('button', { name: /Saved & recent reading/ }).click()
  await wiki
    .locator('.wiki-saved-list')
    .getByRole('button', {
      name: 'RoPE: position as a rotation Resume at your reading position',
      exact: true,
    })
    .click()
  await expect(wiki.locator('.wiki-article h3')).toHaveText('RoPE: position as a rotation')
  await expect(wiki.getByRole('tab', { name: 'Interactive lab', exact: true })).toHaveAttribute(
    'aria-selected',
    'true',
  )
  await expect
    .poll(() => wiki.locator('.wiki-reading-pane').evaluate((el) => el.scrollTop))
    .toBeGreaterThan(200)
  const workspace = await page.evaluate(() =>
    JSON.parse(localStorage.getItem('model-visualizer.workspace.v1')!),
  )
  expect(workspace.extensions?.[READING_KEY]).toBeUndefined()
  expect(workspace.ui.notes).not.toContain('RoPE: position as a rotation')
})

test('distinct reading depths retain the experiment and support keyboard navigation', async ({
  page,
}) => {
  const wiki = await openWiki(page, 'RMSNorm, gain and epsilon')
  const intro = wiki.getByRole('tab', { name: 'Introduction', exact: true })
  await expect(intro).toHaveAttribute('aria-selected', 'true')
  await expect(wiki.getByRole('tabpanel')).toContainText('Another way to picture it')
  await expect(wiki.getByRole('tabpanel').locator('.katex')).toHaveCount(0)
  await intro.focus()
  await page.keyboard.press('ArrowRight')
  await expect(wiki.getByRole('tab', { name: 'Technical detail', exact: true })).toBeFocused()
  await expect(wiki.getByRole('tabpanel').locator('.katex').first()).toBeVisible()
  await page.screenshot({ path: 'docs/images/wiki-technical-depth.png' })
  await page.keyboard.press('End')
  const example = wiki.getByRole('region', { name: 'Interactive worked example' })
  await example.getByLabel('Set Example RMS input 0', { exact: true }).fill('2.5')
  await intro.click()
  await expect(example).not.toBeVisible()
  await wiki.getByRole('tab', { name: 'Interactive lab', exact: true }).click()
  await expect(example.getByLabel('Example RMS input 0', { exact: true })).toHaveValue('2.5')
})

test('attention interventions compare a pinned baseline and cache policies recalculate retained state', async ({
  page,
}) => {
  const wiki = await openWiki(page, 'What self-attention computes')
  await wiki.getByRole('tab', { name: 'Interactive lab', exact: true }).click()
  const example = wiki.getByRole('region', { name: 'Interactive worked example' })
  await example.getByRole('button', { name: 'Pin current result', exact: true }).click()
  await example.getByText('Edit a key or value vector', { exact: true }).click()
  await example.getByLabel('Set Example value feature 0', { exact: true }).fill('4')
  await expect(
    example.locator('.wiki-pin-comparison').filter({ hasText: 'Probability changes' }),
  ).toContainText('[0, 0, 0]')
  await expect(
    example.locator('.wiki-calculation').filter({ hasText: 'Output changes since pin' }),
  ).not.toContainText('[0, 0]')
  await wiki
    .getByRole('navigation', { name: 'Wiki topics' })
    .getByRole('button', { name: 'Calculate logical KV memory', exact: true })
    .click()
  await wiki.getByRole('tab', { name: 'Interactive lab', exact: true }).click()
  await example.getByLabel('Set Scratch processed extent S', { exact: true }).fill('12')
  await example
    .getByLabel('Scratch cache retention', { exact: true })
    .selectOption('sliding-window')
  await example.getByLabel('Set Scratch retention window', { exact: true }).fill('3')
  await expect(example.locator('.wiki-position-strip')).toHaveText('91011')
  await expect(
    example.locator('.wiki-calculation').filter({ hasText: 'After this call' }),
  ).toContainText('3')
  await example.getByLabel('Scratch scalar bytes', { exact: true }).selectOption('1')
  await expect(
    example.locator('.wiki-calculation').filter({ hasText: 'Logical persistent' }),
  ).toContainText('4.5 KiB')
})

test('a concept leads to its actual model operation and launches its separate numerical example', async ({
  page,
}) => {
  const wiki = await openWiki(page, 'RoPE: position as a rotation')
  await wiki.locator('.wiki-model-connections summary').click()
  await wiki.getByRole('button', { name: 'Show RoPE · queries on canvas', exact: true }).click()
  await expect(page.getByRole('heading', { level: 1 })).toHaveText(
    'Attention, operation by operation.',
  )
  await expect(page.getByRole('heading', { name: 'RoPE · queries', exact: true })).toBeVisible()
  await page.getByRole('button', { name: 'Explain this in the wiki', exact: true }).click()
  await wiki
    .locator('.wiki-deeper')
    .getByRole('button', { name: /RoPE: position as a rotation/ })
    .click()
  await wiki.getByRole('button', { name: 'Open numerical example', exact: true }).click()
  const lab = page.getByRole('dialog', { name: 'The mathematics, alive.', exact: true })
  await expect(lab.locator('.numerical-step-card h3')).toHaveText('Rotate query feature pairs', {
    timeout: 20000,
  })
  await expect(lab.locator('.numerical-badge')).toContainText('UNTRAINED')
  await lab.getByRole('button', { name: 'Explain this', exact: true }).click()
  await wiki
    .getByRole('navigation', { name: 'Wiki topics' })
    .getByRole('button', { name: 'Calculate logical KV memory', exact: true })
    .click()
  await wiki.getByRole('button', { name: 'Explore a worked example', exact: true }).click()
  await expect(wiki.getByRole('region', { name: 'Interactive worked example' })).toContainText(
    '2 layers, Hkv = 2, Dh = 4',
  )
  await wiki.getByRole('button', { name: 'Inspect in this run', exact: true }).click()
  await expect(lab.locator('.numerical-step-card h3')).toHaveText(
    'Retain K/V; keep working keys available',
  )
})

test('unknown future reading-library data is preserved', async ({ page }) => {
  const future = '{"version":99,"keep":{"private":"metadata"}}'
  await page.addInitScript(({ key, value }) => localStorage.setItem(key, value), {
    key: READING_KEY,
    value: future,
  })
  const wiki = await openWiki(page, 'RMSNorm, gain and epsilon')
  await wiki.getByRole('button', { name: 'Save article', exact: true }).click()
  await wiki.getByRole('button', { name: /Saved & recent reading/ }).click()
  await expect(wiki.getByRole('status')).toContainText('newer reading-library version')
  expect(await page.evaluate((key) => localStorage.getItem(key), READING_KEY)).toBe(future)
})

test('projection, residual and embedding explorers expose editable operands', async ({ page }) => {
  const wiki = await openWiki(page, 'Linear projections and matrix multiplication')
  const topics = wiki.getByRole('navigation', { name: 'Wiki topics' })
  const example = wiki.getByRole('region', { name: 'Interactive worked example' })
  await wiki.getByRole('tab', { name: 'Interactive lab', exact: true }).click()
  await expect(
    example.locator('.wiki-calculation').filter({ hasText: 'Output feature 0' }),
  ).toContainText('6 + -5 = 1')
  await example.getByRole('button', { name: 'Edit weight row 1, column 0', exact: true }).click()
  await example.getByLabel('Set Selected projection weight', { exact: true }).fill('2')
  await expect(
    example.locator('.wiki-calculation').filter({ hasText: 'Output feature 0' }),
  ).toContainText('6 + -2 = 4')
  await topics
    .getByRole('button', { name: 'Residual stream and decoder blocks', exact: true })
    .click()
  await wiki.getByRole('tab', { name: 'Interactive lab', exact: true }).click()
  await expect(example.locator('.wiki-residual-path')).toContainText('1.7')
  await example.getByLabel('Include the identity paths').uncheck()
  await expect(example.locator('.wiki-residual-path')).toContainText('Block output0.2')
  await topics
    .getByRole('button', { name: 'Embedding lookup and vocabulary identity', exact: true })
    .click()
  await wiki.getByRole('tab', { name: 'Interactive lab', exact: true }).click()
  await example.getByLabel('Set Toy vocabulary ID', { exact: true }).fill('1.5')
  await example.getByLabel('Set Toy vocabulary ID', { exact: true }).blur()
  await expect(example.getByLabel('Toy vocabulary ID', { exact: true })).toHaveValue('1')
  await example
    .getByRole('button', { name: 'Gather fictional embedding ID 2', exact: true })
    .click()
  await example.getByLabel('Set Selected embedding value', { exact: true }).fill('1.5')
  await expect(example.locator('.wiki-calculation')).toContainText('[1.5, 0.3, 0.1]')
  await page.screenshot({ path: 'docs/images/wiki-embedding-explorer.png' })
})

test('normalization, gates, head mappings and cache scratch controls expose their actual arithmetic', async ({
  page,
}) => {
  const wiki = await openWiki(page, 'RMSNorm, gain and epsilon')
  const topics = wiki.getByRole('navigation', { name: 'Wiki topics' })
  const example = wiki.getByRole('region', { name: 'Interactive worked example' })
  await wiki.getByRole('button', { name: 'Explore a worked example', exact: true }).click()
  await example.getByLabel('Example RMS input 0', { exact: true }).fill('0')
  await example.getByLabel('Example RMS input 1', { exact: true }).fill('0')
  await expect(
    example.locator('.wiki-calculation').filter({ hasText: '4 · Divide' }),
  ).toContainText('[0, 0]')
  await topics
    .getByRole('button', { name: 'SwiGLU and position-wise feed-forward networks', exact: true })
    .click()
  await wiki.getByRole('button', { name: 'Explore a worked example', exact: true }).click()
  await example.getByLabel('Example gate preactivation', { exact: true }).fill('-1')
  await example.getByLabel('Example up-branch feature', { exact: true }).fill('1')
  await expect(
    example.locator('.wiki-calculation').filter({ hasText: 'SiLU(gate)' }),
  ).toContainText('-0.26894')
  await topics.getByRole('button', { name: 'MHA, GQA and MQA', exact: true }).click()
  await wiki.getByRole('button', { name: 'Explore a worked example', exact: true }).click()
  await example.getByLabel('Scratch query head', { exact: true }).fill('7')
  await expect(
    example.getByRole('button', { name: 'Inspect query head 7 mapping', exact: true }),
  ).toHaveAttribute('aria-pressed', 'true')
  await expect(example).toContainText('Q7 reads K1 and V1')
  await topics.getByRole('button', { name: 'Calculate logical KV memory', exact: true }).click()
  await wiki.getByRole('button', { name: 'Explore a worked example', exact: true }).click()
  await example.getByLabel('Scratch processed extent S', { exact: true }).fill('12')
  await example.getByLabel('Scratch query count T', { exact: true }).fill('4')
  await expect(example).toContainText('8 + 4 = 12')
  await expect(example).toContainText('36 KiB')
  await topics
    .getByRole('button', { name: 'Greedy choice, sampling and temperature', exact: true })
    .click()
  await wiki.getByRole('button', { name: 'Explore a worked example', exact: true }).click()
  await example.getByLabel('Example vocabulary temperature', { exact: true }).fill('0.1')
  await expect(
    example.locator('.wiki-calculation').filter({ hasText: 'Candidate 2' }),
  ).toContainText('99.99546%')
})
