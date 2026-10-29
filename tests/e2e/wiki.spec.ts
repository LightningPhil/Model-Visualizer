import { expect, test } from '@playwright/test'
import { readFileSync } from 'node:fs'

const starter = JSON.parse(
  readFileSync(
    new URL('../../examples/transformer-foundations.model.json', import.meta.url),
    'utf8',
  ),
)

test.beforeEach(async ({ page }) => {
  await page.goto('/')
})

test('opens a connected wiki, follows concepts and retraces reading history', async ({ page }) => {
  await page.getByRole('button', { name: 'Explain this in the wiki', exact: true }).click()
  const wiki = page.getByRole('dialog', { name: 'The model wiki', exact: true })
  await expect(wiki.locator('.wiki-article h3')).toHaveText('Transformer blocks')
  await expect(wiki.locator('.wiki-document-context')).toContainText('overview / decoder')
  await wiki
    .locator('.wiki-related')
    .getByRole('button', { name: 'Residual stream and decoder blocks', exact: true })
    .click()
  await expect(wiki.locator('.wiki-article h3')).toHaveText('Residual stream and decoder blocks')
  await wiki.getByRole('tab', { name: 'Technical detail', exact: true }).click()
  await expect(wiki.locator('.katex').first()).toBeVisible()
  await wiki.getByRole('button', { name: 'Wiki back', exact: true }).click()
  await expect(wiki.locator('.wiki-article h3')).toHaveText('Transformer blocks')
  await wiki.getByRole('button', { name: 'Wiki forward', exact: true }).click()
  await expect(wiki.locator('.wiki-article h3')).toHaveText('Residual stream and decoder blocks')
  await wiki.getByRole('button', { name: 'Wiki home', exact: true }).click()
  await expect(wiki.locator('.wiki-paths > section')).toHaveCount(6)
  await page.keyboard.press('/')
  await expect(wiki.getByRole('textbox', { name: 'Search the model wiki' })).toBeFocused()
  await expect(page.getByRole('dialog', { name: 'Find a component.' })).toHaveCount(0)
  await wiki.getByRole('textbox', { name: 'Search the model wiki' }).fill('root mean square')
  await wiki
    .locator('.wiki-results > button')
    .filter({ hasText: 'RMSNorm, gain and epsilon' })
    .click()
  await expect(wiki.locator('.wiki-article h3')).toHaveText('RMSNorm, gain and epsilon')
  await expect(wiki.locator('.wiki-references a').first()).toHaveAttribute(
    'href',
    'https://arxiv.org/abs/1910.07467',
  )
  await page.keyboard.press('Escape')
  await expect(wiki).not.toBeVisible()
  await expect(page.getByRole('heading', { level: 1 })).toHaveText('The architecture, unfolded.')
})

test('searches authored operations and returns to the selected connection on its canvas', async ({
  page,
}) => {
  await page.getByRole('button', { name: 'Open model wiki', exact: true }).click()
  const wiki = page.getByRole('dialog', { name: 'The model wiki', exact: true })
  await wiki.getByRole('textbox', { name: 'Search the model wiki' }).fill('residual')
  await wiki.getByLabel('Wiki collection').selectOption('Model document')
  const result = wiki
    .locator('.wiki-results > button')
    .filter({ hasText: 'block · edge ·' })
    .first()
  await result.click()
  await expect(wiki.locator('.wiki-document-context')).toContainText('block /')
  await wiki.getByRole('button', { name: 'View on canvas', exact: true }).click()
  await expect(wiki).not.toBeVisible()
  await expect(page.getByRole('heading', { level: 1 })).toHaveText('Inside a transformer block.')
  await expect(page.locator('.connection-path')).toBeVisible()
  await page.getByRole('button', { name: 'Explain this in the wiki', exact: true }).click()
  await expect(wiki.locator('.wiki-document-context')).toContainText('block /')
})

test('searches imported explanations and preserves notes, metadata and geometry', async ({
  page,
}) => {
  const imported = structuredClone(starter)
  imported.graphs[0].nodes[0].description = 'customwikiword is an authored tokenizer explanation.'
  imported.ui.notes = 'A note the wiki must preserve.'
  imported.extensions = { ...imported.extensions, 'org.example.wiki-test': { retain: true } }
  await page.getByLabel('Open model JSON').setInputFiles({
    name: 'wiki.model.json',
    mimeType: 'application/json',
    buffer: Buffer.from(JSON.stringify(imported)),
  })
  await expect(page.getByRole('status')).toContainText('Opened')
  await expect
    .poll(() =>
      page.evaluate(
        () => JSON.parse(localStorage.getItem('model-visualizer.workspace.v1') ?? '{}').ui?.notes,
      ),
    )
    .toBe(imported.ui.notes)
  const before = await page.evaluate(() => localStorage.getItem('model-visualizer.workspace.v1'))
  await page.getByRole('button', { name: 'Open model wiki', exact: true }).click()
  const wiki = page.getByRole('dialog', { name: 'The model wiki', exact: true })
  await wiki.getByRole('textbox', { name: 'Search the model wiki' }).fill('customwikiword')
  await expect(wiki.locator('.wiki-results > button')).toHaveCount(1)
  await wiki.locator('.wiki-results > button').click()
  await expect(wiki.locator('.wiki-article')).toContainText(
    'customwikiword is an authored tokenizer explanation.',
  )
  await page.keyboard.press('Escape')
  const after = await page.evaluate(() => localStorage.getItem('model-visualizer.workspace.v1'))
  expect(after).toBe(before)
})

test('fits both themes on narrow screens and allows keyboard search and topic navigation', async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await page.getByRole('button', { name: 'Switch to light mode' }).click()
  await page.getByRole('button', { name: 'Open model wiki', exact: true }).click()
  const wiki = page.getByRole('dialog', { name: 'The model wiki', exact: true })
  await expect(wiki.locator('.wiki-home')).toBeVisible()
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(390)
  expect(await wiki.evaluate((element) => element.scrollWidth <= element.clientWidth)).toBe(true)
  await wiki.getByRole('button', { name: 'Browse wiki topics' }).click()
  await wiki
    .getByRole('navigation', { name: 'Wiki topics' })
    .getByRole('button', { name: 'RoPE: position as a rotation', exact: true })
    .click()
  await expect(wiki.locator('.wiki-article h3')).toHaveText('RoPE: position as a rotation')
  await expect(wiki.locator('.katex-error')).toHaveCount(0)
  await wiki
    .getByRole('textbox', { name: 'Search the model wiki' })
    .fill('absolutelymissingterm123')
  await expect(wiki.locator('.wiki-empty')).toBeVisible()
  await wiki.getByRole('button', { name: 'Clear filters' }).click()
  await expect(wiki.locator('.wiki-article h3')).toHaveText('RoPE: position as a rotation')
  await page.screenshot({ path: 'docs/images/model-wiki-mobile.png' })
  await page.keyboard.press('Escape')
  await expect(wiki).not.toBeVisible()
})

test('opens help above the numerical workbench and returns without replacing the study', async ({
  page,
}) => {
  await page.getByRole('button', { name: /Tensor laboratory/ }).click()
  const lab = page.getByRole('dialog', { name: 'The mathematics, alive.', exact: true })
  await expect(lab.getByRole('button', { name: 'Explain this', exact: true })).toBeVisible({
    timeout: 20000,
  })
  await lab.getByRole('button', { name: 'Explain this', exact: true }).click()
  const wiki = page.getByRole('dialog', { name: 'The model wiki', exact: true })
  await expect(wiki.locator('.wiki-mode')).toContainText('untrained parameters')
  await expect(wiki.locator('.wiki-document-context')).toBeVisible()
  const step = await lab.locator('.numerical-step-card h3').textContent()
  await wiki
    .getByRole('navigation', { name: 'Wiki topics' })
    .getByRole('button', { name: 'Softmax and stable probability calculations', exact: true })
    .click()
  await expect(wiki.locator('.wiki-article h3')).toHaveText(
    'Softmax and stable probability calculations',
  )
  await page.keyboard.press('ArrowRight')
  await page.keyboard.press('Space')
  await page.keyboard.press('Escape')
  await expect(wiki).not.toBeVisible()
  await expect(lab).toBeVisible()
  await expect(lab.locator('.numerical-badge')).toContainText('UNTRAINED')
  await expect(lab.locator('.numerical-step-card h3')).toHaveText(step!)
  await lab
    .getByRole('button', { name: 'Where does a probability come from?', exact: true })
    .click()
  await lab.getByRole('button', { name: 'Explain this', exact: true }).click()
  await wiki.getByRole('textbox', { name: 'Search the model wiki' }).fill('Query projection')
  await wiki.getByLabel('Wiki collection').selectOption('Model document')
  await wiki
    .locator('.wiki-results > button')
    .filter({ hasText: /node · layer-0\.attn\.q$/ })
    .click()
  await wiki.getByRole('button', { name: 'View on canvas', exact: true }).click()
  await expect(lab.locator('.numeric-current-tensor')).toContainText('Split head axis')
})

test('all concept equations render and article links stay within the bundled wiki', async ({
  page,
}) => {
  test.setTimeout(60000)
  const errors: string[] = []
  page.on('pageerror', (error) => errors.push(error.message))
  await page.getByRole('button', { name: 'Open model wiki', exact: true }).click()
  const wiki = page.getByRole('dialog', { name: 'The model wiki', exact: true })
  await expect(wiki.locator('.wiki-nav-group > button')).toHaveCount(49)
  const topics = await wiki.locator('.wiki-nav-group > button').allTextContents()
  for (const topic of topics) {
    await wiki
      .getByRole('navigation', { name: 'Wiki topics' })
      .getByRole('button', { name: topic, exact: true })
      .click()
    await expect(wiki.locator('.wiki-article h3')).toHaveText(topic)
    await wiki.getByRole('tab', { name: 'Introduction', exact: true }).click()
    await expect(wiki.getByRole('tabpanel').locator('.wiki-section')).toHaveCount(4)
    await expect(wiki.getByRole('tabpanel').locator('.katex:visible')).toHaveCount(0)
    await wiki.getByRole('tab', { name: 'Technical detail', exact: true }).click()
    await expect(wiki.getByRole('tab', { name: 'Technical detail', exact: true })).toHaveAttribute(
      'aria-selected',
      'true',
    )
    await expect(wiki.locator('.katex-error')).toHaveCount(0)
  }
  await wiki
    .getByRole('navigation', { name: 'Wiki topics' })
    .getByRole('button', { name: 'What self-attention computes', exact: true })
    .click()
  await wiki.getByRole('tab', { name: 'Technical detail', exact: true }).click()
  await wiki.locator('.wiki-article .wiki-inline-link').filter({ hasText: 'RoPE' }).first().click()
  await expect(wiki.locator('.wiki-article h3')).toHaveText('RoPE: position as a rotation')
  await page.screenshot({ path: 'docs/images/model-wiki-dark.png' })
  expect(errors).toEqual([])
})

test('simulation help follows the current step and preserves paused playback', async ({ page }) => {
  await page.getByRole('button', { name: /Run simulation/ }).click()
  const simulator = page.getByRole('complementary', { name: 'Execution simulator' })
  await expect(
    simulator.getByRole('button', { name: 'Explain this step', exact: true }),
  ).toBeVisible({ timeout: 20000 })
  const title = await simulator.locator('.execution-card h3').textContent()
  await simulator.getByRole('button', { name: 'Explain this step', exact: true }).click()
  const wiki = page.getByRole('dialog', { name: 'The model wiki', exact: true })
  await expect(wiki.locator('.wiki-mode')).toContainText('scripted choices')
  await expect(wiki.locator('.wiki-document-context')).toContainText('whole-model /')
  await page.keyboard.press('Escape')
  await expect(wiki).not.toBeVisible()
  await expect(simulator).toBeVisible()
  await expect(simulator.locator('.execution-card h3')).toHaveText(title!)
})

test('every guide keeps introductory and technical reading contained on a narrow light screen', async ({
  page,
}) => {
  test.setTimeout(120000)
  await page.setViewportSize({ width: 390, height: 844 })
  await page.getByRole('button', { name: 'Switch to light mode' }).click()
  await page.getByRole('button', { name: 'Open model wiki', exact: true }).click()
  const wiki = page.getByRole('dialog', { name: 'The model wiki', exact: true })
  await expect(wiki.locator('.wiki-nav-group > button')).toHaveCount(49)
  const topics = await wiki.locator('.wiki-nav-group > button').allTextContents()
  for (const topic of topics) {
    await wiki.getByRole('button', { name: 'Browse wiki topics', exact: true }).click()
    await wiki
      .getByRole('navigation', { name: 'Wiki topics' })
      .getByRole('button', { name: topic, exact: true })
      .click()
    await wiki.getByRole('tab', { name: 'Introduction', exact: true }).click()
    await expect(wiki.getByRole('tabpanel').locator('.katex:visible')).toHaveCount(0)
    await wiki.getByRole('tab', { name: 'Technical detail', exact: true }).click()
    await expect(wiki.locator('.katex-error')).toHaveCount(0)
    expect(
      await wiki.locator('.wiki-article').evaluate((e) => e.scrollWidth <= e.clientWidth),
      topic,
    ).toBe(true)
    expect(await wiki.evaluate((e) => e.scrollWidth <= e.clientWidth), topic).toBe(true)
  }
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(390)
  await page.screenshot({ path: 'docs/images/wiki-editorial-review-mobile.png' })
})

test('trained-model help uses the actual runtime graph and retains the captured study', async ({
  page,
}) => {
  test.setTimeout(90000)
  await page.getByRole('button', { name: /Local trained model/ }).click()
  await page.getByLabel('Local model output budget').selectOption('1')
  await page.getByRole('button', { name: /Load trained model/ }).click()
  await expect(page.getByRole('button', { name: 'Generate & inspect' })).toBeEnabled({
    timeout: 30000,
  })
  await page.getByRole('button', { name: 'Generate & inspect' }).click()
  const lab = page.getByRole('dialog', { name: 'The mathematics, alive.', exact: true })
  await expect(lab.getByRole('button', { name: 'Explain this', exact: true })).toBeVisible({
    timeout: 45000,
  })
  await lab
    .getByRole('button', { name: 'Where does a probability come from?', exact: true })
    .click()
  await lab.getByRole('button', { name: 'Explain this', exact: true }).click()
  const wiki = page.getByRole('dialog', { name: 'The model wiki', exact: true })
  await expect(wiki.locator('.wiki-mode')).toContainText('actual learned weights')
  await expect(wiki.locator('.wiki-document-context')).toContainText('softmax-0')
  await expect(wiki.locator('.wiki-tensors')).toContainText('Current study:')
  await wiki.getByRole('button', { name: 'View on canvas', exact: true }).click()
  await expect(wiki).not.toBeVisible()
  await expect(lab.locator('.numerical-badge')).toContainText('TRAINED')
  await expect(lab.locator('.numeric-current-tensor')).toContainText('attention probabilities')
})
