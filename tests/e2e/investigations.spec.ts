import { expect, test } from '@playwright/test'
import { readFile } from 'node:fs/promises'

test('guided investigation records the exact setup, exports and restores without editing the workspace', async ({
  page,
}) => {
  await page.goto('/')
  await page.getByRole('button', { name: 'Start with a question', exact: true }).click()
  const wiki = page.getByRole('dialog', { name: 'The model wiki', exact: true })
  await wiki.getByLabel('Guided prediction').fill('The probability row will not change.')
  await wiki.getByLabel('Investigation edited feature').fill('3')
  await wiki.getByRole('button', { name: 'Observe the calculation', exact: true }).click()
  await expect(wiki.locator('.study-observation')).toContainText('Δoutput')
  await wiki.getByText('Record this investigation', { exact: true }).click()
  await wiki
    .getByLabel('Investigation conclusion', { exact: true })
    .fill('Changing values changes the weighted read, not the attention row.')
  await wiki.getByRole('button', { name: 'Save investigation', exact: true }).click()
  const library = wiki.getByRole('region', { name: 'Saved investigations', exact: true })
  await expect(library.locator('details')).toHaveCount(1)
  const download = page.waitForEvent('download')
  await library.getByRole('button', { name: 'Export investigations', exact: true }).click()
  const file = await download
  const exported = JSON.parse(await readFile((await file.path())!, 'utf8'))
  expect(exported.records[0].recipe.value).toBe(3)
  expect(exported.records[0].conclusion).toContain('weighted read')
  expect(exported.records[0].evidence.summary).toContain('The probability row will not change.')
  await library.locator('summary').click()
  await library.getByRole('button', { name: 'Reopen setup', exact: true }).click()
  await expect(wiki.getByLabel('Investigation edited feature')).toHaveValue('3')
  await page.reload()
  await page.getByRole('button', { name: 'Open model wiki', exact: true }).click()
  const reopened = page.getByRole('dialog', { name: 'The model wiki', exact: true })
  await reopened.getByRole('button', { name: 'Investigations & records', exact: true }).click()
  await expect(
    reopened.getByRole('region', { name: 'Saved investigations' }).locator('details'),
  ).toHaveCount(1)
  await reopened.getByLabel('Import investigation file').setInputFiles({
    name: 'unsupported.json',
    mimeType: 'application/json',
    buffer: Buffer.from(JSON.stringify({ version: 2, records: [] })),
  })
  await expect(reopened.getByRole('region', { name: 'Saved investigations' })).toContainText(
    'Unsupported investigation file',
  )
  await expect(
    reopened.getByRole('region', { name: 'Saved investigations' }).locator('details'),
  ).toHaveCount(1)
})

test('trainable network exposes actual updates, holdout loss and replay on a narrow light screen', async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await page.goto('/')
  await page.getByRole('button', { name: 'Switch to light mode' }).click()
  await page.getByRole('button', { name: 'Open model wiki', exact: true }).click()
  const wiki = page.getByRole('dialog', { name: 'The model wiki', exact: true })
  await wiki.getByRole('button', { name: /What actually changes when a network learns/ }).click()
  const lab = wiki.getByRole('region', { name: 'Trainable network' })
  await expect(lab).toContainText('Step 0')
  await lab.getByRole('button', { name: 'Train 25 steps', exact: true }).click()
  await expect(lab).toContainText('Step 25')
  await expect(lab).toContainText('Raw m')
  await lab.getByLabel('Network optimizer').selectOption('sgd')
  await expect(lab).toContainText('Step 0')
  await lab.getByRole('button', { name: 'Train one step', exact: true }).click()
  await expect(lab).toContainText('Step 1')
  await lab.getByText('Record this investigation', { exact: true }).click()
  await lab.getByRole('button', { name: 'Save investigation', exact: true }).click()
  const library = wiki.getByRole('region', { name: 'Saved investigations' })
  await library.locator('summary').click()
  await library.getByRole('button', { name: 'Reopen setup', exact: true }).click()
  await expect(wiki.getByLabel('Network optimizer')).toHaveValue('sgd')
  await expect(wiki.getByRole('region', { name: 'Trainable network' })).toContainText('Step 1')
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(390)
  await wiki.locator('.study-training svg').scrollIntoViewIfNeeded()
  await page.screenshot({ path: 'docs/images/trainable-network-mobile.png' })
})

test('docked explanations follow canvas selection, leave controls interactive and expand on narrow screens', async ({
  page,
}) => {
  const errors: string[] = []
  page.on('pageerror', (e) => errors.push(e.message))
  await page.goto('/')
  await page.getByRole('button', { name: 'Open model wiki', exact: true }).click()
  const wiki = page.getByRole('dialog', { name: 'The model wiki', exact: true })
  await wiki.getByRole('button', { name: 'Dock beside model', exact: true }).click()
  await expect(wiki).toHaveAttribute('aria-modal', 'false')
  await wiki.getByLabel('Follow selection', { exact: true }).check()
  await page.getByRole('button', { name: 'Explore Transformer blocks', exact: true }).click()
  await expect(page.getByRole('heading', { level: 1 })).toHaveText('Inside a transformer block.')
  await expect(wiki.locator('.wiki-document-context')).toContainText('block')
  await page.screenshot({ path: 'docs/images/wiki-docked.png' })
  await page.setViewportSize({ width: 390, height: 844 })
  await expect(wiki).toHaveAttribute('aria-modal', 'true')
  await page.keyboard.press('Escape')
  await expect(wiki).not.toBeVisible()
  expect(errors).toEqual([])
})

test('a pinned numerical comparison opens the first changed output and saves an inspectable run', async ({
  page,
}) => {
  await page.goto('/')
  await page.getByRole('button', { name: /Tensor laboratory/ }).click()
  const lab = page.getByRole('dialog', { name: 'The mathematics, alive.', exact: true })
  await expect(lab.getByRole('button', { name: 'Pin baseline', exact: true })).toBeEnabled({
    timeout: 30000,
  })
  await lab.getByRole('button', { name: 'Pin baseline', exact: true }).click()
  await lab.getByRole('button', { name: 'Open numerical run setup', exact: true }).click()
  await lab.getByLabel('Numerical parameter seed', { exact: true }).fill('38')
  await expect(lab.locator('.run-comparison')).toBeVisible()
  await lab.locator('.run-comparison > summary').click()
  await expect(lab.getByRole('button', { name: /Inspect first changed operation/ })).toBeVisible()
  await lab.getByRole('button', { name: /Inspect first changed operation/ }).click()
  await lab.getByRole('button', { name: 'Open numerical lab notes', exact: true }).click()
  await lab.getByText('Record this investigation', { exact: true }).click()
  await lab.getByRole('button', { name: 'Save investigation', exact: true }).click()
  await expect(
    lab.getByRole('region', { name: 'Saved investigations' }).locator('details'),
  ).toHaveCount(1)
  await lab.getByRole('region', { name: 'Saved investigations' }).locator('summary').click()
  await lab.getByRole('button', { name: 'Reopen setup', exact: true }).click()
  await lab.getByRole('button', { name: 'Open numerical run setup', exact: true }).click()
  await expect(lab.getByLabel('Numerical parameter seed', { exact: true })).toHaveValue('38')
  await lab.getByLabel('Numerical parameter seed', { exact: true }).fill('39')
  await lab.getByRole('button', { name: 'Open numerical lab notes', exact: true }).click()
  await lab.getByRole('region', { name: 'Saved investigations' }).locator('summary').click()
  await lab.getByRole('button', { name: 'Reopen setup', exact: true }).click()
  await lab.getByRole('button', { name: 'Open numerical run setup', exact: true }).click()
  await expect(lab.getByLabel('Numerical parameter seed', { exact: true })).toHaveValue('38')
})

test('trained zero ablation and donor patch recompute real next-token probabilities and preserve the checkpoint', async ({
  page,
}) => {
  test.setTimeout(120000)
  const errors: string[] = [],
    external: string[] = []
  page.on('pageerror', (e) => errors.push(e.message))
  page.on('request', (r) => {
    if (!r.url().startsWith('http://127.0.0.1:5200') && !r.url().startsWith('data:'))
      external.push(r.url())
  })
  await page.goto('/')
  await page.getByRole('button', { name: /Local trained model/ }).click()
  await page.getByRole('button', { name: /Load trained model/ }).click()
  await expect(page.getByRole('button', { name: 'Generate & inspect', exact: true })).toBeEnabled({
    timeout: 30000,
  })
  await page.getByText('Experiment on a trained head', { exact: true }).click()
  await page.getByRole('button', { name: 'Run controlled intervention', exact: true }).click()
  const lab = page.getByRole('dialog', { name: 'The mathematics, alive.', exact: true })
  await expect(lab.locator('.run-comparison')).toContainText('Controlled zero ablation', {
    timeout: 30000,
  })
  await expect(lab.getByRole('button', { name: /Inspect first changed operation/ })).toContainText(
    'Head 0',
  )
  await lab.getByRole('button', { name: /Inspect first changed operation/ }).click()
  await expect(lab).toContainText('A controlled head-output intervention')
  await lab.getByLabel('Comparison target token').fill('403')
  await expect(lab.locator('.run-comparison')).toContainText('Token #403')
  await page.screenshot({ path: 'docs/images/trained-intervention.png' })
  await lab.getByRole('button', { name: 'Open numerical run setup', exact: true }).click()
  await lab.getByText('Experiment on a trained head', { exact: true }).click()
  await lab.getByLabel('Trained intervention replacement').selectOption('patch')
  await lab.getByLabel('Trained donor prompt').fill('Once upon a time')
  await lab.getByRole('button', { name: 'Run controlled intervention', exact: true }).click()
  await expect(lab.locator('.run-comparison')).toContainText('Controlled donor patch')
  await expect(lab.locator('.run-comparison')).toContainText('No aligned output cells exceeded')
  await lab.getByRole('button', { name: 'Open numerical lab notes', exact: true }).click()
  await lab.getByText('Record this investigation', { exact: true }).click()
  await lab.getByRole('button', { name: 'Save investigation', exact: true }).click()
  await lab.getByRole('region', { name: 'Saved investigations' }).locator('summary').click()
  await lab.getByRole('button', { name: 'Reopen setup', exact: true }).click()
  await expect(page.getByRole('button', { name: 'Generate & inspect', exact: true })).toBeDisabled()
  await page.getByText('Experiment on a trained head', { exact: true }).click()
  await expect(page.getByLabel('Trained intervention replacement')).toHaveValue('patch')
  await expect(page.getByLabel('Trained donor prompt')).toHaveValue('Once upon a time')
  expect(errors).toEqual([])
  expect(external).toEqual([])
})

test('a docked numerical reader follows the selected cell without blocking the microscope', async ({
  page,
}) => {
  await page.setViewportSize({ width: 1600, height: 1000 })
  await page.goto('/')
  await page.getByRole('button', { name: /Tensor laboratory/ }).click()
  const lab = page.getByRole('dialog', { name: 'The mathematics, alive.', exact: true })
  await lab.getByRole('button', { name: 'How is one attention score made?', exact: true }).click()
  await lab.getByRole('button', { name: 'Explain this', exact: true }).click()
  const wiki = page.getByRole('dialog', { name: 'The model wiki', exact: true })
  await wiki.getByRole('button', { name: 'Dock beside model', exact: true }).click()
  await expect(wiki.locator('.wiki-live-cell')).toBeVisible()
  await lab.locator('.tensor-view .numeric-cell[data-flat-index="1"]').click()
  await expect(wiki.locator('.wiki-live-cell')).toContainText('[0, 0, 1]')
  await wiki.getByText('Explain this value', { exact: true }).click()
  await expect(wiki.locator('.wiki-live-cell .katex-error')).toHaveCount(0)
  await expect(lab).toHaveClass(/wiki-docked-host/)
  await page.screenshot({ path: 'docs/images/wiki-docked-numerical.png' })
  await wiki.getByRole('button', { name: 'Close dialog', exact: true }).click()
  await expect(lab).not.toHaveClass(/wiki-docked-host/)
})

test('unsupported saved libraries survive a temporary experiment and can still export it', async ({
  page,
}) => {
  const raw = JSON.stringify({ version: 9, records: [{ future: 'keep exactly' }] })
  await page.addInitScript(
    (value) => localStorage.setItem('model-visualizer.investigations.v1', value),
    raw,
  )
  await page.goto('/')
  await page.getByRole('button', { name: 'Start with a question', exact: true }).click()
  const wiki = page.getByRole('dialog', { name: 'The model wiki', exact: true })
  await wiki.getByRole('button', { name: 'Observe the calculation', exact: true }).click()
  await wiki.getByText('Record this investigation', { exact: true }).click()
  await wiki.getByRole('button', { name: 'Save investigation', exact: true }).click()
  await expect(wiki.locator('.study-record-form')).toContainText('kept in this session')
  expect(
    await page.evaluate(() => localStorage.getItem('model-visualizer.investigations.v1')),
  ).toBe(raw)
  const download = page.waitForEvent('download')
  await wiki
    .locator('.study-record-form')
    .getByRole('button', { name: 'Export this session record', exact: true })
    .click()
  const file = await download
  expect(JSON.parse(await readFile((await file.path())!, 'utf8')).records).toHaveLength(1)
})

test('an explicit wiki opening replaces the dock destination and closing its owner removes the reader', async ({
  page,
}) => {
  await page.goto('/')
  await page.getByRole('button', { name: 'Open model wiki', exact: true }).click()
  const wiki = page.getByRole('dialog', { name: 'The model wiki', exact: true })
  await wiki.getByRole('button', { name: 'Dock beside model', exact: true }).click()
  await wiki.getByRole('button', { name: 'Browse wiki topics', exact: true }).click()
  await wiki
    .getByRole('navigation', { name: 'Wiki topics' })
    .getByRole('button', { name: 'Residual stream and decoder blocks', exact: true })
    .click()
  await expect(wiki.locator('.wiki-article h3')).toContainText('Residual stream')
  await page
    .locator('.topbar-actions')
    .getByRole('button', { name: 'Open model wiki', exact: true })
    .click()
  await expect(wiki.locator('.wiki-paths')).toBeVisible()
  await wiki.getByRole('button', { name: 'Close dialog', exact: true }).click()
  await page.getByRole('button', { name: /Tensor laboratory/ }).click()
  const lab = page.getByRole('dialog', { name: 'The mathematics, alive.', exact: true })
  await lab.getByRole('button', { name: 'Explain this', exact: true }).click()
  await expect(wiki).toHaveAttribute('aria-modal', 'false')
  await lab.locator(':scope > .modal-heading > button').click()
  await expect(lab).not.toBeVisible()
  await expect(wiki).not.toBeVisible()
  await page.getByRole('button', { name: 'Open model wiki', exact: true }).click()
  await expect(wiki).toBeVisible()
})
