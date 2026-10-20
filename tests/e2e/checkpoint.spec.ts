import { expect, test, type Page } from '@playwright/test'
import { readFile } from 'node:fs/promises'
import reference from '../fixtures/stories260k-reference.json' with { type: 'json' }
import { storiesManifest, CHECKPOINT_SESSION_KEY } from '../../src/model/checkpoints/stories'
import { NUMERICAL_SESSION_KEY } from '../../src/model/numerics/identity'
import { visibleGeometry } from './helpers/geometry'

async function openTrained(page: Page, generate = 4) {
  await page.goto('/')
  await page.getByRole('button', { name: /Local trained model/ }).click()
  const intro = page.getByRole('dialog', { name: 'A real model. Inside your browser.' })
  await expect(intro.getByRole('button', { name: 'Generate & inspect' })).toBeDisabled()
  await intro.getByLabel('Local model output budget').selectOption(String(generate))
  await intro.getByRole('button', { name: /Load trained model/ }).click()
  await expect(intro.getByRole('button', { name: 'Generate & inspect' })).toBeEnabled({
    timeout: 30000,
  })
  await intro.getByRole('button', { name: 'Generate & inspect' }).click()
  const lab = page.getByRole('dialog', { name: 'The mathematics, alive.' })
  await expect(lab.getByRole('button', { name: 'Play', exact: true })).toBeEnabled({
    timeout: 120000,
  })
  await expect(lab.locator('.routing-message')).toHaveCount(0)
  return lab
}
test('trained generation exposes real weights, attention and cached activations without external inference', async ({
  page,
}) => {
  test.setTimeout(180000)
  const errors: string[] = [],
    external: string[] = [],
    assets: string[] = []
  page.on('pageerror', (e) => errors.push(e.message))
  page.on('request', (r) => {
    if (!r.url().startsWith('http://127.0.0.1:5200') && !r.url().startsWith('data:'))
      external.push(r.url())
    if (r.url().includes('/models/stories260k/')) assets.push(r.url())
  })
  await page.goto('/')
  await page.getByRole('button', { name: /Local trained model/ }).click()
  expect(assets).toHaveLength(0)
  await page.getByLabel('Local model output budget').selectOption('12')
  await page.getByRole('button', { name: /Load trained model/ }).click()
  await expect(page.getByRole('button', { name: 'Generate & inspect' })).toBeEnabled({
    timeout: 30000,
  })
  expect(assets).toHaveLength(2)
  await page.getByRole('button', { name: 'Generate & inspect' }).click()
  const lab = page.getByRole('dialog', { name: 'The mathematics, alive.' })
  await expect(lab.getByRole('button', { name: 'Play', exact: true })).toBeEnabled({
    timeout: 120000,
  })
  await expect(lab.locator('.numerical-badge')).toContainText('LOCAL · TRAINED')
  await expect(lab.locator('.react-flow__node')).toHaveCount(354)
  await expect(lab.getByLabel('Numerical input token 1')).toHaveValue('403')
  await expect(lab.getByLabel('Numerical input token 1')).not.toBeEditable()
  await lab.getByRole('button', { name: 'How is one attention score made?', exact: true }).click()
  await expect(lab.locator('.tensor-shape-line')).toContainText('[1 × 5 × 5]')
  await expect(lab.getByTestId('cell-result')).toHaveText('1.98879039')
  await lab.getByRole('button', { name: 'Query head 0 features [0, 0, 0] ↗', exact: true }).click()
  await expect(lab.locator('.numeric-current-tensor')).toContainText('Query head 0 features')
  await lab.getByRole('button', { name: /Back through the calculation/ }).click()
  await lab.locator('.parameter-library summary').click()
  await expect(lab.locator('.parameter-library strong')).toContainText('260,032 values')
  await lab.getByLabel('Find a numerical parameter').fill('vocabulary projection')
  await lab.locator('.parameter-library button').click()
  await expect(lab.locator('.cell-calculation h3')).toHaveText('A tied checkpoint parameter')
  await expect(lab.locator('.cell-calculation')).toContainText(storiesManifest.weights.sha256)
  await lab.getByRole('button', { name: 'Finish numerical playback' }).click()
  await expect(lab.locator('.trained-text')).toContainText('girl named Lily.')
  await lab.getByRole('button', { name: 'How is one attention score made?', exact: true }).click()
  await lab.getByLabel('Numerical decoder call').selectOption('2')
  await lab.getByRole('button', { name: 'How is one attention score made?', exact: true }).click()
  // Questions deliberately jump to their specified teaching call; select a
  // later real operation through the call-specific step list instead.
  await lab.getByLabel('Numerical decoder call').selectOption('2')
  await lab
    .locator('.numerical-steps button')
    .filter({ hasText: 'Q0 · K0ᵀ' })
    .filter({ hasText: 'L0' })
    .click()
  await expect(lab.locator('.tensor-shape-line')).toContainText('[1 × 1 × 7]')
  await expect(lab.locator('.numerical-step-card .execution-phase')).toContainText('call 2')
  const download = page.waitForEvent('download')
  await lab.getByRole('button', { name: 'Export trace', exact: true }).click()
  const artifact = JSON.parse(await readFile((await (await download).path())!, 'utf8'))
  expect(artifact.outputIds).toEqual(reference.outputIds.slice(0, 12))
  expect(artifact.checkpoint.sha256).toBe(storiesManifest.weights.sha256)
  expect(artifact.tokenization.inputIds).toEqual(reference.inputIds)
  expect(artifact.tensors['call2/l0/attention-0'].shape).toEqual([1, 1, 7])
  await lab.getByRole('button', { name: 'Finish numerical playback' }).click()
  await page.setViewportSize({ width: 1600, height: 1200 })
  await lab.getByRole('button', { name: 'Choose & loop', exact: true }).click()
  await page.screenshot({ path: 'docs/images/trained-model-dark.png' })
  expect(external).toEqual([])
  expect(errors).toEqual([])
})

test('trained layout, notebook and prompt round-trip separately, with clear routing in both directions and narrow themes', async ({
  page,
}) => {
  test.setTimeout(240000)
  const lab = await openTrained(page)
  const before = await page.evaluate(() =>
    JSON.parse(localStorage.getItem('model-visualizer.workspace.v1')!),
  )
  await lab
    .getByRole('button', { name: 'Where does a probability come from?', exact: true })
    .click()
  await lab.getByRole('button', { name: 'Bookmark', exact: true }).click()
  await lab.getByLabel('Open numerical lab notes').click()
  await lab
    .getByLabel('Numerical laboratory notes')
    .fill('A real trained head. Keep this notebook separate.')
  await lab.getByRole('button', { name: 'Auto-layout', exact: true }).click()
  await expect(lab.getByRole('status')).toContainText('Layout arranged', { timeout: 90000 })
  await expect.poll(() => visibleGeometry(page, false), { timeout: 30000 }).toEqual([])
  await lab.getByLabel('Switch layout direction').click()
  await expect(lab.getByRole('status')).toContainText('Layout arranged', { timeout: 90000 })
  await expect.poll(() => visibleGeometry(page, false), { timeout: 30000 }).toEqual([])
  await lab.getByLabel('Toggle numerical theme').click()
  await page.setViewportSize({ width: 390, height: 844 })
  await lab.getByRole('button', { name: 'Flow', exact: true }).click()
  await expect(lab.locator('.trained-text')).toBeVisible()
  expect((await lab.locator('.graph-canvas').boundingBox())!.height).toBeGreaterThan(80)
  expect(await lab.evaluate((e) => e.scrollWidth <= e.clientWidth + 1)).toBe(true)
  await lab.getByRole('button', { name: 'Inspect', exact: true }).click()
  await expect(lab.getByLabel('Numerical laboratory notes')).toBeVisible()
  expect(await lab.evaluate((e) => e.scrollWidth <= e.clientWidth + 1)).toBe(true)
  await page.screenshot({ path: 'docs/images/trained-model-mobile.png' })
  await lab.getByLabel('Close dialog').click()
  await page.setViewportSize({ width: 1440, height: 960 })
  const download = page.waitForEvent('download')
  await page.getByRole('button', { name: 'Export model', exact: true }).click()
  const path = (await (await download).path())!,
    saved = JSON.parse(await readFile(path, 'utf8')),
    session = saved.extensions[CHECKPOINT_SESSION_KEY]
  expect(saved.config).toEqual(before.config)
  expect(saved.graphs).toEqual(before.graphs)
  expect(saved.extensions[NUMERICAL_SESSION_KEY]).toEqual(before.extensions[NUMERICAL_SESSION_KEY])
  expect(session.study.notes).toContain('real trained head')
  expect(Object.keys(session.study.layout.positions)).toHaveLength(354)
  await page.getByLabel('Open model JSON').setInputFiles(path)
  await page.getByRole('button', { name: /Local trained model/ }).click()
  await expect(page.getByRole('button', { name: 'Generate & inspect' })).toBeDisabled()
  await page.getByRole('button', { name: /Load trained model/ }).click()
  await expect(page.getByRole('button', { name: 'Generate & inspect' })).toBeEnabled({
    timeout: 30000,
  })
  await page.getByRole('button', { name: 'Generate & inspect' }).click()
  await expect(lab.getByRole('button', { name: 'Play', exact: true })).toBeEnabled({
    timeout: 120000,
  })
  await lab.getByLabel('Open numerical lab notes').click()
  await expect(lab.getByLabel('Numerical laboratory notes')).toHaveValue(session.study.notes)
  await lab.locator('.numerical-bookmark > button').first().click()
  await expect(lab.locator('.numeric-current-tensor')).toContainText('attention probabilities')
})

test('corrupted weights are visibly refused before inference', async ({ page }) => {
  const body = await readFile('public/models/stories260k/stories260K.bin')
  body[101] ^= 1
  await page.route('**/models/stories260k/stories260K.bin', (route) =>
    route.fulfill({ body, contentType: 'application/octet-stream' }),
  )
  await page.goto('/')
  await page.getByRole('button', { name: /Local trained model/ }).click()
  await page.getByRole('button', { name: /Load trained model/ }).click()
  await expect(page.getByRole('alert')).toContainText('SHA-256 verification failed')
  await expect(page.getByRole('button', { name: 'Generate & inspect' })).toBeDisabled()
})
