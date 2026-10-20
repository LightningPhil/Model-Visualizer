import { expect, test } from '@playwright/test'
import { readFile } from 'node:fs/promises'
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

test('drills into components and inspects a mathematical connection', async ({ page }) => {
  await expect(page.locator('.model-node')).toHaveCount(7)
  await page.getByRole('button', { name: 'Explore Transformer blocks', exact: true }).click()
  await expect(page.getByRole('heading', { level: 1 })).toHaveText('Inside a transformer block.')
  await expect(page.locator('.model-node')).toHaveCount(7)
  await page.locator('.edge-label').filter({ hasText: 'residual +' }).first().click()
  await expect(page.getByRole('heading', { name: 'OPERATION', exact: true })).toBeVisible()
  await expect(page.locator('.connection-path')).toContainText('Residual stream')
  await expect(page.locator('.katex')).toBeVisible()
})

test('changes KV sharing and decode tensor shapes', async ({ page }) => {
  await page.getByRole('button', { name: 'Explore KV cache', exact: true }).click()
  await expect(page.getByLabel('Cache KV head')).toHaveValue('0')
  await page.getByRole('button', { name: 'MQA', exact: true }).click()
  await expect(page.getByLabel('Cache KV head').locator('option')).toHaveCount(1)
  await expect(page.locator('.cache-size')).toContainText('9 KiB')
  await page.getByRole('button', { name: 'MHA', exact: true }).click()
  await expect(page.getByLabel('Cache KV head').locator('option')).toHaveCount(8)
  await expect(page.locator('.cache-size')).toContainText('72 KiB')
  await page.getByRole('button', { name: 'Decode step', exact: true }).click()
  await expect(page.locator('.inference-caption')).toHaveText('T = 1 · S = 7 · one new token')
  await expect(
    page.locator('.model-node').filter({ hasText: 'Persistent K tensor' }),
  ).toContainText('[6 × 1 × 8 × 7 × 64]')
})

test('records draggable positions, restores them, and imports the same graph ID', async ({
  page,
}) => {
  const node = page.locator('.react-flow__node[data-id="decoder"]')
  await expect(node).toBeVisible()
  const before = await node.getAttribute('style')
  const box = (await node.boundingBox())!
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2)
  await page.mouse.down()
  await page.mouse.move(box.x + box.width / 2 + 50, box.y + box.height / 2 + 30, { steps: 12 })
  await page.mouse.up()
  await expect(node).not.toHaveAttribute('style', before!)
  const dragged = await node.getAttribute('style')
  await expect
    .poll(() =>
      page.evaluate(() => {
        const saved = localStorage.getItem('model-visualizer.workspace.v1')
        return saved ? JSON.parse(saved).graphs[0].layout.positions.decoder.x : 365
      }),
    )
    .not.toBe(365)
  await page.reload()
  await expect(node).toHaveAttribute('style', dragged!)
  await page.getByLabel('Open model JSON').setInputFiles({
    name: 'reset.model.json',
    mimeType: 'application/json',
    buffer: Buffer.from(JSON.stringify(starter)),
  })
  await expect(node).toHaveAttribute('style', /translate\(365px, 205px\)/)
})

test('auto-layout works and direction switch keeps nodes distinct', async ({ page }) => {
  await page.getByRole('button', { name: 'Auto-layout', exact: true }).click()
  await expect(page.getByRole('status')).toContainText('Layout arranged')
  await page.getByRole('button', { name: 'Switch layout direction' }).click()
  await expect(page.getByRole('button', { name: 'Auto-layout', exact: true })).toBeEnabled()
  await expect
    .poll(() =>
      page.evaluate(() => {
        const saved = localStorage.getItem('model-visualizer.workspace.v1')
        return saved ? JSON.parse(saved).graphs[0].layout.direction : null
      }),
    )
    .toBe('DOWN')
  const transforms = await page
    .locator('.react-flow__node')
    .evaluateAll((nodes) => nodes.map((n) => (n as HTMLElement).style.transform))
  expect(new Set(transforms).size).toBe(7)
})

test('exports notes, theme, model content and saved layouts as valid JSON', async ({ page }) => {
  await page.getByRole('button', { name: 'Notebook', exact: true }).click()
  await page.getByLabel('Workspace notes').fill('Inspect GQA sharing next.\nUse source equations.')
  await page.getByRole('button', { name: 'Switch to light mode' }).click()
  const downloadPromise = page.waitForEvent('download')
  await page.getByRole('button', { name: 'Export model', exact: true }).click()
  const download = await downloadPromise
  const exported = JSON.parse(await readFile((await download.path())!, 'utf8'))
  expect(exported.ui.notes).toBe('Inspect GQA sharing next.\nUse source equations.')
  expect(exported.ui.theme).toBe('light')
  expect(exported.graphs[0].layout.positions.decoder).toEqual({ x: 365, y: 205 })
  expect(exported.graphs).toHaveLength(9)
  await page.getByLabel('Open model JSON').setInputFiles((await download.path()) as string)
  await expect(page.getByLabel('Workspace notes')).toHaveValue(exported.ui.notes)
})

test('rejects malformed imports without replacing the working model', async ({ page }) => {
  const broken = structuredClone(starter)
  broken.graphs[0].edges[0].target = 'missing'
  await page.getByLabel('Open model JSON').setInputFiles({
    name: 'broken.json',
    mimeType: 'application/json',
    buffer: Buffer.from(JSON.stringify(broken)),
  })
  await expect(page.getByRole('status')).toContainText('Dangling connection')
  await expect(page.locator('.model-node')).toHaveCount(7)
})

test('works with Unicode and an empty prompt, and exposes real BPE merge steps', async ({
  page,
}) => {
  await page.getByLabel('PROMPT', { exact: true }).fill('🧠 café')
  await page.getByRole('button', { name: 'Open tokenizer lab', exact: true }).click()
  await expect(page.getByLabel('BPE merge step')).toBeVisible()
  await page.getByLabel('BPE merge step').fill('0')
  await expect(page.locator('.merge-pieces > span')).toHaveCount(
    new TextEncoder().encode('🧠 café').length,
  )
  await page.getByLabel('PROMPT', { exact: true }).fill('')
  await expect(page.locator('.empty-lab')).toBeVisible()
  await expect(page.getByRole('heading', { level: 1 })).toHaveText('From text to tokens.')
})

test('searches across graphs and switches attention heads', async ({ page }) => {
  await page.getByRole('button', { name: 'Search components', exact: true }).click()
  await page.getByRole('textbox', { name: 'Search model components' }).fill('Q · Kᵀ')
  await page.locator('.search-results > button').click()
  await expect(page.getByRole('heading', { name: 'Q · Kᵀ', exact: true })).toBeVisible()
  await page.getByRole('button', { name: 'Q7', exact: true }).click()
  await expect(page.locator('.head-mapping')).toContainText('K1 · V1')
  await page.getByRole('button', { name: 'Query 0, key 1: masked', exact: true }).click()
  await expect(page.locator('.heatmap-detail')).toContainText('masked')
  await page.getByRole('button', { name: 'Decode step', exact: true }).click()
  await expect(page.getByRole('heading', { name: 'One query. A cached prefix.' })).toBeVisible()
  await expect(page.locator('.heat-cell')).toHaveCount(7)
  await page.getByLabel('Active decoder layer').selectOption('5')
  await expect(page.locator('.layer-context')).toContainText('ℓ = 5')
})

test('shows navigation and a usable canvas on a narrow screen', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await page.getByRole('button', { name: 'Toggle inspector', exact: true }).click()
  await page.getByRole('button', { name: 'Open navigation', exact: true }).click()
  await page.getByRole('button', { name: 'N-gram memory 08', exact: true }).click()
  await expect(page.getByLabel('N-gram ending token')).toBeVisible()
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(390)
})

test('renders every graph in both themes without runtime errors', async ({ page }) => {
  const errors: string[] = []
  page.on('pageerror', (error) => errors.push(error.message))
  await expect(page.locator('.model-node')).toHaveCount(7)
  await page.evaluate(() => document.fonts.ready)
  await expect(page.locator('.routed-connection')).toHaveCount(starter.graphs[0].edges.length)
  await page.screenshot({ path: 'docs/images/workbench-dark.png' })
  await page.getByRole('button', { name: 'Switch to light mode' }).click()
  await page.screenshot({ path: 'docs/images/workbench-light.png' })
  const names = [
    'Architecture overview',
    'Tokenization',
    'Transformer block',
    'Attention & heads',
    'KV cache',
    'Feed-forward network',
    'Output & decoding',
    'N-gram memory',
    'Multi-token prediction',
  ]
  for (const [index, name] of names.entries()) {
    await page.locator('.nav-item').filter({ hasText: name }).click()
    await expect(page.locator('.model-node').first()).toBeVisible()
    await expect(page.locator('.routed-connection')).toHaveCount(
      starter.graphs[index].edges.length,
      { timeout: 15000 },
    )
    await expect(page.locator('.katex-error')).toHaveCount(0)
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(1440)
  }
  await page.getByRole('button', { name: 'Switch to dark mode' }).click()
  await page.locator('.nav-item').filter({ hasText: 'KV cache' }).click()
  await page.getByLabel('Cache layer').selectOption('3')
  await page.getByLabel('Cache KV head').selectOption('1')
  await expect(page.locator('.routed-connection')).toHaveCount(starter.graphs[4].edges.length)
  await page.screenshot({ path: 'docs/images/kv-cache.png' })
  expect(errors).toEqual([])
})
