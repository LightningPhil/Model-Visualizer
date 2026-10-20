import { expect, test } from '@playwright/test'
import starter from '../../examples/transformer-foundations.model.json' with { type: 'json' }
import { graphicalAdapter } from '../../src/model/simulation'
import type { ModelConfig } from '../../src/model/schema'

test('split heads expose independent operations and correct sharing in MHA, GQA and MQA', async ({
  page,
}) => {
  test.setTimeout(90000)
  await page.goto('/')
  await page.locator('.nav-item').filter({ hasText: 'Attention & heads' }).click()
  await page.getByRole('button', { name: 'Split attention heads', exact: true }).click()
  await expect(page.locator('.model-node')).toHaveCount(54)
  await expect(page.locator('.routed-connection')).toHaveCount(80, { timeout: 60000 })
  await page.getByLabel('Inspect query head').selectOption('7')
  await expect
    .poll(async () => {
      const n = (await page.locator('.react-flow__node[data-id="scores-7"]').boundingBox())!
      const c = (await page.getByTestId('graph-canvas').boundingBox())!
      return Math.abs(n.x + n.width / 2 - c.x - c.width / 2) < 10
    })
    .toBe(true)
  await expect(page.getByRole('heading', { name: 'Q7 · K1ᵀ', exact: true })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Q7', exact: true })).toHaveAttribute(
    'aria-pressed',
    'true',
  )
  await expect(page.locator('.head-mapping')).toContainText('Q7')
  await expect(page.locator('.katex-error')).toHaveCount(0)
  await page.screenshot({ path: 'docs/images/attention-heads.png' })
  for (const [mode, nodes, edges, kv] of [
    ['MHA', 66, 104, 7],
    ['MQA', 52, 76, 0],
    ['GQA', 54, 80, 1],
  ] as const) {
    await page.getByRole('button', { name: mode, exact: true }).click()
    await expect(page.locator('.model-node')).toHaveCount(nodes)
    await expect(page.locator('.routed-connection')).toHaveCount(edges, { timeout: 60000 })
    await page.getByRole('button', { name: 'Q7', exact: true }).click()
    await expect(page.locator('.head-mapping')).toContainText(`K${kv} · V${kv}`)
    await expect(page.locator('.routing-message')).toHaveCount(0)
  }
})

test('cache policies preserve absolute indices, change storage and survive export/reload in both themes', async ({
  page,
}) => {
  await page.goto('/')
  await page.getByRole('button', { name: 'Explore KV cache', exact: true }).click()
  await expect(page.locator('.routed-connection')).toHaveCount(8, { timeout: 30000 })
  await page.getByLabel('Cached sequence length').fill('12')
  await page.getByLabel('KV cache retention').selectOption('sliding-window')
  await page.getByLabel('KV window size').fill('3')
  await expect(page.locator('.cache-size')).toContainText('9 KiB')
  await expect(page.locator('.retention-cells [data-retained="true"]')).toHaveCount(3)
  await expect(page.locator('.retention-cells [data-position="9"]')).toHaveAttribute(
    'data-retained',
    'true',
  )
  await expect(page.locator('.ring-example')).toContainText('pos 9 → slot 0')
  await page.getByLabel('KV cache retention').selectOption('sparse')
  await page.getByLabel('Sparse anchor stride').fill('4')
  await expect(page.locator('.cache-size')).toContainText('18 KiB')
  await expect(page.locator('.retention-cells [data-retained="true"]')).toHaveCount(6)
  await expect(page.locator('.retention-cells [data-position="4"]')).toHaveAttribute(
    'data-retained',
    'true',
  )
  await expect(page.locator('.retention-cells [data-position="5"]')).toHaveAttribute(
    'data-retained',
    'false',
  )
  await page.screenshot({ path: 'docs/images/cache-sparse.png' })
  await page.getByRole('button', { name: 'Switch to light mode' }).click()
  await page.screenshot({ path: 'docs/images/cache-sparse-light.png' })
  await expect
    .poll(() =>
      page.evaluate(
        () => JSON.parse(localStorage.getItem('model-visualizer.workspace.v1')!).config.kvCache,
      ),
    )
    .toEqual({ mode: 'sparse', windowSize: 3, stride: 4 })
  const downloadPromise = page.waitForEvent('download')
  await page.getByRole('button', { name: 'Export model', exact: true }).click()
  const download = await downloadPromise
  await page.getByLabel('Open model JSON').setInputFiles((await download.path())!)
  await expect(page.getByLabel('KV cache retention')).toHaveValue('sparse')
  await page.reload()
  await expect(page.getByLabel('KV cache retention')).toHaveValue('sparse')
  await page.setViewportSize({ width: 390, height: 844 })
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(390)
  await page.getByLabel('KV cache retention').selectOption('full')
  await expect(page.getByLabel('KV window size')).toHaveCount(0)
})

test('whole-model split view saves separately and windowed MTP retains checkpoints until rollback', async ({
  page,
}) => {
  test.setTimeout(180000)
  await page.goto('/')
  await page
    .getByRole('button', { name: 'Whole model Every layer on one canvas', exact: true })
    .click()
  await expect(page.locator('.routed-connection')).toHaveCount(358, { timeout: 60000 })
  await expect
    .poll(() =>
      page.evaluate(
        () =>
          Object.keys(
            JSON.parse(localStorage.getItem('model-visualizer.workspace.v1')!).graphs.find(
              (g: { id: string }) => g.id === 'whole-model',
            ).layout.positions,
          ).length,
      ),
    )
    .toBe(252)
  const baseline = await page.evaluate(
    () =>
      JSON.parse(localStorage.getItem('model-visualizer.workspace.v1')!).graphs.find(
        (g: { id: string }) => g.id === 'whole-model',
      ).layout.positions,
  )
  await page.getByRole('button', { name: 'Split attention heads', exact: true }).click()
  await expect(page.locator('.model-node')).toHaveCount(420)
  await expect(page.locator('.routed-connection')).toHaveCount(610, { timeout: 120000 })
  await expect
    .poll(() =>
      page.evaluate(
        () =>
          Object.keys(
            JSON.parse(localStorage.getItem('model-visualizer.workspace.v1')!).graphs.find(
              (g: { id: string }) => g.id === 'whole-model-heads',
            ).layout.positions,
          ).length,
      ),
    )
    .toBe(420)
  expect(
    await page.evaluate(
      () =>
        JSON.parse(localStorage.getItem('model-visualizer.workspace.v1')!).graphs.find(
          (g: { id: string }) => g.id === 'whole-model',
        ).layout.positions,
    ),
  ).toEqual(baseline)
  await page.reload()
  await expect(page.getByRole('heading', { level: 1 })).toHaveText(
    'Every layer. Every head. Every operation.',
  )
  await expect(page.locator('.routed-connection')).toHaveCount(610, { timeout: 60000 })
  await page
    .getByRole('button', { name: 'Run simulation Follow the execution', exact: true })
    .click()
  await page.getByText('Run settings', { exact: false }).click()
  await page.getByLabel('KV cache retention').selectOption('sliding-window')
  await page.getByLabel('Verification scenario').selectOption('reject')
  const config: ModelConfig = {
    ...(starter.config as ModelConfig),
    kvCache: { mode: 'sliding-window', windowSize: 4 },
  }
  const trace = graphicalAdapter.run(config, {
    promptTokens: 6,
    newTokens: 8,
    mtp: true,
    scenario: 'reject',
    splitHeads: true,
  })
  await page
    .getByLabel('Simulation timeline')
    .fill(String(trace.frames.findIndex((f) => f.title === 'Write tentative K/V')))
  await expect(page.locator('.cache-layer-list b').first()).toHaveText('7')
  await page
    .getByLabel('Simulation timeline')
    .fill(String(trace.frames.findIndex((f) => f.phase === 'rollback')))
  await expect(page.locator('.cache-layer-list b')).toHaveText(Array(6).fill('4'))
  await expect(
    page.locator('.cache-layer-list > div').first().locator('[data-sequence-position]'),
  ).toHaveText(['3', '4', '5', '6'])
  await expect(page.getByTestId('cache-evictions')).toBeVisible()
  await expect(page.locator('.routing-message')).toHaveCount(0)
  await page.getByRole('button', { name: 'Run to completion' }).click()
  await expect(page.getByTestId('generated-count')).toHaveText('8 / 8')
})
