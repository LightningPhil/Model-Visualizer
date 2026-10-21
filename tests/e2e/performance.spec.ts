import { expect, test, type Page } from '@playwright/test'
import { readFile } from 'node:fs/promises'
import starter from '../../examples/transformer-foundations.model.json' with { type: 'json' }
import { executeTeachingModel, defaultNumericalSettings } from '../../src/model/numerics/engine'
import { visibleGeometry } from './helpers/geometry'

type Traffic = {
  diagram: Record<string, number>
  scenes: number
  palette?: Record<string, string>
  scene?: { nodes: { id: string; x: number; y: number }[]; edges: { id: string; path: string }[] }
  runs: number[]
  exports: number
  stopped: string[]
}
declare global {
  interface Window {
    performanceTraffic: Traffic
  }
}

async function instrument(page: Page, failure = '') {
  await page.addInitScript((failure) => {
    const traffic: Traffic = (window.performanceTraffic = {
      diagram: {},
      scenes: 0,
      runs: [],
      exports: 0,
      stopped: [],
    })
    const NativeWorker = Worker
    window.Worker = class extends NativeWorker {
      source: string
      constructor(url: string | URL, options?: WorkerOptions) {
        if (failure === 'worker' && String(url).includes('overview.worker'))
          throw new Error('Test unavailable renderer')
        super(url, options)
        this.source = String(url)
      }
      postMessage(message: any, options?: Transferable[] | StructuredSerializeOptions) {
        if (this.source.includes('diagram.worker'))
          traffic.diagram[message.type] = (traffic.diagram[message.type] ?? 0) + 1
        if (this.source.includes('overview.worker') && message.type === 'scene') {
          traffic.scenes++
          traffic.scene = message.scene
          traffic.palette = message.palette
        }
        if (this.source.includes('trace.worker') && message.settings)
          traffic.runs.push(message.settings.seed)
        if (this.source.includes('trace.worker') && message.trace) traffic.exports++
        if (Array.isArray(options)) super.postMessage(message, options)
        else super.postMessage(message, options)
      }
      terminate() {
        traffic.stopped.push(this.source)
        super.terminate()
      }
    }
    if (failure === 'unsupported')
      Object.defineProperty(HTMLCanvasElement.prototype, 'transferControlToOffscreen', {
        value: undefined,
      })
  }, failure)
}

async function wholeMap(page: Page) {
  const doc = structuredClone(starter)
  doc.ui.activeGraph = 'whole-model-heads'
  await page.addInitScript(
    (model) => localStorage.setItem('model-visualizer.workspace.v1', JSON.stringify(model)),
    doc,
  )
  await page.goto('/')
  await expect(page.locator('.routed-connection')).toHaveCount(610, { timeout: 60000 })
}

test('worker overview draws the exact complete map; panning does not compile or reroute it', async ({
  page,
}) => {
  test.setTimeout(90000)
  const errors: string[] = []
  page.on('pageerror', (error) => errors.push(error.message))
  await instrument(page)
  await wholeMap(page)
  const canvas = page.locator('.graph-canvas')
  await expect(canvas).toHaveAttribute('data-renderer', 'overview')
  await expect(page.locator('.react-flow__node')).toHaveCount(420)
  await expect
    .poll(() => page.evaluate(() => window.performanceTraffic.scene?.edges.length))
    .toBe(610)
  const scene = await page.evaluate(() => window.performanceTraffic.scene!)
  const paths = await page
    .locator('.routed-connection')
    .evaluateAll((groups) =>
      Object.fromEntries(
        groups.map((group) => [
          group.getAttribute('data-edge-id'),
          group.querySelector('.react-flow__edge-path')!.getAttribute('d'),
        ]),
      ),
    )
  expect(Object.fromEntries(scene.edges.map((edge) => [edge.id, edge.path]))).toEqual(paths)
  expect(scene.nodes.map((node) => node.id).sort()).toEqual(
    await page
      .locator('.react-flow__node')
      .evaluateAll((nodes) => nodes.map((node) => node.getAttribute('data-id')!).sort()),
  )
  expect(await visibleGeometry(page)).toEqual([])
  // Let the initial fit/persistence settle before observing camera-only work.
  await page.waitForTimeout(500)
  const before = await page.evaluate(() => ({ ...window.performanceTraffic.diagram }))
  const sceneCount = await page.evaluate(() => window.performanceTraffic.scenes)
  const box = (await canvas.boundingBox())!
  await page.mouse.move(box.x + box.width * 0.5, box.y + box.height * 0.5)
  await page.mouse.down({ button: 'middle' })
  await page.mouse.move(box.x + box.width * 0.5 + 70, box.y + box.height * 0.5 + 35, { steps: 24 })
  await page.mouse.up({ button: 'middle' })
  await page.waitForTimeout(500)
  expect(await page.evaluate(() => window.performanceTraffic.diagram)).toEqual(before)
  expect(await page.evaluate(() => window.performanceTraffic.scenes)).toBe(sceneCount)
  // The visible React Flow wrapper still owns clicking and dragging; canvas
  // pixels are a drawing surface and never replace the interaction geometry.
  const first = page.locator('.react-flow__node').first()
  const originalPosition = await first.getAttribute('style')
  const nodeBox = (await first.boundingBox())!
  await page.mouse.move(nodeBox.x + nodeBox.width / 2, nodeBox.y + nodeBox.height / 2)
  await page.mouse.down()
  await page.mouse.move(nodeBox.x + nodeBox.width / 2 - 20, nodeBox.y + nodeBox.height / 2 - 20, {
    steps: 30,
  })
  await expect(canvas).toHaveAttribute('data-renderer', 'overview')
  await page.mouse.up()
  await expect(first).not.toHaveAttribute('style', originalPosition!)
  await expect(page.locator('.routed-connection')).toHaveCount(610, { timeout: 60000 })
  expect(await visibleGeometry(page, false)).toEqual([])
  await first.click()
  await expect(first).toHaveClass(/selected/)
  await page.getByRole('button', { name: 'Actual size', exact: true }).click()
  await expect(canvas).toHaveAttribute('data-renderer', 'svg')
  expect(
    await page
      .locator('.model-node')
      .first()
      .evaluate((node) => getComputedStyle(node).visibility),
  ).toBe('visible')
  await page.getByRole('button', { name: 'Fit graph', exact: true }).click()
  await expect(canvas).toHaveAttribute('data-renderer', 'overview')
  await page.getByRole('button', { name: 'Switch to light mode', exact: true }).click()
  await expect
    .poll(() => page.evaluate(() => window.performanceTraffic.palette?.panel))
    .toBe('#fff')
  await expect(canvas).toHaveAttribute('data-renderer', 'overview')
  await page.setViewportSize({ width: 1100, height: 760 })
  await expect
    .poll(() =>
      canvas.evaluate((root) => root.querySelector('canvas')!.width <= root.clientWidth * 2 + 1),
    )
    .toBe(true)
  const dimensions = await canvas.evaluate((root) => ({
    width: root.clientWidth,
    height: root.clientHeight,
    bitmapWidth: root.querySelector('canvas')!.width,
    bitmapHeight: root.querySelector('canvas')!.height,
  }))
  expect(dimensions.bitmapWidth).toBeLessThanOrEqual(dimensions.width * 2 + 1)
  expect(dimensions.bitmapHeight).toBeLessThanOrEqual(dimensions.height * 2 + 1)
  expect(errors).toEqual([])
})

for (const failure of ['unsupported', 'worker'])
  test(`complete SVG remains usable when overview rendering is ${failure}`, async ({ page }) => {
    test.setTimeout(60000)
    await instrument(page, failure)
    await wholeMap(page)
    await expect(page.locator('.graph-canvas')).toHaveAttribute('data-renderer', 'svg')
    expect(
      await page
        .locator('.model-node')
        .first()
        .evaluate((node) => getComputedStyle(node).visibility),
    ).toBe('visible')
    await page.locator('.react-flow__node').first().click()
    await expect(page.locator('.react-flow__node').first()).toHaveClass(/selected/)
    expect(await page.evaluate(() => window.performanceTraffic.scenes)).toBe(0)
  })

test('reference workers publish the latest run and export its unchanged actual arrays', async ({
  page,
}) => {
  test.setTimeout(90000)
  await instrument(page)
  await page.goto('/')
  await page.getByRole('button', { name: /Tensor laboratory/ }).click()
  const lab = page.getByRole('dialog', { name: 'The mathematics, alive.' })
  await expect(lab.getByRole('button', { name: 'Play', exact: true })).toBeEnabled({
    timeout: 60000,
  })
  await lab.getByLabel('Open numerical run setup').click()
  const seed = lab.getByLabel('Numerical parameter seed')
  await seed.fill('9')
  await seed.fill('91')
  await seed.fill('912')
  await lab.getByLabel('Numerical head sharing').selectOption('MQA')
  await lab.getByLabel('Numerical head sharing').selectOption('MHA')
  await lab.getByRole('button', { name: 'Explore the run' }).click()
  await expect(lab.getByTestId('numerical-computing')).toHaveCount(0)
  await expect(lab.getByRole('button', { name: 'Play', exact: true })).toBeEnabled({
    timeout: 60000,
  })
  await lab.getByRole('button', { name: 'How is one attention score made?', exact: true }).click()
  const cell = await lab.getByTestId('cell-result').textContent()
  const downloadPromise = page.waitForEvent('download')
  await lab.getByRole('button', { name: 'Export trace', exact: true }).click()
  const download = await downloadPromise
  const trace = JSON.parse(await readFile((await download.path())!, 'utf8'))
  const expected = executeTeachingModel({
    ...defaultNumericalSettings,
    seed: 912,
    attention: 'MHA',
  })
  expect(trace.settings).toEqual(expected.settings)
  expect(trace.outputIds).toEqual(expected.outputIds)
  for (const id of ['weights/L0 Wq', 'call0/l0/scores-0', 'call0/l0/attention-0'])
    expect(trace.tensors[id].data).toEqual(Array.from(expected.tensors[id].data))
  expect(await lab.getByTestId('cell-result').textContent()).toBe(cell)
  const traffic = await page.evaluate(() => window.performanceTraffic)
  expect(traffic.runs.at(-1)).toBe(912)
  expect(traffic.exports).toBe(1)
  expect(
    traffic.stopped.filter((source) => source.includes('trace.worker')).length,
  ).toBeGreaterThan(0)
})
