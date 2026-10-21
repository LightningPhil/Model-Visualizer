import { visibleGeometry } from './helpers/geometry'
import { expect, test, type Page } from '@playwright/test'
import { readFileSync } from 'node:fs'
import { parseDocument } from '../../src/model/schema'
import { expandWholeModel, instanceOf } from '../../src/model/whole-model'

const starter = JSON.parse(
  readFileSync(
    new URL('../../examples/transformer-foundations.model.json', import.meta.url),
    'utf8',
  ),
)
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
  'Generation & verification',
  'Split attention heads',
]

async function connectionsReady(page: Page, count: number) {
  await expect(page.locator('.routed-connection')).toHaveCount(count, { timeout: 15000 })
  await expect(page.locator('.edge-label')).toHaveCount(count)
  await expect(page.locator('.routing-message')).toHaveCount(0)
}

for (const id of ['whole-model', 'whole-model-heads'])
  test(`${id}: auto-layout compacts widely separated sections in both directions`, async ({
    page,
  }) => {
    test.setTimeout(180000)
    const document = parseDocument(starter)
    document.ui.activeGraph = id
    const graph = expandWholeModel(
      document,
      document.graphs.find((g) => g.id === id)!,
    )
    const stages = [...new Set(graph.nodes.map((n) => instanceOf(n)!.stage))]
    await page.goto('/')
    await page.getByLabel('Open model JSON').setInputFiles({
      name: 'whole.model.json',
      mimeType: 'application/json',
      buffer: Buffer.from(JSON.stringify(document)),
    })
    await expect(page.locator('.routed-connection')).toHaveCount(graph.edges.length, {
      timeout: 60000,
    })
    const readPositions = () =>
      page.locator('.react-flow__node').evaluateAll((nodes) =>
        Object.fromEntries(
          nodes.map((node) => {
            const transform = new DOMMatrixReadOnly((node as HTMLElement).style.transform)
            return [node.getAttribute('data-id')!, { x: transform.e, y: transform.f }]
          }),
        ),
      )
    const initial = await readPositions()
    const spread = Object.fromEntries(
      graph.nodes.map((n) => [
        n.id,
        { x: initial[n.id].x + stages.indexOf(instanceOf(n)!.stage) * 20000, y: initial[n.id].y },
      ]),
    )
    document.graphs.find((g) => g.id === id)!.layout.positions = spread
    await page.getByLabel('Open model JSON').setInputFiles({
      name: 'spread.model.json',
      mimeType: 'application/json',
      buffer: Buffer.from(JSON.stringify(document)),
    })
    await expect(page.locator('.routed-connection')).toHaveCount(graph.edges.length, {
      timeout: 60000,
    })
    await page.getByRole('button', { name: 'Auto-layout', exact: true }).click()
    await expect(page.getByRole('button', { name: 'Undo auto-layout' })).toBeEnabled({
      timeout: 60000,
    })
    await connectionsReady(page, graph.edges.length)
    const compact = await readPositions()
    const width = (positions: typeof compact) =>
      Math.max(...Object.values(positions).map((p) => p.x)) -
      Math.min(...Object.values(positions).map((p) => p.x))
    expect(width(compact)).toBeLessThan(width(spread) * 0.5)
    expect(Object.keys(compact).sort()).toEqual(Object.keys(spread).sort())
    expect(await visibleGeometry(page)).toEqual([])
    if (id === 'whole-model') await page.screenshot({ path: 'docs/images/whole-model-compact.png' })
    await page.getByRole('button', { name: 'Undo auto-layout' }).click()
    await expect.poll(readPositions).toEqual(spread)
    await connectionsReady(page, graph.edges.length)
    await page.getByRole('button', { name: 'Switch layout direction' }).click()
    await expect(page.getByRole('button', { name: 'Auto-layout', exact: true })).toBeEnabled({
      timeout: 60000,
    })
    await connectionsReady(page, graph.edges.length)
    expect(await visibleGeometry(page)).toEqual([])
    await expect
      .poll(() =>
        page.evaluate(
          (graphId) =>
            JSON.parse(
              localStorage.getItem('model-visualizer.workspace.v1') ?? 'null',
            )?.graphs.find((g: { id: string }) => g.id === graphId)?.layout.direction,
          id,
        ),
      )
      .toBe('DOWN')
    const final = await readPositions()
    const savedFinal = await page.evaluate(
      (graphId) =>
        JSON.parse(localStorage.getItem('model-visualizer.workspace.v1')!).graphs.find(
          (g: { id: string }) => g.id === graphId,
        ).layout.positions,
      id,
    )
    // CSS serialization and DOMMatrix use lower precision than saved canvas
    // coordinates. Check that every rendered point matches within 0.01 px.
    for (const [nodeId, position] of Object.entries(final)) {
      expect(Math.abs(savedFinal[nodeId].x - position.x)).toBeLessThan(0.01)
      expect(Math.abs(savedFinal[nodeId].y - position.y)).toBeLessThan(0.01)
    }
    await page.reload()
    await expect(page.locator('.routed-connection')).toHaveCount(graph.edges.length, {
      timeout: 60000,
    })
    expect(await readPositions()).toEqual(final)
  })

for (const [index, name] of names.entries())
  test(`${name}: every rendered link and label stays clear in both directions`, async ({
    page,
  }) => {
    const errors: string[] = []
    page.on('pageerror', (error) => errors.push(error.message))
    await page.goto('/')
    await page.locator('.nav-item').filter({ hasText: name }).click()
    const doc = parseDocument(starter)
    const graph =
      name === 'Split attention heads'
        ? doc.graphs.find((g) => g.id === 'attention-heads')!
        : doc.graphs[index]
    const count = expandWholeModel(doc, graph).edges.length
    await connectionsReady(page, count)
    await page.getByRole('button', { name: 'Auto-layout', exact: true }).click()
    await expect(page.getByRole('button', { name: 'Undo auto-layout' })).toBeEnabled({
      timeout: 15000,
    })
    await connectionsReady(page, count)
    await expect.poll(() => visibleGeometry(page), { timeout: 10000 }).toEqual([])
    await page.getByRole('button', { name: 'Switch layout direction' }).click()
    await expect(page.getByRole('button', { name: 'Auto-layout', exact: true })).toBeEnabled({
      timeout: 15000,
    })
    await connectionsReady(page, count)
    await expect.poll(() => visibleGeometry(page), { timeout: 10000 }).toEqual([])
    if (name === 'Feed-forward network') {
      await page.setViewportSize({ width: 1440, height: 1200 })
      await page.getByRole('button', { name: 'Toggle inspector', exact: true }).click()
      await page.getByRole('button', { name: 'Fit graph', exact: true }).click()
      await expect.poll(() => visibleGeometry(page)).toEqual([])
      await page.screenshot({ path: 'docs/images/auto-layout.png' })
    }
    expect(errors).toEqual([])
  })

test('undo restores authored positions, ports and viewport; JSON reload preserves automatic routing', async ({
  page,
}) => {
  await page.goto('/')
  await connectionsReady(page, starter.graphs[0].edges.length)
  const transforms = () =>
    page
      .locator('.react-flow__node')
      .evaluateAll((nodes) => nodes.map((n) => (n as HTMLElement).style.transform))
  const initial = await transforms()
  const initialRoutes = await page
    .locator('.routed-connection')
    .evaluateAll((nodes) => nodes.map((n) => n.getAttribute('data-route-points')))
  const initialViewport = await page.locator('.react-flow__viewport').getAttribute('style')
  await page.getByRole('button', { name: 'Auto-layout', exact: true }).click()
  await expect(page.getByRole('button', { name: 'Undo auto-layout' })).toBeEnabled({
    timeout: 15000,
  })
  await connectionsReady(page, starter.graphs[0].edges.length)
  await expect.poll(transforms).not.toEqual(initial)
  await page.getByRole('button', { name: 'Undo auto-layout' }).click()
  await expect.poll(transforms).toEqual(initial)
  await connectionsReady(page, starter.graphs[0].edges.length)
  expect(
    await page
      .locator('.routed-connection')
      .evaluateAll((nodes) => nodes.map((n) => n.getAttribute('data-route-points'))),
  ).toEqual(initialRoutes)
  await expect(page.locator('.react-flow__viewport')).toHaveAttribute('style', initialViewport!)
  await page.getByRole('button', { name: 'Auto-layout', exact: true }).click()
  await expect(page.getByRole('button', { name: 'Undo auto-layout' })).toBeEnabled({
    timeout: 15000,
  })
  await connectionsReady(page, starter.graphs[0].edges.length)
  const automatic = await transforms()
  const automaticPositions = await page.locator('.react-flow__node').evaluateAll((nodes) =>
    Object.fromEntries(
      nodes.map((node) => {
        const matrix = new DOMMatrixReadOnly((node as HTMLElement).style.transform)
        return [node.getAttribute('data-id'), { x: matrix.e, y: matrix.f }]
      }),
    ),
  )
  const routes = await page
    .locator('.routed-connection')
    .evaluateAll((nodes) => nodes.map((n) => n.getAttribute('data-route-points')))
  await expect
    .poll(() =>
      page.evaluate(
        () =>
          JSON.parse(localStorage.getItem('model-visualizer.workspace.v1') ?? 'null')?.graphs[0]
            .layout.positions,
      ),
    )
    .toEqual(automaticPositions)
  await page.reload()
  await connectionsReady(page, starter.graphs[0].edges.length)
  expect(await transforms()).toEqual(automatic)
  expect(
    await page
      .locator('.routed-connection')
      .evaluateAll((nodes) => nodes.map((n) => n.getAttribute('data-route-points'))),
  ).toEqual(routes)
  expect(await visibleGeometry(page, false)).toEqual([])
})

test('dragging a component across a connection reroutes the actual SVG around it', async ({
  page,
}) => {
  const imported = structuredClone(starter)
  const base = imported.graphs[0]
  const graph = {
    ...base,
    id: 'obstacle-test',
    nodes: ['a', 'b', 'barrier'].map((id) => ({
      ...base.nodes[0],
      id,
      title: id,
      childGraph: undefined,
    })),
    edges: [
      {
        ...base.edges[0],
        id: 'test-link',
        source: 'a',
        target: 'b',
        label: 'dot product',
        sourceSide: undefined,
        targetSide: undefined,
      },
    ],
    layout: {
      direction: 'RIGHT',
      positions: { a: { x: 0, y: 0 }, b: { x: 840, y: 0 }, barrier: { x: 420, y: 350 } },
    },
  }
  imported.graphs.push(graph)
  imported.ui.activeGraph = graph.id
  await page.goto('/')
  await page.getByLabel('Open model JSON').setInputFiles({
    name: 'obstacle.model.json',
    mimeType: 'application/json',
    buffer: Buffer.from(JSON.stringify(imported)),
  })
  await connectionsReady(page, 1)
  await page.getByRole('button', { name: 'Fit graph', exact: true }).click()
  await expect.poll(() => visibleGeometry(page)).toEqual([])
  const node = page.locator('.react-flow__node[data-id="barrier"]')
  const a = (await page.locator('.react-flow__node[data-id="a"]').boundingBox())!
  const box = (await node.boundingBox())!
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2)
  await page.mouse.down()
  await page.mouse.move(box.x + box.width / 2, a.y + a.height / 2, { steps: 20 })
  await page.mouse.up()
  await connectionsReady(page, 1)
  await expect.poll(() => visibleGeometry(page, false)).toEqual([])
  expect(
    JSON.parse((await page.locator('.routed-connection').getAttribute('data-route-points'))!)
      .length,
  ).toBe(6)
})
