import { expect, test, type Page } from '@playwright/test'
import { readFileSync } from 'node:fs'

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
]

async function connectionsReady(page: Page, count: number) {
  await expect(page.locator('.routed-connection')).toHaveCount(count, { timeout: 15000 })
  await expect(page.locator('.edge-label')).toHaveCount(count)
  await expect(page.locator('.routing-message')).toHaveCount(0)
}

/** Inspect real SVG curves and HTML boxes after transforms, not router metadata. */
async function visibleGeometry(page: Page, fit = true) {
  return page.evaluate((fit) => {
    const problems: string[] = []
    const canvas = document.querySelector('.graph-canvas')!.getBoundingClientRect()
    const nodes = [...document.querySelectorAll('.model-node')].map((n) => ({
      id: n.closest('[data-id]')!.getAttribute('data-id')!,
      box: n.getBoundingClientRect(),
    }))
    const labels = [...document.querySelectorAll('.edge-label')].map((n) => ({
      id: n.getAttribute('data-label-edge')!,
      box: n.getBoundingClientRect(),
    }))
    for (const label of document.querySelectorAll<HTMLElement>('.edge-label')) {
      if (label.scrollWidth > label.clientWidth + 1 || label.scrollHeight > label.clientHeight + 1)
        problems.push(`label ${label.dataset.labelEdge} clips its text`)
    }
    const overlap = (a: DOMRect, b: DOMRect) =>
      a.left < b.right - 0.3 &&
      a.right > b.left + 0.3 &&
      a.top < b.bottom - 0.3 &&
      a.bottom > b.top + 0.3
    const contains = (r: DOMRect, x: number, y: number) =>
      x > r.left + 0.3 && x < r.right - 0.3 && y > r.top + 0.3 && y < r.bottom - 0.3
    for (const [i, n] of nodes.entries())
      for (const other of nodes.slice(i + 1))
        if (overlap(n.box, other.box)) problems.push(`nodes ${n.id}/${other.id} overlap`)
    for (const [i, l] of labels.entries()) {
      for (const n of nodes)
        if (overlap(l.box, n.box)) problems.push(`label ${l.id} overlaps ${n.id}`)
      for (const other of labels.slice(i + 1))
        if (overlap(l.box, other.box)) problems.push(`labels ${l.id}/${other.id} overlap`)
    }
    for (const group of document.querySelectorAll('.routed-connection')) {
      const id = group.getAttribute('data-edge-id')!
      const path = group.querySelector<SVGPathElement>('.react-flow__edge-path')!
      const matrix = path.getScreenCTM()!
      const length = path.getTotalLength()
      // Sample at <= 1 canvas pixel, including the rounded corner geometry.
      for (let d = 0; d <= Math.ceil(length); d++) {
        const p = path.getPointAtLength(Math.min(d, length))
        const x = p.x * matrix.a + p.y * matrix.c + matrix.e,
          y = p.x * matrix.b + p.y * matrix.d + matrix.f
        for (const n of nodes) if (contains(n.box, x, y)) problems.push(`${id} hidden by ${n.id}`)
        for (const l of labels)
          if (l.id !== id && contains(l.box, x, y)) problems.push(`${id} hidden by label ${l.id}`)
        if (
          fit &&
          (x < canvas.left + 20 ||
            x > canvas.right - 20 ||
            y < canvas.top + 50 ||
            y > canvas.bottom - 65)
        )
          problems.push(`${id} outside fitted drawing area`)
      }
    }
    if (fit)
      for (const { id, box } of [...nodes, ...labels])
        if (
          box.left < canvas.left + 20 ||
          box.right > canvas.right - 20 ||
          box.top < canvas.top + 50 ||
          box.bottom > canvas.bottom - 65
        )
          problems.push(`${id} outside fitted drawing area`)
    return [...new Set(problems)]
  }, fit)
}

for (const [index, name] of names.entries())
  test(`${name}: every rendered link and label stays clear in both directions`, async ({
    page,
  }) => {
    const errors: string[] = []
    page.on('pageerror', (error) => errors.push(error.message))
    await page.goto('/')
    await page.locator('.nav-item').filter({ hasText: name }).click()
    const count = starter.graphs[index].edges.length
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
  const routes = await page
    .locator('.routed-connection')
    .evaluateAll((nodes) => nodes.map((n) => n.getAttribute('data-route-points')))
  await expect
    .poll(() =>
      page.evaluate(
        () =>
          JSON.parse(localStorage.getItem('model-visualizer.workspace.v1')!).graphs[0].layout
            .positions,
      ),
    )
    .not.toEqual(starter.graphs[0].layout.positions)
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
