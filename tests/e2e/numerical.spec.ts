import { expect, test, type Page } from '@playwright/test'
import { readFile } from 'node:fs/promises'
import { executeTeachingModel, defaultNumericalSettings } from '../../src/model/numerics/engine'
import { valueLabel } from '../../src/model/numerics/explain'
import { NUMERICAL_SESSION_KEY } from '../../src/model/numerics/session'
import {
  LEGACY_DOCUMENTATION,
  LEGACY_SIMULATION_NOTE,
  TEACHING_PROVENANCE_KEY,
} from '../../src/model/teaching-provenance'

async function openLab(page: Page) {
  await page.goto('/')
  await page.getByRole('button', { name: /Tensor laboratory/ }).click()
  const lab = page.getByRole('dialog', { name: 'The mathematics, alive.' })
  await expect(lab.getByRole('button', { name: 'Play', exact: true })).toBeEnabled({
    timeout: 60000,
  })
  await expect(lab.locator('.routing-message')).toHaveCount(0)
  return lab
}
test('older stock explanations are clarified while custom prose and the notebook survive', async ({
  page,
}) => {
  await page.goto('/')
  const doc = JSON.parse(
    await readFile(
      new URL('../../examples/transformer-foundations.model.json', import.meta.url),
      'utf8',
    ),
  )
  delete doc.extensions[TEACHING_PROVENANCE_KEY]
  doc.documentation = LEGACY_DOCUMENTATION
  doc.ui.notes = 'My existing notebook.'
  doc.graphs[0].layout.viewport = { x: 12, y: 34, zoom: 0.7 }
  doc.graphs.find((g: { id: string }) => g.id === 'generation').nodes[0].notes = [
    LEGACY_SIMULATION_NOTE,
    'My independent annotation.',
  ]
  await page.getByLabel('Open model JSON').setInputFiles({
    name: 'earlier-workspace.model.json',
    mimeType: 'application/json',
    buffer: Buffer.from(JSON.stringify(doc)),
  })
  await expect
    .poll(() =>
      page.evaluate(
        () => JSON.parse(localStorage.getItem('model-visualizer.workspace.v1')!)?.ui?.notes,
      ),
    )
    .toBe('My existing notebook.')
  await expect
    .poll(() =>
      page.evaluate(
        (key) => JSON.parse(localStorage.getItem('model-visualizer.workspace.v1')!).extensions[key],
        TEACHING_PROVENANCE_KEY,
      ),
    )
    .toBe(1)
  const saved = await page.evaluate(() =>
    JSON.parse(localStorage.getItem('model-visualizer.workspace.v1')!),
  )
  expect(saved.documentation).toContain('absolute processed extent')
  expect(saved.ui.notes).toBe('My existing notebook.')
  expect(saved.graphs.find((g: { id: string }) => g.id === 'generation').nodes[0].notes).toContain(
    'My independent annotation.',
  )
  expect(saved.graphs.map((g: { layout: unknown }) => g.layout)).toEqual(
    doc.graphs.map((g: { layout: unknown }) => g.layout),
  )
})
test('a question chosen during initial layout keeps its camera destination when routing finishes', async ({
  page,
}) => {
  test.setTimeout(90000)
  await page.goto('/')
  await page.getByRole('button', { name: /Tensor laboratory/ }).click()
  const lab = page.getByRole('dialog', { name: 'The mathematics, alive.' })
  await lab.getByRole('button', { name: 'How is one attention score made?', exact: true }).click()
  await expect(lab.getByRole('button', { name: 'Play', exact: true })).toBeEnabled({
    timeout: 60000,
  })
  await expect(lab.locator('.numerical-step-card h3')).toHaveText('Q0 · K0ᵀ')
  await expect
    .poll(async () => {
      const node = (await lab
          .locator('.react-flow__node[data-id="layer-0.attn.scores-0"]')
          .boundingBox())!,
        canvas = (await lab.getByTestId('graph-canvas').boundingBox())!
      return (
        Math.abs(node.x + node.width / 2 - canvas.x - canvas.width / 2) < 10 &&
        Math.abs(node.y + node.height / 2 - canvas.y - canvas.height / 2) < 10
      )
    })
    .toBe(true)
})
test('architecture operations open their numerical example and every parameter matrix is inspectable', async ({
  page,
}) => {
  test.setTimeout(90000)
  await page.goto('/')
  await page.locator('.nav-item').filter({ hasText: 'Attention & heads' }).click()
  await page.locator('.react-flow__node[data-id="ropeq"]').click()
  const title = await page.getByRole('heading', { level: 1 }).innerText()
  await page.getByRole('button', { name: /Try a numerical example/ }).click()
  const lab = page.getByRole('dialog', { name: 'The mathematics, alive.' })
  await expect(lab.locator('.numerical-step-card h3')).toHaveText('Rotate query feature pairs')
  await expect(lab.locator('.numerical-lesson')).toContainText(
    'not activations from the architecture',
  )
  await lab.getByText('All parameter tensors', { exact: false }).click()
  await lab.getByLabel('Find a numerical parameter').fill('Wq')
  await expect(lab.locator('.parameter-library button')).toHaveCount(2)
  await lab.locator('.parameter-library button').filter({ hasText: 'L0 Wq' }).click()
  await expect(lab.locator('.numeric-current-tensor')).toContainText('parameter · parameter')
  await expect(lab.locator('.tensor-shape-line')).toContainText('[16 × 16]')
  const trace = executeTeachingModel(defaultNumericalSettings)
  await expect(lab.getByTestId('cell-result')).toHaveText(
    valueLabel(trace.tensors['weights/L0 Wq'].data[0], 8),
  )
  await expect(lab.locator('.parameter-library')).toContainText('12,880 values')
  await lab.getByLabel('Close dialog').click()
  await expect(page.getByRole('heading', { level: 1 })).toHaveText(title)
})

test('the numerical reference remains independent of edited or imported architecture templates', async ({
  page,
}) => {
  test.setTimeout(90000)
  await page.goto('/')
  const doc = JSON.parse(
    await readFile(
      new URL('../../examples/transformer-foundations.model.json', import.meta.url),
      'utf8',
    ),
  )
  doc.graphs
    .find((g: { id: string }) => g.id === 'attention')
    .nodes.find((n: { id: string }) => n.id === 'q').title = 'A custom architecture projection'
  doc.sources.find((s: { id: string }) => s.id === 'rope').title = 'A different document source'
  await page.getByLabel('Open model JSON').setInputFiles({
    name: 'edited-architecture.model.json',
    mimeType: 'application/json',
    buffer: Buffer.from(JSON.stringify(doc)),
  })
  await page.getByRole('button', { name: /Tensor laboratory/ }).click()
  const lab = page.getByRole('dialog', { name: 'The mathematics, alive.' })
  await lab.getByRole('button', { name: 'How does RoPE rotate a vector?', exact: true }).click()
  await expect(
    lab.locator('.model-node').filter({ hasText: 'A custom architecture projection' }),
  ).toHaveCount(0)
  await lab.locator('.numeric-architecture summary').click()
  await expect(lab.locator('.numeric-architecture')).toContainText('RoFormer: Enhanced Transformer')
  await expect(lab.locator('.numeric-architecture')).not.toContainText(
    'A different document source',
  )
  await lab.getByLabel('Close dialog').click()
  await expect
    .poll(() =>
      page.evaluate(
        () =>
          JSON.parse(localStorage.getItem('model-visualizer.workspace.v1')!)
            .graphs.find((g: { id: string }) => g.id === 'attention')
            .nodes.find((n: { id: string }) => n.id === 'q').title,
      ),
    )
    .toBe('A custom architecture projection')
})

test('actual attention values, masks, scalar provenance and RoPE stay aligned with playback', async ({
  page,
}) => {
  test.setTimeout(90000)
  const errors: string[] = []
  page.on('pageerror', (e) => errors.push(e.message))
  const lab = await openLab(page),
    trace = executeTeachingModel(defaultNumericalSettings)
  await lab.getByRole('button', { name: 'How is one attention score made?', exact: true }).click()
  await expect(lab.locator('.tensor-view')).toHaveCount(1)
  await expect(lab.getByTestId('cell-result')).toHaveText(
    valueLabel(trace.tensors['call0/l0/scores-0'].data[0], 8),
  )
  await expect(lab.locator('.proof-term')).toHaveCount(4)
  await expect(lab.getByLabel('Matrix contraction diagram')).toBeVisible()
  await lab.locator('[data-flat-index="11"]').click()
  await expect(lab.getByTestId('cell-result')).toHaveText(
    valueLabel(trace.tensors['call0/l0/scores-0'].data[11], 8),
  )
  await lab.locator('.proof-term button').first().click()
  await expect(lab.locator('.numeric-current-tensor')).toContainText('Query head 0')
  await lab.getByRole('button', { name: 'Back through the calculation' }).click()
  await expect(lab.getByTestId('cell-result')).toHaveText(
    valueLabel(trace.tensors['call0/l0/scores-0'].data[11], 8),
  )
  await lab.getByRole('button', { name: 'Why is the future invisible?', exact: true }).click()
  await expect(lab.getByTestId('cell-result')).toHaveText('−∞')
  await expect(lab.locator('[data-flat-index="1"]')).toHaveClass(/numeric-masked/)
  await lab
    .getByRole('button', { name: 'Where does a probability come from?', exact: true })
    .click()
  await lab.locator('[data-flat-index="1"]').click()
  await expect(lab.getByTestId('cell-result')).toHaveText('0')
  await lab.getByRole('button', { name: 'How does RoPE rotate a vector?', exact: true }).click()
  await expect(lab.getByLabel('Rotary position diagram')).toBeVisible()
  await lab.getByLabel('Tensor Hq slice').selectOption('2')
  await expect(lab.getByTestId('cell-result')).toHaveText(
    valueLabel(trace.tensors['call0/l0/rope-q'].data[44], 8),
  )
  await lab.getByRole('button', { name: 'Transpose view', exact: true }).click()
  await expect(lab.locator('[aria-selected="true"]')).toHaveAttribute('data-flat-index', '44')
  await expect(lab.locator('.katex-error')).toHaveCount(0)
  await page.setViewportSize({ width: 1600, height: 1200 })
  await page.screenshot({ path: 'docs/images/numerical-laboratory-dark.png' })
  expect(errors).toEqual([])
})

test('future nodes reveal no values until run, and playback responds to keyboard control', async ({
  page,
}) => {
  test.setTimeout(90000)
  const lab = await openLab(page)
  await lab.getByRole('button', { name: 'Fit graph', exact: true }).click()
  await lab.locator('.react-flow__node[data-id="layer-1.add2"]').click()
  await expect(lab.locator('.numeric-empty')).toContainText('This operation hasn’t run yet.')
  await expect(lab.locator('.tensor-view')).toHaveCount(0)
  await lab.getByRole('button', { name: 'Run to this operation' }).click()
  await expect(lab.locator('.numerical-step-card h3')).toHaveText('Layer 1 complete')
  await expect(lab.locator('.numeric-current-tensor')).toContainText('Layer output residual')
  await lab.locator('.numerical-step-card h3').click()
  await page.keyboard.press('ArrowLeft')
  await expect(lab.locator('.numerical-step-card h3')).toHaveText('Project F → D')
  await page.keyboard.press('Space')
  await expect(lab.getByRole('button', { name: 'Pause', exact: true })).toBeVisible()
  await expect(lab.locator('.numerical-step-card h3')).not.toHaveText('Project F → D')
  await lab.locator('[data-flat-index="0"]').click()
  await expect(lab.getByRole('button', { name: 'Play', exact: true })).toBeVisible()
})

test('a newer laboratory extension remains byte-for-byte preserved during a temporary experiment', async ({
  page,
}) => {
  test.setTimeout(90000)
  await page.goto('/')
  const doc = JSON.parse(
      await readFile(
        new URL('../../examples/transformer-foundations.model.json', import.meta.url),
        'utf8',
      ),
    ),
    future = {
      version: 2,
      notes: 'Keep my future experiment.',
      settings: { model: 'something-new' },
      metadata: { custom: ['untouched'] },
    }
  doc.extensions = { ...doc.extensions, [NUMERICAL_SESSION_KEY]: future }
  await page.getByLabel('Open model JSON').setInputFiles({
    name: 'future-lab.model.json',
    mimeType: 'application/json',
    buffer: Buffer.from(JSON.stringify(doc)),
  })
  await page.getByRole('button', { name: /Tensor laboratory/ }).click()
  await expect(page.locator('.numerical-version-note')).toContainText('will not overwrite it')
  await page.getByLabel('Open numerical lab notes').click()
  await page.getByLabel('Numerical laboratory notes').fill('A temporary note.')
  await page.getByLabel('Close dialog').click()
  await expect
    .poll(() =>
      page.evaluate(
        (key) => JSON.parse(localStorage.getItem('model-visualizer.workspace.v1')!).extensions[key],
        NUMERICAL_SESSION_KEY,
      ),
    )
    .toEqual(future)
})

test('a complete numerical run has independent heads, inspected links and real confirmed IDs', async ({
  page,
}) => {
  test.setTimeout(90000)
  const lab = await openLab(page),
    trace = executeTeachingModel(defaultNumericalSettings)
  await lab.getByLabel('Numerical execution timeline').fill('32')
  await lab.getByText('Attention head observatory', { exact: false }).click()
  await expect(lab.locator('.observatory-head')).toHaveCount(4)
  await lab.getByRole('button', { name: /Head 3, query 0, key 0:/ }).click()
  await expect(lab.locator('.numeric-current-tensor')).toContainText(
    'Head 3 attention probabilities',
  )
  await lab.getByRole('button', { name: 'How is one attention score made?', exact: true }).click()
  await lab.locator('.edge-label').filter({ hasText: 'dot product' }).first().click()
  await expect(lab.locator('.numeric-architecture summary')).toHaveText('About this connection')
  await expect(lab.locator('.numeric-tensor-picker')).toContainText('Query head 0')
  await lab.getByRole('button', { name: 'Finish numerical playback' }).click()
  await expect(lab.locator('.numerical-output-token')).toHaveCount(3)
  for (let i = 0; i < 3; i++)
    await expect(lab.locator('.numerical-output-token').nth(i)).toContainText(
      `#${trace.outputIds[i]}`,
    )
  await expect(lab.locator('.numerical-step-card h3')).toHaveText('Numerical run complete')
  const exported = page.waitForEvent('download')
  await lab.getByRole('button', { name: 'Export trace', exact: true }).click()
  const raw = JSON.parse(await readFile((await (await exported).path())!, 'utf8'))
  expect(raw.provenance.kind).toBe('numerical')
  expect(raw.outputIds).toEqual(trace.outputIds)
  expect(raw.tensors['call0/l0/masked-0'].data[1]).toBe('-Infinity')
  expect(raw.tensors['call0/l0/scores-0'].data).toEqual(
    Array.from(trace.tensors['call0/l0/scores-0'].data),
  )
})

test('cache experiments, layouts and the numerical notebook round-trip without changing the main model', async ({
  page,
}) => {
  test.setTimeout(120000)
  const lab = await openLab(page)
  const before = await page.evaluate(() =>
    JSON.parse(localStorage.getItem('model-visualizer.workspace.v1')!),
  )
  await lab.getByLabel('Open numerical run setup').click()
  await lab.getByLabel('Numerical head sharing').selectOption('MQA')
  await lab.getByLabel('KV cache retention').selectOption('sliding-window')
  await lab.getByLabel('KV window size').fill('2')
  await lab.getByRole('button', { name: 'Explore the run' }).click()
  await expect(lab.getByRole('button', { name: 'Play', exact: true })).toBeEnabled({
    timeout: 60000,
  })
  await lab.getByRole('button', { name: 'What survives into the next token?', exact: true }).click()
  await expect(lab.locator('.tensor-shape-line')).toContainText('[1 × 1 × 2 × 4]')
  await expect(lab.locator('.tensor-grid-row > .tensor-index')).toHaveText(['3', '4'])
  await lab.getByRole('button', { name: 'Bookmark', exact: true }).click()
  await expect(lab.getByRole('status')).toContainText('Bookmarked')
  await lab.getByLabel('Open numerical lab notes').click()
  await lab
    .getByLabel('Numerical laboratory notes')
    .fill('Sliding state keeps absolute positions 3 and 4.\nCompare sparse anchors next.')
  await lab.getByRole('button', { name: 'Auto-layout', exact: true }).click()
  await expect(lab.getByRole('status')).toContainText('Layout arranged', { timeout: 60000 })
  await expect
    .poll(async () =>
      Number((await lab.locator('.zoom-controls > span').innerText()).replace('%', '')),
    )
    .toBeLessThan(20)
  await lab.getByLabel('Switch layout direction').click()
  await expect(lab.getByRole('status')).toContainText('Layout arranged', { timeout: 60000 })
  await expect
    .poll(async () =>
      Number((await lab.locator('.zoom-controls > span').innerText()).replace('%', '')),
    )
    .toBeLessThan(20)
  await lab.getByLabel('Close dialog').click()
  const downloaded = page.waitForEvent('download')
  await page.getByRole('button', { name: 'Export model', exact: true }).click()
  const path = (await (await downloaded).path())!,
    saved = JSON.parse(await readFile(path, 'utf8')),
    session = saved.extensions[NUMERICAL_SESSION_KEY]
  expect(saved.config).toEqual(before.config)
  expect(saved.ui.notes).toEqual(before.ui.notes)
  expect(saved.graphs).toEqual(before.graphs)
  expect(session.settings.attention).toBe('MQA')
  expect(session.settings.kvCache).toEqual({ mode: 'sliding-window', windowSize: 2 })
  expect(session.bookmarks).toHaveLength(1)
  expect(Object.keys(session.layout.positions).length).toBeGreaterThan(80)
  await page.getByLabel('Open model JSON').setInputFiles(path)
  await page.reload()
  await page.getByRole('button', { name: /Tensor laboratory/ }).click()
  await page.getByLabel('Open numerical lab notes').click()
  await expect(page.getByLabel('Numerical laboratory notes')).toHaveValue(session.notes)
  await page.locator('.numerical-bookmark > button').first().click()
  await expect(page.locator('.numeric-current-tensor')).toContainText('Retained K state')
})

test('a pinned baseline compares compatible cells and supports light and narrow views', async ({
  page,
}) => {
  test.setTimeout(90000)
  const lab = await openLab(page)
  await lab.getByRole('button', { name: 'Pin baseline', exact: true }).click()
  await lab.getByLabel('Numerical input token 0').fill('8')
  await lab.getByRole('button', { name: 'Input', exact: true }).click()
  await lab.getByRole('button', { name: 'Δ baseline', exact: true }).click()
  const a = executeTeachingModel(defaultNumericalSettings),
    b = executeTeachingModel({ ...defaultNumericalSettings, inputIds: [8, 19, 31, 42] })
  await expect(lab.locator('[data-flat-index="0"]')).toHaveText(
    valueLabel(b.tensors['call0/embedding'].data[0] - a.tensors['call0/embedding'].data[0], 3),
  )
  await lab.getByLabel('Toggle numerical theme').click()
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'light')
  await lab.getByRole('button', { name: 'How is one attention score made?', exact: true }).click()
  await expect(lab.locator('.tensor-view')).toHaveCount(1)
  await page.screenshot({ path: 'docs/images/numerical-laboratory-light.png' })
  await page.setViewportSize({ width: 390, height: 844 })
  await lab.getByRole('button', { name: 'Inspect', exact: true }).click()
  await expect(lab.getByLabel('Selected cell calculation')).toBeVisible()
  expect(await lab.evaluate((el) => el.scrollWidth <= el.clientWidth + 1)).toBe(true)
  await page.screenshot({ path: 'docs/images/numerical-laboratory-mobile.png' })
  await lab.getByRole('button', { name: 'Flow', exact: true }).click()
  await expect(lab.getByLabel('Next numerical operation')).toBeVisible()
})
