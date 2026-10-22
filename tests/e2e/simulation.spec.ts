import { expect, test, type Page } from '@playwright/test'
import starter from '../../examples/transformer-foundations.model.json' with { type: 'json' }
import { graphicalAdapter } from '../../src/model/simulation'
import type { ModelConfig } from '../../src/model/schema'

const trace = graphicalAdapter.run(starter.config as ModelConfig, {
  promptTokens: 6,
  newTokens: 8,
  mtp: true,
  scenario: 'mixed',
})
const frameAt = (predicate: (frame: (typeof trace.frames)[number]) => boolean) =>
  trace.frames.findIndex(predicate)
async function openSimulation(page: Page) {
  await page.goto('/')
  await page
    .getByRole('button', { name: 'Run simulation Follow the execution', exact: true })
    .click()
  await expect(page.getByRole('button', { name: 'Play', exact: true })).toBeEnabled({
    timeout: 60000,
  })
  await expect(page.locator('.model-node')).toHaveCount(252)
  await expect(page.locator('.routed-connection')).toHaveCount(358)
}

test('playback stays at the chosen view and optional follow preserves zoom across both renderers', async ({
  page,
}) => {
  test.setTimeout(90000)
  await page.addInitScript(() => {
    const NativeWorker = Worker
    window.Worker = class extends NativeWorker {
      postMessage(message: any, options?: Transferable[] | StructuredSerializeOptions) {
        if (message.type === 'scene')
          (
            window as Window & {
              playbackScene?: {
                active: string[]
                playing: boolean
              }
            }
          ).playbackScene = {
            active: message.scene.nodes
              .filter((node: { execution?: string }) => node.execution === 'active')
              .map((node: { id: string }) => node.id),
            playing: message.scene.edges.some(
              (edge: { active?: boolean; playing?: boolean }) => edge.active && edge.playing,
            ),
          }
        if (Array.isArray(options)) super.postMessage(message, options)
        else super.postMessage(message, options)
      }
    }
  })
  await openSimulation(page)
  const canvas = page.getByTestId('graph-canvas'),
    viewport = canvas.locator('.react-flow__viewport'),
    timeline = page.getByLabel('Simulation timeline'),
    follow = page.getByRole('checkbox', { name: 'Follow the active operation', exact: true }),
    proposal = frameAt((f) => f.nodeIds.includes('generation.head2')),
    rollback = frameAt((f) => f.phase === 'rollback'),
    camera = () =>
      viewport.evaluate((element) => {
        const matrix = new DOMMatrix(getComputedStyle(element).transform)
        return { x: matrix.e, y: matrix.f, zoom: matrix.a }
      })
  await expect(follow).not.toBeChecked()
  await page.getByRole('button', { name: 'Fit graph', exact: true }).click()
  await expect(canvas).toHaveAttribute('data-renderer', 'overview')
  await page.waitForTimeout(400)
  const wholeView = await camera()
  expect(wholeView.zoom).toBeLessThan(0.18)
  await timeline.fill(String(proposal))
  await expect(page.locator('.model-node[data-execution="active"]')).toHaveCount(1)
  await expect
    .poll(() =>
      page.evaluate(
        () => (window as Window & { playbackScene?: { active: string[] } }).playbackScene?.active,
      ),
    )
    .toEqual(['generation.head2'])
  expect(await camera()).toEqual(wholeView)
  await page.getByLabel('Playback speed').selectOption('250')
  await page.getByRole('button', { name: 'Play', exact: true }).click()
  await expect
    .poll(() =>
      page.evaluate(
        () => (window as Window & { playbackScene?: { playing: boolean } }).playbackScene?.playing,
      ),
    )
    .toBe(true)
  await expect.poll(() => timeline.inputValue()).not.toBe(String(proposal))
  await page.getByRole('button', { name: 'Pause', exact: true }).click()
  expect(await camera()).toEqual(wholeView)

  await page.getByRole('button', { name: 'Actual size', exact: true }).click()
  await expect.poll(async () => (await camera()).zoom).toBe(1)
  await expect(canvas).toHaveAttribute('data-renderer', 'svg')
  const detailView = await camera()
  await timeline.fill(String(rollback))
  await page.getByRole('button', { name: 'Next simulation step' }).click()
  expect(await camera()).toEqual(detailView)
  const beforePlay = await timeline.inputValue()
  await page.getByRole('button', { name: 'Play', exact: true }).click()
  await expect.poll(() => timeline.inputValue()).not.toBe(beforePlay)
  await page.getByRole('button', { name: 'Zoom out', exact: true }).click()
  await expect.poll(async () => (await camera()).zoom).toBeCloseTo(1 / 1.2, 5)
  const runningView = await camera(),
    beforeZoomedStep = await timeline.inputValue()
  await expect.poll(() => timeline.inputValue()).not.toBe(beforeZoomedStep)
  await page.getByRole('button', { name: 'Pause', exact: true }).click()
  expect(await camera()).toEqual(runningView)

  await follow.check()
  await page.getByRole('button', { name: 'Zoom out', exact: true }).click()
  await page.waitForTimeout(400)
  const chosenZoom = (await camera()).zoom
  await timeline.fill(String(proposal))
  await page.waitForTimeout(400)
  expect((await camera()).zoom).toBeCloseTo(chosenZoom, 6)
  await page.getByRole('button', { name: 'Fit graph', exact: true }).click()
  await page.waitForTimeout(400)
  const overviewZoom = (await camera()).zoom
  await timeline.fill(String(rollback))
  await page.waitForTimeout(400)
  expect((await camera()).zoom).toBeCloseTo(overviewZoom, 6)
  await expect(canvas).toHaveAttribute('data-renderer', 'overview')
  const active = (await canvas
      .locator('.react-flow__node[data-id="generation.reject"]')
      .boundingBox())!,
    bounds = (await canvas.boundingBox())!
  expect(Math.abs(active.x + active.width / 2 - bounds.x - bounds.width / 2)).toBeLessThan(2)
  expect(Math.abs(active.y + active.height / 2 - bounds.y - bounds.height / 2)).toBeLessThan(2)
  await follow.uncheck()
  const stoppedCamera = await camera()
  await timeline.fill('1')
  await page.waitForTimeout(400)
  expect(await camera()).toEqual(stoppedCamera)
})

test('whole model exposes every layer and supports full-map fit, jump, inspection and reload', async ({
  page,
}) => {
  test.setTimeout(90000)
  const errors: string[] = []
  page.on('pageerror', (e) => errors.push(e.message))
  await page.goto('/')
  await page
    .getByRole('button', { name: 'Whole model Every layer on one canvas', exact: true })
    .click()
  await expect(page.locator('.model-node')).toHaveCount(252)
  await expect(page.locator('.routed-connection')).toHaveCount(358, { timeout: 60000 })
  await expect(page.locator('.routing-message')).toHaveCount(0)
  await page.getByRole('button', { name: 'Fit graph', exact: true }).click()
  await expect
    .poll(() =>
      page.locator('.model-node').evaluateAll((nodes) => {
        const canvas = document.querySelector('.graph-canvas')!.getBoundingClientRect()
        return nodes.every((n) => {
          const r = n.getBoundingClientRect()
          return (
            r.left >= canvas.left &&
            r.right <= canvas.right &&
            r.top >= canvas.top &&
            r.bottom <= canvas.bottom
          )
        })
      }),
    )
    .toBe(true)
  await page.screenshot({ path: 'docs/images/whole-model.png' })
  await page.getByLabel('Jump to model stage').selectOption('Layer 05')
  await expect
    .poll(async () => {
      const n = (await page.locator('.react-flow__node[data-id="layer-5.input"]').boundingBox())!
      const c = (await page.getByTestId('graph-canvas').boundingBox())!
      return Math.abs(n.x + n.width / 2 - c.x - c.width / 2) < 10
    })
    .toBe(true)
  await page.locator('.react-flow__node[data-id="layer-5.input"]').click()
  await expect(
    page.getByRole('heading', { name: 'L5 · Residual stream', exact: true }),
  ).toBeVisible()
  await page.getByRole('button', { name: 'Search components' }).click()
  await page.getByLabel('Search model components').fill('L5 · Query head 7')
  await page.getByRole('button', { name: /^L5 · Query head 7 Whole model · Q7/ }).click()
  await expect(page.getByRole('heading', { name: 'L5 · Query head 7', exact: true })).toBeVisible()
  await expect(page.locator('.katex-error')).toHaveCount(0)
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
  await page.reload()
  await expect(page.locator('.routed-connection')).toHaveCount(358, { timeout: 60000 })
  await expect(page.getByRole('heading', { level: 1 })).toHaveText('Every layer. Every connection.')
  expect(errors).toEqual([])
})

test('graphical MTP playback shows proposals, checks, rollback and confirmed packets without text input', async ({
  page,
}) => {
  test.setTimeout(90000)
  await openSimulation(page)
  await expect(page.getByLabel('PROMPT', { exact: true })).toHaveCount(0)
  await page
    .getByLabel('Simulation timeline')
    .fill(String(frameAt((f) => f.nodeIds.includes('generation.head2'))))
  await expect(page.locator('.proposal-tray [data-packet-status="proposed"]')).toHaveCount(3)
  await expect(page.locator('.model-node[data-execution="active"]')).toHaveCount(1)
  await expect(page.locator('.execution-packet')).not.toHaveCount(0)
  await page
    .getByLabel('Simulation timeline')
    .fill(String(frameAt((f) => f.proposals.some((p) => p.status === 'rejected'))))
  await expect(page.locator('.proposal-tray [data-packet-status="accepted"]')).toHaveCount(2)
  await expect(page.locator('.proposal-tray [data-packet-status="rejected"]')).toHaveCount(1)
  await page.getByLabel('Simulation timeline').fill(String(frameAt((f) => f.phase === 'rollback')))
  await expect(page.locator('.execution-card h3')).toHaveText('Roll back tentative K/V')
  await expect(page.locator('.cache-layer-list b')).toHaveText(Array(6).fill('8'))
  await page.screenshot({ path: 'docs/images/simulation.png' })
  await page.getByRole('button', { name: 'Run to completion' }).click()
  await expect(page.getByTestId('generated-count')).toHaveText('8 / 8')
  await expect(page.locator('.packet-stream [data-packet-status="rejected"]')).toHaveCount(0)
  await expect(page.locator('.packet-stream-group').last().locator('.packet-glyph')).toHaveCount(8)
  await page.getByRole('button', { name: 'Reset simulation' }).click()
  await expect(page.getByTestId('generated-count')).toHaveText('0 / 8')
  await page.getByLabel('Playback speed').selectOption('250')
  await page.getByRole('button', { name: 'Play', exact: true }).click()
  await expect.poll(() => page.getByLabel('Simulation timeline').inputValue()).not.toBe('0')
  await page.getByRole('button', { name: 'Pause', exact: true }).click()
  await expect(page.getByRole('button', { name: 'Play', exact: true })).toBeEnabled()
})

test('ordinary decoding, detail steps, light mode and narrow-screen controls stay usable', async ({
  page,
}) => {
  test.setTimeout(90000)
  await openSimulation(page)
  await page.getByText('Run settings', { exact: false }).click()
  await page.getByRole('button', { name: 'Autoregressive', exact: true }).click()
  await page.getByLabel('Simulation step detail').selectOption('operation')
  await page.getByRole('button', { name: 'Next simulation step' }).click()
  await expect(page.getByLabel('Simulation timeline')).toHaveValue('1')
  await page.getByRole('button', { name: 'Previous simulation step' }).click()
  await expect(page.getByLabel('Simulation timeline')).toHaveValue('0')
  await page.getByRole('button', { name: 'Run to completion' }).click()
  await expect(page.getByTestId('generated-count')).toHaveText('8 / 8')
  await expect(page.getByLabel('Verification scenario')).toHaveCount(0)
  await page.getByRole('button', { name: 'Switch to light mode' }).click()
  await page.screenshot({ path: 'docs/images/simulation-light.png' })
  await page.setViewportSize({ width: 390, height: 844 })
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(390)
  await page.getByRole('button', { name: 'Close simulator', exact: true }).click()
  await expect(page.getByLabel('Execution simulator')).toHaveCount(0)
  await page.getByRole('button', { name: 'Toggle inspector', exact: true }).click()
  await expect(page.getByLabel('Execution simulator')).toBeVisible()
})
