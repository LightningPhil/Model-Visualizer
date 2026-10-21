/** Reproducible, isolated production benchmark. Start a preview on port 5201,
 * then npm run benchmark -- --label after. Does not use the user's browser. */
import { chromium } from '@playwright/test'
import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { dirname } from 'node:path'

const argument = (name: string, fallback: string) => {
  const index = process.argv.indexOf(`--${name}`)
  return index < 0 ? fallback : process.argv[index + 1]
}
const url = argument('url', 'http://127.0.0.1:5201/')
if (!['127.0.0.1', 'localhost', '[::1]'].includes(new URL(url).hostname))
  throw new Error('Benchmark a local development or preview server.')
const label = argument('label', 'current')
const output = argument('output', `verification.local/benchmark-${label}.json`)
await mkdir(dirname(output), { recursive: true })
const browser = await chromium.launch({ channel: process.env.CI ? undefined : 'chrome' })
try {
  const context = await browser.newContext({ viewport: { width: 1440, height: 960 } })
  const page = await context.newPage()
  const document = JSON.parse(
    await readFile(
      new URL('../examples/transformer-foundations.model.json', import.meta.url),
      'utf8',
    ),
  )
  document.ui.activeGraph = 'whole-model-heads'
  await page.addInitScript(
    (doc) => localStorage.setItem('model-visualizer.workspace.v1', JSON.stringify(doc)),
    document,
  )
  if (process.argv.includes('--svg'))
    await page.addInitScript(() => {
      Object.defineProperty(HTMLCanvasElement.prototype, 'transferControlToOffscreen', {
        value: undefined,
      })
    })
  const errors: string[] = []
  page.on('pageerror', (error) => errors.push(error.message))
  await page.goto(url)
  await page.locator('.routed-connection').nth(609).waitFor({ timeout: 120000 })
  if (process.argv.includes('--detail'))
    await page.getByRole('button', { name: 'Actual size', exact: true }).click()
  await page.waitForTimeout(1200)
  const session = await context.newCDPSession(page)
  const browserSession = await browser.newBrowserCDPSession()
  const { gpu } = await browserSession.send('SystemInfo.getInfo')
  await session.send('Performance.enable')
  if (process.argv.includes('--profile')) await session.send('Profiler.enable')
  const runs = []
  for (let run = 0; run < 3; run++) {
    await page.evaluate(() => {
      const sample = {
        frames: [] as number[],
        longTasks: [] as number[],
        nodeMutations: 0,
        last: performance.now(),
        running: true,
      }
      const loop = (now: number) => {
        if (!sample.running) return
        sample.frames.push(now - sample.last)
        sample.last = now
        requestAnimationFrame(loop)
      }
      requestAnimationFrame(loop)
      const longTasks = new PerformanceObserver((list) =>
        sample.longTasks.push(...list.getEntries().map((entry) => entry.duration)),
      )
      longTasks.observe({ type: 'longtask' })
      const mutations = new MutationObserver((list) => {
        sample.nodeMutations += list.length
      })
      mutations.observe(window.document.querySelector('.react-flow__nodes')!, {
        subtree: true,
        attributes: true,
        childList: true,
      })
      Object.assign(window, {
        endBenchmark: () => {
          sample.running = false
          longTasks.disconnect()
          mutations.disconnect()
          return sample
        },
      })
    })
    const baseline = await session.send('Performance.getMetrics')
    if (!run && process.argv.includes('--profile')) await session.send('Profiler.start')
    const box = (await page.locator('.graph-canvas').boundingBox())!
    const x = box.x + box.width / 2,
      y = box.y + box.height / 2
    await page.mouse.move(x, y)
    await page.mouse.down({ button: 'middle' })
    const start = performance.now()
    for (let i = 1; i <= 120; i++)
      await page.mouse.move(
        x + Math.sin((i / 120) * Math.PI * 2) * 170,
        y + Math.sin((i / 120) * Math.PI * 4) * 110,
      )
    const elapsedMs = performance.now() - start
    await page.mouse.up({ button: 'middle' })
    const final = await session.send('Performance.getMetrics')
    if (!run && process.argv.includes('--profile')) {
      const { profile } = await session.send('Profiler.stop')
      await writeFile(`${output}.cpuprofile`, JSON.stringify(profile))
    }
    const sample = await page.evaluate(() =>
      (
        window as Window & {
          endBenchmark: () => { frames: number[]; longTasks: number[]; nodeMutations: number }
        }
      ).endBenchmark(),
    )
    const difference = (name: string) =>
      (final.metrics.find((m) => m.name === name)!.value -
        baseline.metrics.find((m) => m.name === name)!.value) *
      1000
    const frames = sample.frames.slice(2).sort((a, b) => a - b)
    runs.push({
      elapsedMs,
      scriptMs: difference('ScriptDuration'),
      taskMs: difference('TaskDuration'),
      layoutMs: difference('LayoutDuration'),
      frames: frames.length,
      frameP50: frames[Math.floor(frames.length * 0.5)],
      frameP95: frames[Math.floor(frames.length * 0.95)],
      longTaskCount: sample.longTasks.length,
      longTaskMs: sample.longTasks.reduce((a, b) => a + b, 0),
      nodeMutations: sample.nodeMutations,
    })
    await page.waitForTimeout(800)
  }
  const result = {
    label,
    userAgent: await page.evaluate(() => navigator.userAgent),
    renderer: (await page.locator('.graph-canvas').getAttribute('data-renderer')) ?? 'svg',
    nodes: await page.locator('.model-node').count(),
    edges: await page.locator('.routed-connection').count(),
    gpu: { devices: gpu.devices, featureStatus: gpu.featureStatus },
    runs,
    errors,
  }
  await writeFile(output, `${JSON.stringify(result, null, 2)}\n`)
  console.log(JSON.stringify(result, null, 2))
} finally {
  await browser.close()
}
