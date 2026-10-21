import {
  OVERVIEW_ZOOM,
  type OverviewPalette,
  type OverviewRequest,
  type OverviewScene,
  type OverviewView,
} from './scene'

let canvas: OffscreenCanvas | undefined
let context: OffscreenCanvasRenderingContext2D | null = null
let scene: OverviewScene | undefined,
  palette: OverviewPalette | undefined,
  view: OverviewView | undefined
let revision = 0,
  scheduled = false
let paths: { path: Path2D; leader?: Path2D }[] = []

function schedule() {
  if (scheduled) return
  scheduled = true
  // Dedicated-worker rAF is supported with transferred canvases. A timer is
  // sufficient on browsers that expose OffscreenCanvas without worker rAF.
  if (typeof requestAnimationFrame === 'function') requestAnimationFrame(draw)
  else setTimeout(() => draw(performance.now()), 16)
}

function draw(now: number) {
  scheduled = false
  if (!canvas || !context || !scene || !palette || !view || view.zoom >= OVERVIEW_ZOOM) return
  const ctx = context,
    colors = palette,
    z = view.zoom
  const width = Math.max(1, Math.round(view.width * view.dpr)),
    height = Math.max(1, Math.round(view.height * view.dpr))
  if (canvas.width !== width) canvas.width = width
  if (canvas.height !== height) canvas.height = height
  ctx.resetTransform()
  ctx.clearRect(0, 0, width, height)
  ctx.setTransform(view.dpr * z, 0, 0, view.dpr * z, view.dpr * view.x, view.dpr * view.y)
  ctx.lineJoin = 'round'
  ctx.lineCap = 'round'
  const rect = (box: { x: number; y: number; width: number; height: number }, radius: number) => {
    ctx.beginPath()
    ctx.roundRect(box.x, box.y, box.width, box.height, radius)
  }
  const visible = (box: { x: number; y: number; width: number; height: number }) =>
    box.x * z + view!.x <= view!.width &&
    (box.x + box.width) * z + view!.x >= 0 &&
    box.y * z + view!.y <= view!.height &&
    (box.y + box.height) * z + view!.y >= 0
  for (const [name, box] of scene.stages) {
    if (!visible(box)) continue
    ctx.lineWidth = Math.min(2, Math.max(1, 0.35 / z))
    ctx.strokeStyle = colors.border
    rect(box, 14)
    ctx.stroke()
    if (20 * z >= 2.5) {
      ctx.fillStyle = colors.muted
      ctx.font = '20px sans-serif'
      ctx.fillText(name, box.x + 14, box.y + 28)
    }
  }
  scene.edges.forEach((edge, i) => {
    ctx.strokeStyle =
      edge.selected || edge.active
        ? colors.cyan
        : edge.kind === 'state'
          ? colors.violet
          : edge.kind === 'auxiliary'
            ? colors.amber
            : edge.kind === 'residual'
              ? colors.cyan
              : colors.edge
    // Heavier overview strokes stay bounded in canvas units so they cannot
    // erase the router's clearances at extreme zoom levels.
    ctx.lineWidth = Math.min(
      6,
      Math.max(
        edge.selected || edge.active ? 2.2 : 1.4,
        (edge.selected || edge.active ? 0.9 : 0.4) / z,
      ),
    )
    ctx.setLineDash(edge.kind === 'flow' ? [] : [5, 5])
    ctx.stroke(paths[i].path)
    if (paths[i].leader) ctx.stroke(paths[i].leader!)
    ctx.setLineDash([])
    const last = edge.points.at(-1)!,
      before = edge.points.at(-2)!
    if (before) {
      const angle = Math.atan2(last.y - before.y, last.x - before.x),
        size = Math.min(14, Math.max(8, 1.8 / z))
      ctx.fillStyle = ctx.strokeStyle
      ctx.beginPath()
      ctx.moveTo(last.x, last.y)
      ctx.lineTo(last.x - Math.cos(angle - 0.4) * size, last.y - Math.sin(angle - 0.4) * size)
      ctx.lineTo(last.x - Math.cos(angle + 0.4) * size, last.y - Math.sin(angle + 0.4) * size)
      ctx.closePath()
      ctx.fill()
    }
    if (visible(edge.label)) {
      ctx.fillStyle = colors.canvas
      ctx.strokeStyle = colors.border
      ctx.lineWidth = Math.min(2, Math.max(1, 0.25 / z))
      rect(edge.label, 4)
      ctx.fill()
      ctx.stroke()
    }
    if (edge.active) {
      const lengths = edge.points
        .slice(1)
        .map((p, j) => Math.hypot(p.x - edge.points[j].x, p.y - edge.points[j].y))
      let distance =
        lengths.reduce((n, length) => n + length, 0) * (edge.playing ? (now % 1000) / 1000 : 0.5)
      for (let j = 0; j < lengths.length; j++) {
        if (distance > lengths[j]) {
          distance -= lengths[j]
          continue
        }
        const a = edge.points[j],
          b = edge.points[j + 1],
          fraction = lengths[j] ? distance / lengths[j] : 0
        ctx.fillStyle = colors.cyan
        ctx.beginPath()
        ctx.arc(
          a.x + (b.x - a.x) * fraction,
          a.y + (b.y - a.y) * fraction,
          Math.max(4, 1.6 / z),
          0,
          Math.PI * 2,
        )
        ctx.fill()
        break
      }
    }
  })
  for (const node of scene.nodes) {
    if (!visible(node)) continue
    const category =
      node.category === 'data'
        ? colors.cyan
        : node.category === 'memory'
          ? colors.violet
          : node.category === 'output'
            ? colors.amber
            : colors.compute
    ctx.fillStyle = colors.panel
    ctx.strokeStyle =
      node.selected || node.execution === 'active' || node.execution === 'done'
        ? colors.cyan
        : colors.border
    ctx.lineWidth = Math.min(
      6,
      Math.max(
        node.selected || node.execution === 'active' ? 3 : 1,
        (node.selected ? 1.2 : 0.55) / z,
      ),
    )
    ctx.setLineDash(node.optional ? [8, 5] : [])
    rect(node, 8)
    ctx.fill()
    ctx.stroke()
    ctx.setLineDash([])
    ctx.fillStyle = category
    ctx.fillRect(node.x + 10, node.y + 12, 6, node.height - 24)
    if (node.execution === 'done') {
      ctx.strokeStyle = colors.cyan
      ctx.lineWidth = 4
      ctx.beginPath()
      ctx.moveTo(node.x + node.width - 34, node.y + 20)
      ctx.lineTo(node.x + node.width - 28, node.y + 26)
      ctx.lineTo(node.x + node.width - 16, node.y + 12)
      ctx.stroke()
    }
    if (13 * z >= 2) {
      ctx.fillStyle = colors.text
      ctx.font = '13px sans-serif'
      ctx.fillText(node.title, node.x + 25, node.y + 62, node.width - 35)
    }
  }
  self.postMessage({ revision, edgeCount: scene.edges.length })
  if (scene.edges.some((edge) => edge.active && edge.playing)) schedule()
}

self.onmessage = (event: MessageEvent<OverviewRequest>) => {
  try {
    const message = event.data
    if (message.type === 'init') {
      canvas = message.canvas
      context = canvas.getContext('2d')
      if (!context) throw new Error('Canvas rendering is unavailable.')
      canvas.addEventListener('contextlost', () => self.postMessage({ error: true }))
    } else if (message.type === 'scene') {
      scene = message.scene
      palette = message.palette
      revision = message.revision
      paths = scene.edges.map((edge) => ({
        path: new Path2D(edge.path),
        leader: edge.leader ? new Path2D(edge.leader) : undefined,
      }))
    } else view = message.view
    schedule()
  } catch {
    self.postMessage({ error: true })
  }
}
