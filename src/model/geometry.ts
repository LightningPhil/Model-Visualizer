/** Canvas-space geometry shared by the router, renderer and regression tests. */
export type Point = { x: number; y: number }
export type Rect = Point & { width: number; height: number }
export type Side = 'top' | 'right' | 'bottom' | 'left'
export type Segment = { a: Point; b: Point }
export const EPSILON = 0.01
export const NODE_WIDTH = 246
export const NODE_HEIGHT = 112
export const NODE_CLEARANCE = 12
export const LABEL_CLEARANCE = 10

export function expandRect(rect: Rect, amount: number): Rect {
  return {
    x: rect.x - amount,
    y: rect.y - amount,
    width: rect.width + amount * 2,
    height: rect.height + amount * 2,
  }
}

export function rectanglesOverlap(a: Rect, b: Rect, padding = 0) {
  return (
    a.x < b.x + b.width + padding - EPSILON &&
    a.x + a.width > b.x - padding + EPSILON &&
    a.y < b.y + b.height + padding - EPSILON &&
    a.y + a.height > b.y - padding + EPSILON
  )
}

export function segments(points: Point[]): Segment[] {
  return points.slice(1).map((b, i) => ({ a: points[i], b }))
}

export function segmentLength({ a, b }: Segment) {
  return Math.abs(b.x - a.x) + Math.abs(b.y - a.y)
}

export function segmentHitsRect({ a, b }: Segment, rect: Rect, padding = 0) {
  const r = expandRect(rect, padding)
  if (Math.abs(a.y - b.y) < EPSILON)
    return (
      a.y > r.y + EPSILON &&
      a.y < r.y + r.height - EPSILON &&
      Math.max(a.x, b.x) > r.x + EPSILON &&
      Math.min(a.x, b.x) < r.x + r.width - EPSILON
    )
  if (Math.abs(a.x - b.x) < EPSILON)
    return (
      a.x > r.x + EPSILON &&
      a.x < r.x + r.width - EPSILON &&
      Math.max(a.y, b.y) > r.y + EPSILON &&
      Math.min(a.y, b.y) < r.y + r.height - EPSILON
    )
  throw new Error('A routed segment must be horizontal or vertical.')
}

export function crossing(a: Segment, b: Segment): Point | null {
  const horizontalA = Math.abs(a.a.y - a.b.y) < EPSILON
  const horizontalB = Math.abs(b.a.y - b.b.y) < EPSILON
  if (horizontalA === horizontalB) return null
  const h = horizontalA ? a : b
  const v = horizontalA ? b : a
  const p = { x: v.a.x, y: h.a.y }
  return p.x > Math.min(h.a.x, h.b.x) + EPSILON &&
    p.x < Math.max(h.a.x, h.b.x) - EPSILON &&
    p.y > Math.min(v.a.y, v.b.y) + EPSILON &&
    p.y < Math.max(v.a.y, v.b.y) - EPSILON
    ? p
    : null
}

export function simplifyPoints(points: Point[]): Point[] {
  const result: Point[] = []
  for (const original of points) {
    const point = {
      x: Math.round(original.x * 1000) / 1000,
      y: Math.round(original.y * 1000) / 1000,
    }
    const last = result.at(-1)
    if (last && Math.abs(last.x - point.x) < EPSILON && Math.abs(last.y - point.y) < EPSILON)
      continue
    const before = result.at(-2)
    if (
      last &&
      before &&
      ((Math.abs(before.x - last.x) < EPSILON &&
        Math.abs(last.x - point.x) < EPSILON &&
        (last.y - before.y) * (point.y - last.y) >= 0) ||
        (Math.abs(before.y - last.y) < EPSILON &&
          Math.abs(last.y - point.y) < EPSILON &&
          (last.x - before.x) * (point.x - last.x) >= 0))
    )
      result.pop()
    result.push(point)
  }
  return result
}

/** Round corners only. Labels never insert control points or change the path. */
export function roundedPath(points: Point[], radius = 7): string {
  if (points.length < 2) return ''
  const p = simplifyPoints(points)
  let path = `M ${p[0].x} ${p[0].y}`
  for (let i = 1; i < p.length - 1; i++) {
    const prev = p[i - 1],
      current = p[i],
      next = p[i + 1]
    const before = segmentLength({ a: prev, b: current })
    const after = segmentLength({ a: current, b: next })
    const r = Math.min(radius, before / 2, after / 2)
    const entry = {
      x: current.x - Math.sign(current.x - prev.x) * r,
      y: current.y - Math.sign(current.y - prev.y) * r,
    }
    const exit = {
      x: current.x + Math.sign(next.x - current.x) * r,
      y: current.y + Math.sign(next.y - current.y) * r,
    }
    path += ` L ${entry.x} ${entry.y} Q ${current.x} ${current.y} ${exit.x} ${exit.y}`
  }
  const end = p.at(-1)!
  return `${path} L ${end.x} ${end.y}`
}

export function labelSize(text: string) {
  // IBM Plex Mono is 5.4 px/em at the fixed 9 px label size. Reserve extra
  // space for wider Unicode glyphs and the border; CSS uses these exact bounds.
  const lines = text
    .split('\n')
    .map((line) =>
      [...line].reduce(
        (total, c) => total + (c === '\t' ? 12 : c.codePointAt(0)! > 0x2fff ? 10 : 6),
        0,
      ),
    )
  const width = Math.min(224, Math.max(46, ...lines.map((advance) => advance + 18)))
  const rows = lines.reduce(
    (total, advance) => total + Math.max(1, Math.ceil(advance / (width - 16))),
    0,
  )
  return { width, height: 24 + (rows - 1) * 13 }
}

export function diagramBounds(nodes: Rect[], extra: Rect[] = []): Rect {
  const all = [...nodes, ...extra]
  if (!all.length) return { x: 0, y: 0, width: 1, height: 1 }
  const x = Math.min(...all.map((r) => r.x)),
    y = Math.min(...all.map((r) => r.y))
  return {
    x,
    y,
    width: Math.max(...all.map((r) => r.x + r.width)) - x,
    height: Math.max(...all.map((r) => r.y + r.height)) - y,
  }
}
