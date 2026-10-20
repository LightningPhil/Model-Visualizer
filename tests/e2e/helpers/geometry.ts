import type { Page } from '@playwright/test'

/** Inspect real SVG curves and HTML boxes after transforms, not router metadata. */
export async function visibleGeometry(page: Page, fit = true) {
  return page.evaluate((fit) => {
    const problems: string[] = []
    const root =
      document.querySelector('dialog[open] .graph-canvas') ??
      document.querySelector('.graph-canvas')!
    const canvas = root.getBoundingClientRect()
    const nodes = [...root.querySelectorAll('.model-node')].map((n) => ({
      id: n.closest('[data-id]')!.getAttribute('data-id')!,
      box: n.getBoundingClientRect(),
    }))
    const labels = [...root.querySelectorAll('.edge-label')].map((n) => ({
      id: n.getAttribute('data-label-edge')!,
      box: n.getBoundingClientRect(),
    }))
    for (const label of root.querySelectorAll<HTMLElement>('.edge-label')) {
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
    for (const group of root.querySelectorAll('.routed-connection')) {
      const id = group.getAttribute('data-edge-id')!
      const path = group.querySelector<SVGPathElement>('.react-flow__edge-path')!
      const matrix = path.getScreenCTM()!
      const length = path.getTotalLength()
      // Broad-phase bounds avoid testing every curve sample against hundreds
      // of distant components on expanded models. Sampling remains <= 1 canvas
      // pixel, including rounded corners, with the same collision tolerance.
      const bounds = path.getBoundingClientRect()
      const near = (box: DOMRect) =>
        box.right >= bounds.left - 0.5 &&
        box.left <= bounds.right + 0.5 &&
        box.bottom >= bounds.top - 0.5 &&
        box.top <= bounds.bottom + 0.5
      const nearbyNodes = nodes.filter((n) => near(n.box))
      const nearbyLabels = labels.filter((l) => l.id !== id && near(l.box))
      // Sample at <= 1 canvas pixel, including the rounded corner geometry.
      for (let d = 0; d <= Math.ceil(length); d++) {
        const p = path.getPointAtLength(Math.min(d, length))
        const x = p.x * matrix.a + p.y * matrix.c + matrix.e,
          y = p.x * matrix.b + p.y * matrix.d + matrix.f
        for (const n of nearbyNodes)
          if (contains(n.box, x, y)) problems.push(`${id} hidden by ${n.id}`)
        for (const l of nearbyLabels)
          if (contains(l.box, x, y)) problems.push(`${id} hidden by label ${l.id}`)
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
