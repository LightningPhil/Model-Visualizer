import { useEffect, useLayoutEffect, useMemo, useRef, type RefObject } from 'react'
import { useStoreApi } from '@xyflow/react'
import type { VisualEdge, VisualNode } from '../GraphCanvas'
import type { Rect } from '../../model/geometry'
import {
  OVERVIEW_MIN_NODES,
  OVERVIEW_ZOOM,
  overviewScene,
  staleOverviewRoutes,
  type OverviewScene,
  type OverviewPalette,
  type OverviewRequest,
  type OverviewView,
} from './scene'

export function OverviewCanvas({
  nodes,
  edges,
  stages,
  parent,
  theme,
}: {
  nodes: VisualNode[]
  edges: VisualEdge[]
  stages: [string, Rect][]
  parent: RefObject<HTMLDivElement | null>
  theme: 'dark' | 'light'
}) {
  const host = useRef<HTMLDivElement>(null)
  const store = useStoreApi()
  const renderer = useRef<{ worker: Worker; sendView: () => void; revision: number } | undefined>(
    undefined,
  )
  const scene = useMemo(() => overviewScene(nodes, edges, stages), [nodes, edges, stages])
  const currentScene = useRef<OverviewScene | undefined>(undefined)
  useLayoutEffect(() => {
    // Invalidate an old worker acknowledgement before changed geometry can be
    // painted. A canvas frame from the previous scene cannot restore it.
    if (
      parent.current &&
      (!currentScene.current || staleOverviewRoutes(currentScene.current, scene))
    )
      parent.current.dataset.renderer = 'svg'
    currentScene.current = scene
    if (renderer.current) renderer.current.revision++
  }, [scene, theme, parent])
  useEffect(() => {
    const container = host.current,
      graph = parent.current
    if (!container || !graph) return
    graph.dataset.renderer = 'svg'
    if (
      nodes.length < OVERVIEW_MIN_NODES ||
      !HTMLCanvasElement.prototype.transferControlToOffscreen
    )
      return
    const canvas = document.createElement('canvas')
    container.append(canvas)
    let worker: Worker | undefined,
      unsubscribe: (() => void) | undefined,
      observer: ResizeObserver | undefined
    let frame = 0,
      failed = false
    let currentView: OverviewView
    const fallback = () => {
      failed = true
      graph.dataset.renderer = 'svg'
      worker?.terminate()
      renderer.current = undefined
    }
    try {
      worker = new Worker(new URL('./overview.worker.ts', import.meta.url), { type: 'module' })
      const post = (message: OverviewRequest, transfer: Transferable[] = []) =>
        worker!.postMessage(message, transfer)
      const offscreen = canvas.transferControlToOffscreen()
      post({ type: 'init', canvas: offscreen }, [offscreen])
      const sendView = () => {
        if (failed) return
        const [x, y, zoom] = store.getState().transform
        currentView = {
          x,
          y,
          zoom,
          width: graph.clientWidth,
          height: graph.clientHeight,
          dpr: Math.min(2, window.devicePixelRatio || 1),
        }
        if (zoom >= OVERVIEW_ZOOM && graph.dataset.renderer !== 'svg')
          graph.dataset.renderer = 'svg'
        post({ type: 'view', view: currentView })
      }
      renderer.current = { worker, sendView, revision: 0 }
      worker.onmessage = (
        event: MessageEvent<{ revision?: number; edgeCount?: number; error?: boolean }>,
      ) => {
        if (failed) return
        if (event.data.error) {
          fallback()
          return
        }
        if (
          (event.data.revision === renderer.current?.revision ||
            (currentScene.current?.edges.length === 0 && event.data.edgeCount === 0)) &&
          store.getState().transform[2] < OVERVIEW_ZOOM &&
          graph.dataset.renderer !== 'overview'
        )
          graph.dataset.renderer = 'overview'
      }
      worker.onerror = fallback
      unsubscribe = store.subscribe((state, previous) => {
        if (state.transform === previous.transform) return
        // Return to full SVG immediately on zoom-in, before the worker replies.
        if (state.transform[2] >= OVERVIEW_ZOOM && graph.dataset.renderer !== 'svg')
          graph.dataset.renderer = 'svg'
        if (!frame)
          frame = requestAnimationFrame(() => {
            frame = 0
            sendView()
          })
      })
      observer = new ResizeObserver(sendView)
      observer.observe(graph)
      sendView()
    } catch {
      fallback()
    }
    return () => {
      cancelAnimationFrame(frame)
      unsubscribe?.()
      observer?.disconnect()
      worker?.terminate()
      renderer.current = undefined
      graph.dataset.renderer = 'svg'
      canvas.remove()
    }
    // One renderer per mounted graph. Scene/camera changes update it in place.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [store, parent])
  useEffect(() => {
    const graph = parent.current
    const client = renderer.current
    if (!graph || !client) return
    const revision = ++client.revision
    const frame = requestAnimationFrame(() => {
      if (renderer.current !== client) return
      const styles = getComputedStyle(graph)
      const palette = Object.fromEntries(
        [
          'panel',
          'border',
          'text',
          'muted',
          'canvas',
          'cyan',
          'compute',
          'violet',
          'amber',
          'edge',
        ].map((name) => [name, styles.getPropertyValue(`--${name}`).trim()]),
      ) as OverviewPalette
      client.worker.postMessage({
        type: 'scene',
        scene,
        palette,
        revision,
      } satisfies OverviewRequest)
      client.sendView()
    })
    return () => cancelAnimationFrame(frame)
  }, [scene, theme, parent])
  return <div ref={host} className="overview-canvas" aria-hidden="true" />
}
