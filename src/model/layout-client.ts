import type { ModelGraph } from './schema'
import type { LayoutPlan } from './layout'
import type { Positions, RoutingResult } from './routing'

export type DiagramRequest = {
  id: number
  type: 'layout' | 'route'
  graph: ModelGraph
  positions?: Positions
  direction: 'RIGHT' | 'DOWN'
}
type QueuedRoute = {
  graph: ModelGraph
  positions: Positions
  direction: 'RIGHT' | 'DOWN'
  resolve: (result: RoutingResult) => void
  reject: (error: Error) => void
}

/** Keep both ELK and native routing off the rendering thread. Coalesce drags. */
export class DiagramWorker {
  private worker = new Worker(new URL('./diagram.worker.ts', import.meta.url), { type: 'module' })
  private nextId = 0
  private pending = new Map<
    number,
    { resolve: (value: never) => void; reject: (error: Error) => void }
  >()
  private activeRoute = false
  private queuedRoute?: QueuedRoute
  private disposed = false
  private failure?: Error

  constructor() {
    this.worker.onmessage = (event) => {
      const { id, result, error } = event.data
      const pending = this.pending.get(id)
      if (!pending) return
      this.pending.delete(id)
      if (error) pending.reject(new Error(error))
      else pending.resolve(result as never)
    }
    this.worker.onerror = () => {
      this.failure = new Error('The layout worker could not start. Reload to try again.')
      this.rejectPending(this.failure)
      this.worker.terminate()
    }
  }

  private request<T>(payload: Omit<DiagramRequest, 'id'>): Promise<T> {
    if (this.failure) return Promise.reject(this.failure)
    if (this.disposed) return Promise.reject(new DOMException('Layout cancelled', 'AbortError'))
    return new Promise((resolve, reject) => {
      const id = ++this.nextId
      this.pending.set(id, { resolve: resolve as (value: never) => void, reject })
      this.worker.postMessage({ ...payload, id })
    })
  }

  layout(graph: ModelGraph, direction = graph.layout.direction) {
    this.queuedRoute?.reject(new DOMException('Layout superseded routing', 'AbortError'))
    this.queuedRoute = undefined
    return this.request<LayoutPlan>({ type: 'layout', graph, direction })
  }

  route(
    graph: ModelGraph,
    positions: Positions,
    direction = graph.layout.direction,
  ): Promise<RoutingResult> {
    return new Promise((resolve, reject) => {
      const job = { graph, positions, direction, resolve, reject }
      if (!this.activeRoute) this.runRoute(job)
      else {
        this.queuedRoute?.reject(new DOMException('Routing superseded', 'AbortError'))
        this.queuedRoute = job
      }
    })
  }

  private runRoute(job: QueuedRoute) {
    this.activeRoute = true
    void this.request<RoutingResult>({
      type: 'route',
      graph: job.graph,
      positions: job.positions,
      direction: job.direction,
    })
      .then(job.resolve, job.reject)
      .finally(() => {
        this.activeRoute = false
        const queued = this.queuedRoute
        this.queuedRoute = undefined
        if (queued && !this.disposed) this.runRoute(queued)
      })
  }

  private rejectPending(error: Error) {
    for (const job of this.pending.values()) job.reject(error)
    this.pending.clear()
    this.queuedRoute?.reject(error)
    this.queuedRoute = undefined
  }

  dispose() {
    this.disposed = true
    this.rejectPending(new DOMException('Layout cancelled', 'AbortError'))
    this.worker.terminate()
  }
}
