import { AvoidLib } from 'libavoid-js'
import wasmUrl from 'libavoid-wasm?url'
import elkWorkerUrl from 'elkjs/lib/elk-worker.min.js?url'
import { configureLayoutWorker, planLayout } from './layout'
import { routeGraph, routingProblems } from './routing'
import { routeGrouped, usesStageLayout } from './group-layout'
import type { DiagramRequest } from './layout-client'
import type { ModelGraph } from './schema'

const ready = Promise.all([AvoidLib.load(wasmUrl), configureLayoutWorker(elkWorkerUrl)]).then(() =>
  AvoidLib.getInstance(),
)

let currentGraph: ModelGraph | undefined
self.onmessage = async (event: MessageEvent<DiagramRequest>) => {
  const request = event.data
  try {
    if (request.graph) currentGraph = request.graph
    // Capture before awaiting initialization/layout: a later message can update
    // the worker's cache while this request is suspended.
    const graph = currentGraph
    if (!graph) throw new Error('The diagram worker needs model geometry before routing.')
    const engine = await ready
    if (request.type === 'layout') {
      const result = await planLayout(engine, graph, request.direction)
      self.postMessage({ id: request.id, result })
    } else {
      const result = (usesStageLayout(graph) ? routeGrouped : routeGraph)(
        engine,
        graph,
        request.positions!,
        request.direction,
      )
      const problems = routingProblems(graph, request.positions!, result)
      if (problems.length)
        throw new Error(
          'Connections need more room. Move overlapping components apart or use Auto-layout.',
        )
      self.postMessage({ id: request.id, result })
    }
  } catch (error) {
    self.postMessage({
      id: request.id,
      error: error instanceof Error ? error.message : 'Diagram routing failed.',
    })
  }
}
