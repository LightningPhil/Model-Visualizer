import { AvoidLib } from 'libavoid-js'
import wasmUrl from 'libavoid-wasm?url'
import elkWorkerUrl from 'elkjs/lib/elk-worker.min.js?url'
import { configureLayoutWorker, planLayout } from './layout'
import { routeGraph, routingProblems } from './routing'
import type { DiagramRequest } from './layout-client'

const ready = Promise.all([AvoidLib.load(wasmUrl), configureLayoutWorker(elkWorkerUrl)]).then(() =>
  AvoidLib.getInstance(),
)

self.onmessage = async (event: MessageEvent<DiagramRequest>) => {
  const request = event.data
  try {
    const engine = await ready
    if (request.type === 'layout') {
      const result = await planLayout(engine, request.graph, request.direction)
      self.postMessage({ id: request.id, result })
    } else {
      const result = routeGraph(engine, request.graph, request.positions!, request.direction)
      const problems = routingProblems(request.graph, request.positions!, result)
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
