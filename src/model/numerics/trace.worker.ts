import { executeTeachingModel, type NumericalSettings, type NumericalTrace } from './engine'
import { exportNumericalTrace } from './session'

export type TraceRequest = { settings: NumericalSettings } | { trace: NumericalTrace }

self.onmessage = (event: MessageEvent<TraceRequest>) => {
  try {
    if ('settings' in event.data) {
      const trace = executeTeachingModel(event.data.settings)
      const buffers = [...new Set(Object.values(trace.tensors).map((tensor) => tensor.data.buffer))]
      self.postMessage({ trace }, { transfer: buffers })
    } else {
      const blob = new Blob([exportNumericalTrace(event.data.trace)], { type: 'application/json' })
      self.postMessage({ blob })
    }
  } catch (error) {
    self.postMessage({ error: error instanceof Error ? error.message : 'Numerical work failed.' })
  }
}
