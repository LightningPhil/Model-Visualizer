import type { NumericalTrace } from './engine'

/** Clone the snapshot into a short-lived serializer worker. UI-owned buffers
 * must not be transferred/detached: the microscope can keep reading them. */
export function prepareTraceExport(trace: NumericalTrace): {
  promise: Promise<Blob>
  cancel: () => void
} {
  const worker = new Worker(new URL('./trace.worker.ts', import.meta.url), { type: 'module' })
  let rejectPending: (error: Error) => void = () => {}
  const promise = new Promise<Blob>((resolve, reject) => {
    rejectPending = reject
    worker.onmessage = (event: MessageEvent<{ blob?: Blob; error?: string }>) => {
      worker.terminate()
      if (event.data.blob) resolve(event.data.blob)
      else reject(new Error(event.data.error ?? 'Trace export failed.'))
    }
    worker.onerror = () => {
      worker.terminate()
      reject(new Error('The trace export worker failed.'))
    }
    worker.postMessage({ trace })
  })
  return {
    promise,
    cancel: () => {
      worker.terminate()
      rejectPending(new DOMException('Export cancelled', 'AbortError'))
    },
  }
}
