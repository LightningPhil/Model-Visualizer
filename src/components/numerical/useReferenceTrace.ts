import { useEffect, useState } from 'react'
import type { NumericalSettings, NumericalTrace } from '../../model/numerics/engine'

/** A superseded run is terminated, including its computation. Independent
 * current/baseline workers let both run alongside layout and the UI thread. */
export function useReferenceTrace(settings?: NumericalSettings) {
  const key = settings ? JSON.stringify(settings) : ''
  const [result, setResult] = useState<{ key: string; trace?: NumericalTrace; error?: string }>()
  const [attempt, retry] = useState(0)
  useEffect(() => {
    if (!settings) return
    let cancelled = false
    let worker: Worker | undefined
    const timer = setTimeout(() => {
      try {
        worker = new Worker(new URL('../../model/numerics/trace.worker.ts', import.meta.url), {
          type: 'module',
        })
        worker.onmessage = (event: MessageEvent<{ trace?: NumericalTrace; error?: string }>) => {
          if (!cancelled) setResult({ key, ...event.data })
          worker?.terminate()
        }
        worker.onerror = () => {
          if (!cancelled) setResult({ key, error: 'The numerical worker failed. Retry the run.' })
          worker?.terminate()
        }
        worker.postMessage({ settings })
      } catch (error) {
        setResult({
          key,
          error: error instanceof Error ? error.message : 'The numerical worker could not start.',
        })
      }
    }, 60)
    return () => {
      cancelled = true
      clearTimeout(timer)
      worker?.terminate()
    }
    // Serialize only these tiny settings, never tensors. Equivalent settings
    // do not recompute when camera, notebook or playback state changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, attempt])
  return {
    trace: settings ? result?.trace : undefined,
    error: result?.key === key ? result.error : undefined,
    busy: Boolean(settings && result?.key !== key),
    retry: () => {
      setResult(undefined)
      retry((n) => n + 1)
    },
  }
}
