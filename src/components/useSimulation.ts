import { useEffect, useMemo, useState } from 'react'
import type { ModelConfig } from '../model/schema'
import { graphicalAdapter, nextFrame, type SimulationOptions } from '../model/simulation'

export function useSimulation(config: ModelConfig, enabled: boolean, splitHeads = false) {
  const [options, setOptions] = useState<SimulationOptions>({
    promptTokens: 6,
    newTokens: 8,
    mtp: true,
    scenario: 'mixed',
  })
  const [index, setIndex] = useState(0),
    [playing, setPlaying] = useState(false)
  const [detail, setDetail] = useState<'operation' | 'stage'>('stage')
  const [delay, setDelay] = useState(650),
    [follow, setFollow] = useState(true)
  const result = useMemo(() => {
    if (!enabled) return { trace: null, error: '' }
    try {
      return { trace: graphicalAdapter.run(config, { ...options, splitHeads }), error: '' }
    } catch (error) {
      return {
        trace: null,
        error: error instanceof Error ? error.message : 'Simulation unavailable.',
      }
    }
  }, [config, options, enabled, splitHeads])
  const trace = result.trace
  const cursor = Math.min(index, (trace?.frames.length ?? 1) - 1)
  const frame = trace?.frames[cursor]
  useEffect(() => {
    setIndex(0)
    setPlaying(false)
  }, [trace])
  useEffect(() => {
    if (!playing || !trace) return
    if (cursor >= trace.frames.length - 1) {
      setPlaying(false)
      return
    }
    const timer = setTimeout(() => setIndex((i) => nextFrame(trace, i, detail)), delay)
    return () => clearTimeout(timer)
  }, [playing, trace, cursor, detail, delay])
  const completed = useMemo(
    () => new Set(trace?.frames.slice(0, cursor).flatMap((f) => f.nodeIds)),
    [trace, cursor],
  )
  return {
    ...result,
    frame,
    cursor,
    playing,
    setPlaying,
    options,
    setOptions,
    detail,
    setDetail,
    delay,
    setDelay,
    follow,
    setFollow,
    completed,
    step: (direction = 1) => {
      setPlaying(false)
      if (trace) setIndex(nextFrame(trace, cursor, detail, direction))
    },
    seek: (index: number) => {
      setPlaying(false)
      setIndex(index)
    },
    reset: () => {
      setPlaying(false)
      setIndex(0)
    },
    finish: () => {
      setPlaying(false)
      if (trace) setIndex(trace.frames.length - 1)
    },
  }
}

export type SimulationController = ReturnType<typeof useSimulation>
