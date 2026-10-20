import type { ModelConfig } from '../model/schema'
import { cachePolicy, type CachePolicy } from '../model/attention'

export function CachePolicyControls({
  config,
  onChange,
}: {
  config: ModelConfig
  onChange: (policy: CachePolicy) => void
}) {
  const p = cachePolicy(config)
  return (
    <div className="cache-policy-controls">
      <label>
        KV retention
        <select
          aria-label="KV cache retention"
          value={p.mode}
          onChange={(e) => {
            const mode = e.target.value as CachePolicy['mode']
            onChange(
              mode === 'full'
                ? { mode }
                : mode === 'sliding-window'
                  ? { mode, windowSize: p.mode === 'full' ? 4 : p.windowSize }
                  : {
                      mode,
                      windowSize: p.mode === 'full' ? 4 : p.windowSize,
                      stride: p.mode === 'sparse' ? p.stride : 4,
                    },
            )
          }}
        >
          <option value="full">Full prefix</option>
          <option value="sliding-window">Sliding window</option>
          <option value="sparse">Sparse · anchors + recent</option>
        </select>
      </label>
      {p.mode !== 'full' && (
        <label>
          Window W
          <input
            type="number"
            aria-label="KV window size"
            min={1}
            max={config.contextLength}
            value={p.windowSize}
            onChange={(e) => {
              const n = Number(e.target.value)
              if (Number.isInteger(n) && n > 0 && n <= config.contextLength)
                onChange({ ...p, windowSize: n })
            }}
          />
        </label>
      )}
      {p.mode === 'sparse' && (
        <label>
          Anchor stride
          <input
            type="number"
            aria-label="Sparse anchor stride"
            min={2}
            max={config.contextLength}
            value={p.stride}
            onChange={(e) => {
              const n = Number(e.target.value)
              if (Number.isInteger(n) && n >= 2 && n <= config.contextLength)
                onChange({ ...p, stride: n })
            }}
          />
        </label>
      )}
      <p>
        {p.mode === 'full'
          ? 'Keep every prior K/V position.'
          : p.mode === 'sliding-window'
            ? 'Keep the latest W positions, including the current one. Position IDs do not wrap.'
            : 'Teaching policy: keep absolute positions divisible by the stride, plus the latest W positions. Anchors grow with context.'}
      </p>
    </div>
  )
}
