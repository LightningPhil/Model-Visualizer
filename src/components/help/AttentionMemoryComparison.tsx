import { useState, useId, useEffect } from 'react'
import { formatBytes } from '../../model/schema'
import { compareAttentionMemory, type MemoryScenario } from '../../help/wiki/memory-comparison'

function Dimension({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
}: {
  label: string
  value: number
  min: number
  max: number
  step?: number
  onChange: (value: number) => void
}) {
  const id = useId()
  const [draft, setDraft] = useState(String(value))
  useEffect(() => setDraft(String(value)), [value])
  return (
    <div className="wiki-memory-dimension">
      <label htmlFor={id}>{label}</label>
      <input
        id={id}
        aria-label={label}
        type="number"
        value={draft}
        min={min}
        max={max}
        step={step}
        onChange={(e) => {
          setDraft(e.currentTarget.value)
          if (e.currentTarget.value && e.currentTarget.validity.valid)
            onChange(Number(e.currentTarget.value))
        }}
        onBlur={() => setDraft(String(value))}
      />
      <input
        aria-label={`Adjust ${label}`}
        type="range"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </div>
  )
}

export function AttentionMemoryComparison({ onRead }: { onRead: (id: string) => void }) {
  const [scenario, setScenario] = useState<MemoryScenario>({
    layers: 32,
    batch: 1,
    positions: 4096,
    queryHeads: 32,
    kvHeads: 8,
    headWidth: 128,
    scalarBytes: 2,
    window: 1024,
    latentWidth: 512,
    positionalWidth: 64,
    recurrentFeatures: 64,
    fullAttentionLayers: 4,
    quantBits: 4,
  })
  const [selected, setSelected] = useState('gqa')
  const [phase, setPhase] = useState<'prefill' | 'decode'>('prefill')
  const rows = compareAttentionMemory(scenario),
    baseline = rows[0].bytes
  const current = rows.find((r) => r.id === selected)!
  const growth = [128, 512, 2048, 8192, 32768, 131072].map((positions) => {
    const values = compareAttentionMemory({ ...scenario, positions })
    return {
      positions,
      baseline: values[0].bytes,
      selected: values.find((r) => r.id === selected)!.bytes,
    }
  })
  const maximum = Math.max(...growth.flatMap((r) => [r.baseline, r.selected]))
  function update(key: keyof MemoryScenario, value: number) {
    setScenario((old) => ({
      ...old,
      [key]: value,
      ...(key === 'layers'
        ? { fullAttentionLayers: Math.min(value, old.fullAttentionLayers) }
        : {}),
      ...(key === 'queryHeads' ? { kvHeads: Math.min(old.kvHeads, value) } : {}),
    }))
  }
  const dimension = (
    key: keyof MemoryScenario,
    label: string,
    min: number,
    max: number,
    step = 1,
  ) => (
    <Dimension
      key={key}
      label={label}
      value={scenario[key]}
      min={min}
      max={max}
      step={step}
      onChange={(v) => update(key, v)}
    />
  )
  const scoreEntries = phase === 'prefill' ? scenario.positions ** 2 : scenario.positions
  return (
    <div className="wiki-memory-comparison">
      <h4>What does each mechanism actually keep?</h4>
      <p>
        Hypothetical dimensions, shared across the rows where applicable. This estimates persistent
        attention-state payload, not total VRAM, speed or quality. It does not run or convert your
        model.
      </p>
      <div className="wiki-memory-inputs">
        {dimension('positions', 'Comparison context positions', 128, 131072, 128)}
        {dimension('batch', 'Comparison batch size', 1, 64)}
        {dimension('layers', 'Comparison layer count', 1, 128)}
        <label>
          Query heads
          <select
            aria-label="Comparison query heads"
            value={scenario.queryHeads}
            onChange={(e) => update('queryHeads', Number(e.target.value))}
          >
            {[8, 16, 32, 64].map((v) => (
              <option key={v}>{v}</option>
            ))}
          </select>
        </label>
        <label>
          Grouped K/V heads
          <select
            aria-label="Comparison KV heads"
            value={scenario.kvHeads}
            onChange={(e) => update('kvHeads', Number(e.target.value))}
          >
            {[1, 2, 4, 8, 16, 32, 64]
              .filter((v) => v <= scenario.queryHeads)
              .map((v) => (
                <option key={v}>{v}</option>
              ))}
          </select>
        </label>
        <label>
          K/V and latent scalar storage
          <select
            aria-label="Comparison scalar bytes"
            value={scenario.scalarBytes}
            onChange={(e) => update('scalarBytes', Number(e.target.value))}
          >
            <option value="2">2 bytes · 16 bits</option>
            <option value="4">4 bytes · 32 bits</option>
          </select>
        </label>
      </div>
      <details className="wiki-memory-assumptions">
        <summary>Change representation assumptions</summary>
        <div className="wiki-memory-inputs">
          {dimension('headWidth', 'Comparison head feature width', 32, 256, 32)}
          {dimension('window', 'Comparison local window', 128, 32768, 128)}
          {dimension('latentWidth', 'Comparison joint latent width', 32, 2048, 32)}
          {dimension('positionalWidth', 'Comparison positional key width', 8, 256, 8)}
          {dimension('recurrentFeatures', 'Comparison recurrent feature width', 8, 256, 8)}
          {dimension('fullAttentionLayers', 'Comparison hybrid full layers', 0, scenario.layers)}
          <label>
            Packed GQA precision
            <select
              aria-label="Comparison packed bits"
              value={scenario.quantBits}
              onChange={(e) => update('quantBits', Number(e.target.value))}
            >
              {[2, 4, 8].map((v) => (
                <option key={v} value={v}>
                  {v} bits
                </option>
              ))}
            </select>
          </label>
        </div>
        <p>
          The recurrent accumulator is always FP32, with value width equal to the head feature
          width. Packed groups use 64 scalar slots plus 4 metadata bytes, with final-group padding.
          MLA retains one joint latent and one positional key; these widths are fictional, not
          checkpoint specifications.
        </p>
      </details>
      <div
        className="wiki-memory-table-wrap"
        tabIndex={0}
        role="region"
        aria-label="Scrollable attention state comparison"
      >
        <table className="wiki-memory-table">
          <caption>
            Persistent state · select a mechanism to inspect its assumptions. Scroll horizontally on
            narrow screens for all columns.
          </caption>
          <thead>
            <tr>
              <th>Mechanism</th>
              <th>Logical payload</th>
              <th>Compared with MHA</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.id} data-selected={row.id === selected}>
                <th scope="row">
                  <button aria-pressed={row.id === selected} onClick={() => setSelected(row.id)}>
                    {row.name}
                  </button>
                </th>
                <td>{formatBytes(row.bytes)}</td>
                <td>
                  {((row.bytes / baseline) * 100).toFixed(2)}% of MHA
                  <div className="wiki-memory-ratio" aria-hidden="true">
                    <i
                      style={{
                        width: `${Math.min(100, (row.bytes / Math.max(baseline, ...rows.map((r) => r.bytes))) * 100)}%`,
                      }}
                    />
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <section className="wiki-memory-selected" aria-label="Selected attention mechanism">
        <h5>{current.name}</h5>
        <p>
          <strong>Stored state:</strong> {current.state}
        </p>
        <code>{current.formula}</code>
        <p>
          <strong>Tradeoff:</strong> {current.tradeoff}
        </p>
        <button className="button" onClick={() => onRead(current.article)}>
          Read this mechanism in detail
        </button>
      </section>
      <section className="wiki-memory-growth" aria-label="State growth with context">
        <h5>Grow the context; keep every other setting fixed</h5>
        <p>
          Solid bars: full MHA. Striped bars: {current.name}. One shared linear scale; exact values
          remain visible even when a bar is small. These are payload curves, not performance
          benchmarks.
        </p>
        {growth.map((r) => (
          <div className="wiki-memory-growth-row" key={r.positions}>
            <strong>{r.positions.toLocaleString()} positions</strong>
            <span>
              MHA · {formatBytes(r.baseline)}
              <i aria-hidden="true" style={{ width: `${(r.baseline / maximum) * 100}%` }} />
            </span>
            <span className="is-alternative">
              {current.name} · {formatBytes(r.selected)}
              <i aria-hidden="true" style={{ width: `${(r.selected / maximum) * 100}%` }} />
            </span>
          </div>
        ))}
      </section>
      <section className="wiki-memory-workspace" aria-label="Temporary attention matrix comparison">
        <h5>A different memory cost: temporary attention matrices</h5>
        <label>
          Current call
          <select
            aria-label="Comparison execution phase"
            value={phase}
            onChange={(e) => setPhase(e.target.value as 'prefill' | 'decode')}
          >
            <option value="prefill">Fresh full-prefix prefill · T = S</option>
            <option value="decode">Single-position decode · T = 1</option>
          </select>
        </label>
        <p>
          One head, one sequence, one layer: one materialized score or probability matrix has{' '}
          <strong>
            {scoreEntries.toLocaleString()} entries ·{' '}
            {formatBytes(scoreEntries * scenario.scalarBytes)}
          </strong>{' '}
          at the chosen scalar size. This is a separate component estimate, not a measured peak or
          an amount to multiply by layers and add blindly.
        </p>
        <p>
          <strong>FlashAttention:</strong> avoids the full stored matrix through tiling and online
          normalization; full-context K/V state still grows. <strong>PagedAttention:</strong>{' '}
          manages physical blocks and sharing; unique logical state still needs storage.
        </p>
        <div className="wiki-scenario-presets">
          <button onClick={() => onRead('storage-kernels')}>Understand kernels and paging</button>
          <button onClick={() => onRead('prefix-sharing')}>Understand shared prefixes</button>
        </div>
      </section>
    </div>
  )
}
