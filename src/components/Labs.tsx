import { useState } from 'react'
import {
  ArrowDown,
  ArrowRight,
  Check,
  ChevronLeft,
  ChevronRight,
  Fingerprint,
  Info,
} from 'lucide-react'
import {
  attentionWeights,
  demoTokenLabel,
  demoTokenizer,
  type Tokenization,
} from '../model/tokenizer'
import { cacheBytes, formatBytes, type ModelConfig } from '../model/schema'

type Shared = {
  encoded: Tokenization
  config: ModelConfig
  selectedToken: number
  onToken: (index: number) => void
  decode?: boolean
  layer?: number
  onLayer?: (layer: number) => void
}

export function TokenizerLab({ encoded, selectedToken, onToken }: Shared) {
  const [step, setStep] = useState<number | null>(null)
  const boundedStep = Math.min(step ?? encoded.steps.length, encoded.steps.length)
  const pieces =
    boundedStep === 0
      ? encoded.tokens.flatMap((token) => token.bytes)
      : (encoded.steps[boundedStep - 1]?.pieces ?? [])
  const selected = encoded.tokens[selectedToken]
  return (
    <div className="lab-content tokenizer-lab">
      <div className="lab-heading">
        <div>
          <span className="eyebrow">TOKENIZER LAB</span>
          <h3>Watch the bytes become tokens.</h3>
        </div>
        <span className="quiet-badge">
          <Check size={12} /> Lossless round-trip
        </span>
      </div>
      <div className="merge-controls">
        <button
          className="icon-button"
          aria-label="Previous merge"
          disabled={boundedStep === 0}
          onClick={() => setStep(boundedStep - 1)}
        >
          <ChevronLeft size={15} />
        </button>
        <label>
          Merge <strong>{boundedStep}</strong> / {encoded.steps.length}
          <input
            type="range"
            aria-label="BPE merge step"
            min={0}
            max={encoded.steps.length}
            value={boundedStep}
            onChange={(event) => setStep(Number(event.target.value))}
          />
        </label>
        <button
          className="icon-button"
          aria-label="Next merge"
          disabled={boundedStep === encoded.steps.length}
          onClick={() => setStep(boundedStep + 1)}
        >
          <ChevronRight size={15} />
        </button>
        <span className="merge-stat">
          {encoded.byteLength} bytes <ArrowRight size={12} /> {pieces.length} pieces
        </span>
      </div>
      <div className="merge-pieces" aria-label="Pieces at the selected BPE step">
        {pieces.map((id, i) => (
          <span key={i} className={i % 3 === 0 ? 'piece-violet' : ''} title={`ID ${id}`}>
            {demoTokenLabel(id)}
            <small>{id}</small>
          </span>
        ))}
      </div>
      <div className="lab-footnote">
        {selected && (
          <>
            <Fingerprint size={13} />
            <button
              className="text-button"
              onClick={() => onToken((selectedToken + 1) % encoded.tokens.length)}
            >
              Token {selectedToken}: {selected.label}
            </button>
            <code>UTF-8 [{selected.bytes.join(', ')}]</code>
            <span className="footnote-separator">·</span>
          </>
        )}
        Teaching vocabulary · {demoTokenizer.vocabularySize} entries
      </div>
    </div>
  )
}

export function AttentionLab({ encoded, config, decode = false }: Shared) {
  const [head, setHead] = useState(0)
  const [cell, setCell] = useState<[number, number]>([0, 0])
  const actualHead = Math.min(head, config.nQueryHeads - 1)
  const count = Math.min(encoded.tokens.length + (decode ? 1 : 0), 8)
  const weights = attentionWeights(count, actualHead)
  const kv = Math.floor(actualHead / (config.nQueryHeads / config.nKvHeads))
  const row = decode ? count - 1 : Math.min(cell[0], count - 1)
  const col = Math.min(cell[1], count - 1)
  return (
    <div className="lab-content attention-lab">
      <div className="attention-copy">
        <span className="eyebrow">HEAD EXPLORER</span>
        <h3>{decode ? 'One query. A cached prefix.' : 'Who attends to whom?'}</h3>
        <p>
          {decode
            ? 'One new query attends to the prefix and itself. Earlier K/V are reused.'
            : 'Each row is a query. Columns are keys. Future positions are masked.'}
        </p>
        <div className="head-picker" aria-label="Query head">
          {Array.from({ length: config.nQueryHeads }, (_, h) => (
            <button key={h} aria-pressed={h === actualHead} onClick={() => setHead(h)}>
              Q{h}
            </button>
          ))}
        </div>
        <div className="head-mapping">
          <span>Q{actualHead}</span>
          <ArrowRight size={13} />
          <span className="violet-text">
            K{kv} · V{kv}
          </span>
          <small>{config.nQueryHeads / config.nKvHeads} queries / KV head</small>
        </div>
        <div className="synthetic-note">
          <Info size={12} />
          Synthetic probabilities · no trained weights
        </div>
      </div>
      <div className="heatmap-wrap">
        <span className="heatmap-axis">KEY POSITION →</span>
        <div
          className="heatmap"
          style={{ gridTemplateColumns: `30px repeat(${count}, minmax(21px, 1fr))` }}
        >
          <span />
          {Array.from({ length: count }, (_, i) => (
            <span className="heatmap-index" key={`col-${i}`}>
              {i}
            </span>
          ))}
          {weights.map(
            (values, r) =>
              (!decode || r === count - 1) && (
                <div className="heatmap-row" key={r}>
                  <span className="heatmap-index">{r}</span>
                  {values.map((weight, c) => (
                    <button
                      key={c}
                      aria-label={`Query ${r}, key ${c}: ${c > r ? 'masked' : `${(weight * 100).toFixed(1)} percent`}`}
                      className={`heat-cell ${c > r ? 'masked-cell' : ''} ${row === r && col === c ? 'active-cell' : ''}`}
                      style={
                        c <= r
                          ? {
                              backgroundColor: `color-mix(in srgb, var(--cyan) ${Math.round(8 + weight * 92)}%, var(--heat-base))`,
                            }
                          : {}
                      }
                      onClick={() => setCell([r, c])}
                      title={
                        c > r
                          ? 'Future position: masked'
                          : `Q${actualHead} · (${r},${c}) = ${weight.toFixed(4)}`
                      }
                    >
                      {c > r ? '×' : ''}
                    </button>
                  ))}
                </div>
              ),
          )}
        </div>
        <div className="heatmap-detail">
          q<sub>{row}</sub> → k<sub>{col}</sub>
          <strong>
            {col > row ? 'masked' : `${((weights[row]?.[col] ?? 0) * 100).toFixed(1)}%`}
          </strong>
          <div className="heat-gradient" />
          <span>0 → 1</span>
        </div>
        {encoded.tokens.length + (decode ? 1 : 0) > 8 && (
          <small className="muted">
            Synthetic 8-position example. Full tensor shapes appear above.
          </small>
        )}
      </div>
    </div>
  )
}

export function CacheLab({
  config,
  encoded,
  decode = false,
  layer = 0,
  onLayer = () => {},
}: Shared) {
  const [head, setHead] = useState(0)
  const [type, setType] = useState<'K' | 'V'>('K')
  const [sequence, setSequence] = useState<number | null>(null)
  const actualLayer = Math.min(layer, config.nLayers - 1)
  const actualHead = Math.min(head, config.nKvHeads - 1)
  const length = Math.min(
    sequence ?? encoded.tokens.length + (decode ? 1 : 0),
    config.contextLength,
  )
  return (
    <div className="lab-content cache-lab">
      <div className="cache-copy">
        <span className="eyebrow">STATE EXPLORER</span>
        <h3>A tensor. Five meaningful axes.</h3>
        <div className="cache-controls">
          <label>
            Layer ℓ
            <select
              aria-label="Cache layer"
              value={actualLayer}
              onChange={(e) => onLayer(Number(e.target.value))}
            >
              {Array.from({ length: config.nLayers }, (_, i) => (
                <option key={i} value={i}>
                  {i}
                </option>
              ))}
            </select>
          </label>
          <label>
            KV head
            <select
              aria-label="Cache KV head"
              value={actualHead}
              onChange={(e) => setHead(Number(e.target.value))}
            >
              {Array.from({ length: config.nKvHeads }, (_, i) => (
                <option key={i} value={i}>
                  {i}
                </option>
              ))}
            </select>
          </label>
          <div className="segmented cache-type">
            {(['K', 'V'] as const).map((value) => (
              <button key={value} aria-pressed={type === value} onClick={() => setType(value)}>
                {value}
              </button>
            ))}
          </div>
        </div>
        <label className="sequence-slider">
          Cached positions <strong>{length.toLocaleString()}</strong>
          <input
            aria-label="Cached sequence length"
            type="range"
            min={1}
            max={config.contextLength}
            value={length}
            onChange={(e) => setSequence(Number(e.target.value))}
          />
        </label>
        <div className="cache-size">
          <strong>{formatBytes(cacheBytes(config, length))}</strong>
          <span>logical K + V · all layers · B = 1</span>
        </div>
        <small className="muted">
          At capacity: {formatBytes(cacheBytes(config, config.contextLength))} · metadata excluded
        </small>
      </div>
      <div className="cache-visual">
        <svg
          viewBox="0 0 400 220"
          role="img"
          aria-label={`${type} cache logical slice, layer ${actualLayer}, batch 0, KV head ${actualHead}, ${length} sequence positions by ${config.headDim} features`}
        >
          <defs>
            <linearGradient id="plane" x1="0" y1="0" x2="1" y2="1">
              <stop stopColor="var(--violet)" stopOpacity=".15" />
              <stop offset="1" stopColor="var(--cyan)" stopOpacity=".08" />
            </linearGradient>
          </defs>
          {[3, 2, 1].map((i) => (
            <g key={i} transform={`translate(${i * 11},${-i * 10})`}>
              <path d="M56 78 277 78 339 151 118 151Z" fill="var(--panel)" stroke="var(--border)" />
              <path
                d="M118 151 339 151 339 160 118 160Z"
                fill="var(--surface)"
                stroke="var(--border)"
              />
            </g>
          ))}
          <path
            d="M56 78 277 78 339 151 118 151Z"
            fill="url(#plane)"
            stroke="var(--violet)"
            strokeWidth="1.4"
          />
          {Array.from({ length: 9 }, (_, i) => (
            <line
              key={`x${i}`}
              x1={56 + (i * 221) / 8}
              y1="78"
              x2={118 + (i * 221) / 8}
              y2="151"
              stroke="var(--violet)"
              strokeOpacity=".24"
            />
          ))}
          {Array.from({ length: 7 }, (_, i) => (
            <line
              key={`y${i}`}
              x1={56 + (i * 62) / 6}
              y1={78 + (i * 73) / 6}
              x2={277 + (i * 62) / 6}
              y2={78 + (i * 73) / 6}
              stroke="var(--violet)"
              strokeOpacity=".24"
            />
          ))}
          <path
            d="M56 78 83.6 78 145.6 151 118 151Z"
            fill="var(--cyan)"
            fillOpacity=".22"
            stroke="var(--cyan)"
          />
          <path
            d="M118 151 339 151 339 160 118 160Z"
            fill="var(--violet)"
            fillOpacity=".15"
            stroke="var(--violet)"
            strokeOpacity=".6"
          />
          <path d="M117 180h222m-222-3v6m222-6v6" stroke="var(--muted)" />
          <text x="228" y="200" textAnchor="middle">
            S = {length.toLocaleString()} positions →
          </text>
          <text x="57" y="137" transform="rotate(49 57 137)" textAnchor="middle">
            Dh = {config.headDim}
          </text>
          <text x="216" y="34" textAnchor="middle" className="cache-svg-title">
            {type}[ℓ={actualLayer}, b=0, h={actualHead}, :, :]
          </text>
          <text x="211" y="53" textAnchor="middle">
            one selected layer / head slice
          </text>
        </svg>
        <div className="axis-pills">
          {[
            ['L', config.nLayers],
            ['B', 1],
            ['Hkv', config.nKvHeads],
            ['S', length],
            ['Dh', config.headDim],
          ].map(([axis, value]) => (
            <span key={axis}>
              <small>{axis}</small>
              {value}
            </span>
          ))}
        </div>
        <span className="cache-slice-note">
          Logical indexing · grid is schematic · no activation values
        </span>
      </div>
    </div>
  )
}

function hashNgram(ids: number[]) {
  let hash = 2166136261
  for (const id of ids) hash = Math.imul(hash ^ id, 16777619) >>> 0
  return hash % 4096
}

export function MemoryLab({ encoded, selectedToken, onToken }: Shared) {
  const position = Math.min(Math.max(selectedToken, 0), encoded.tokens.length - 1)
  return (
    <div className="lab-content memory-lab">
      <div className="lab-heading">
        <div>
          <span className="eyebrow">N-GRAM ADDRESS EXPLORER</span>
          <h3>Local patterns, explicit addresses.</h3>
        </div>
        <label className="inline-select">
          Ending at token
          <select
            aria-label="N-gram ending token"
            value={position}
            onChange={(e) => onToken(Number(e.target.value))}
          >
            {encoded.tokens.map((token, i) => (
              <option key={i} value={i}>
                {i}: {token.label}
              </option>
            ))}
          </select>
        </label>
      </div>
      <div className="ngram-cards">
        {[2, 3, 4].map((n) => {
          const tokens = encoded.tokens.slice(Math.max(0, position - n + 1), position + 1)
          return (
            <div className="ngram-card" key={n}>
              <span className="eyebrow">
                {n === 2 ? 'BIGRAM' : n === 3 ? 'TRIGRAM' : 'FOUR-GRAM'}
              </span>
              {tokens.length === n ? (
                <>
                  <div className="ngram-tokens">
                    {tokens.map((t, i) => (
                      <span key={i} title={`Token ID ${t.id}`}>
                        {t.label}
                      </span>
                    ))}
                  </div>
                  <div className="ngram-address">
                    <Fingerprint size={14} />
                    <code>{tokens.map((t) => t.id).join(' · ')}</code>
                    <ArrowRight size={13} />
                    <strong>{hashNgram(tokens.map((t) => t.id))}</strong>
                  </div>
                </>
              ) : (
                <p>Needs {n} prefix tokens.</p>
              )}
            </div>
          )
        })}
      </div>
      <div className="lab-footnote">
        <Info size={12} />
        Demonstration hash · 4,096 slots · collisions possible · not the full Engram address
        function
      </div>
    </div>
  )
}

export function MtpLab({ encoded }: Shared) {
  const [depth, setDepth] = useState(1)
  const enough = encoded.tokens.length >= depth + 2
  return (
    <div className="lab-content mtp-lab">
      <div className="lab-heading">
        <div>
          <span className="eyebrow">TRAINING ALIGNMENT</span>
          <h3>One position. Further future targets.</h3>
        </div>
        <label className="inline-select">
          Prediction depth
          <select
            value={depth}
            aria-label="MTP prediction depth"
            onChange={(e) => setDepth(Number(e.target.value))}
          >
            {[1, 2, 3].map((k) => (
              <option key={k} value={k}>
                k = {k}
              </option>
            ))}
          </select>
        </label>
      </div>
      {enough ? (
        <div className="mtp-steps">
          <div>
            <small>BACKBONE POSITION · i = 0</small>
            <strong>{encoded.tokens[0].label}</strong>
            <span>Hidden state H⁰</span>
          </div>
          <ArrowRight size={18} />
          <div>
            <small>SHIFTED INPUT · i + {depth}</small>
            <strong>{encoded.tokens[depth].label}</strong>
            <span>Teacher-forced embedding</span>
          </div>
          <ArrowRight size={18} />
          <div className="mtp-target">
            <small>TARGET · i + {depth + 1}</small>
            <strong>{encoded.tokens[depth + 1].label}</strong>
            <span>Auxiliary cross-entropy target</span>
          </div>
        </div>
      ) : (
        <p>Add at least {depth + 2} tokens to the prompt to inspect this alignment.</p>
      )}
      <div className="lab-footnote">
        <ArrowDown size={12} />
        At depth k, the auxiliary target is k + 1 positions after i. Ordinary next-token prediction
        targets i + 1.
      </div>
    </div>
  )
}
