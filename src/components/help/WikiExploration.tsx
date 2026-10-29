import { useEffect, useId, useState } from 'react'
import { RotateCcw } from 'lucide-react'
import { formatBytes, type ModelConfig } from '../../model/schema'
import { cachePolicyLabel, kvHeadFor, residentPositionAt } from '../../model/attention'
import {
  attentionExample,
  cacheExample,
  explorationFor,
  gateExample,
  normalizationExample,
  probabilityExample,
  rotationExample,
  attentionKeys,
  attentionValues,
  projectionExample,
  residualExample,
  type Pair,
} from '../../help/wiki/explorations'
import type { WikiContext } from '../../help/wiki/catalog'
import { AttentionMemoryComparison } from './AttentionMemoryComparison'
import { WikiLearningLab, NormalizationComparison } from './WikiLearningLab'
import { WikiOptimizerLab } from './WikiOptimizerLab'

const n = (value: number) =>
  Number.isFinite(value)
    ? Number(value.toFixed(5)).toLocaleString('en-GB', { maximumFractionDigits: 5 })
    : '−∞'
const vector = (values: number[]) => `[${values.map(n).join(', ')}]`

function Slider({
  label,
  value,
  onChange,
  min,
  max,
  step = 1,
}: {
  label: string
  value: number
  onChange: (n: number) => void
  min: number
  max: number
  step?: number
}) {
  const id = useId()
  const [draft, setDraft] = useState(String(value))
  useEffect(() => setDraft(String(value)), [value])
  return (
    <div className="wiki-scratch-slider">
      <span>
        <label htmlFor={id}>{label}</label>
        <input
          aria-label={`Set ${label}`}
          type="number"
          min={min}
          max={max}
          step={step}
          value={draft}
          onChange={(event) => {
            const text = event.target.value
            setDraft(text)
            const parsed = Number(text)
            if (text.trim() && Number.isFinite(parsed) && event.currentTarget.validity.valid)
              onChange(parsed)
          }}
          onBlur={() => setDraft(String(value))}
        />
      </span>
      <input
        id={id}
        aria-label={label}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </div>
  )
}
function Calculation({ label, value }: { label: string; value: string }) {
  return (
    <div className="wiki-calculation">
      <span>{label}</span>
      <code>{value}</code>
    </div>
  )
}

function AttentionExample() {
  const [x, setX] = useState(1),
    [y, setY] = useState(2),
    [position, setPosition] = useState(1),
    [causal, setCausal] = useState(true)
  const [keys, setKeys] = useState<Pair[]>(() => attentionKeys.map((row) => [...row]))
  const [values, setValues] = useState<Pair[]>(() => attentionValues.map((row) => [...row]))
  const [selected, setSelected] = useState(0)
  const [stage, setStage] = useState(0)
  const [baseline, setBaseline] = useState<ReturnType<typeof attentionExample>>()
  const result = attentionExample([x, y], position, causal, keys, values)
  const stages = [
    {
      title: 'Representations',
      text: 'Queries and keys determine comparisons. Values supply the feature content that will be mixed. Changing a value alone leaves the probability row unchanged.',
    },
    {
      title: 'Compare and scale',
      text: 'Multiply matching query/key features and add their contributions. Divide each dot product by the square root of the head width before normalization.',
    },
    {
      title: 'Apply visibility',
      text: 'The causal mask excludes keys whose absolute positions are in the query’s future. A large raw comparison cannot override an exclusion.',
    },
    {
      title: 'Normalize the row',
      text: 'Subtract the visible row maximum, exponentiate, and divide by the sum. All visible positions compete in the same denominator.',
    },
    {
      title: 'Combine values',
      text: 'Multiply each value vector by its probability and sum over positions. The result is a feature vector, not a selected token.',
    },
  ]
  function editRow(plane: 'key' | 'value', feature: number, value: number) {
    const update = (rows: Pair[]) =>
      rows.map((row, index) =>
        index === selected ? (row.map((v, d) => (d === feature ? value : v)) as Pair) : row,
      )
    if (plane === 'key') setKeys(update)
    else setValues(update)
  }
  return (
    <>
      <h4>Build one attention row</h4>
      <p>
        Change the query or its position. Each score contracts two features; the mask acts before
        softmax, and probabilities weight the value vectors.
      </p>
      <div className="wiki-scenario-presets" aria-label="Attention scenarios">
        <button
          onClick={() => {
            setX(0)
            setY(0)
            setPosition(2)
            setCausal(true)
          }}
        >
          Equal comparisons
        </button>
        <button
          onClick={() => {
            setX(1)
            setY(2)
            setPosition(0)
            setCausal(true)
          }}
        >
          Only the current position
        </button>
        <button
          onClick={() => {
            setX(1)
            setY(2)
            setPosition(1)
            setCausal(false)
          }}
        >
          Allow a future key
        </button>
      </div>
      <div className="wiki-scratch-controls">
        <Slider
          label="Example query feature 0"
          value={x}
          onChange={setX}
          min={-3}
          max={3}
          step={0.25}
        />
        <Slider
          label="Example query feature 1"
          value={y}
          onChange={setY}
          min={-3}
          max={3}
          step={0.25}
        />
        <Slider
          label="Example query position"
          value={position}
          onChange={setPosition}
          min={0}
          max={2}
        />
      </div>
      <details className="wiki-edit-operands">
        <summary>Edit a key or value vector</summary>
        <label>
          Position to edit
          <select
            aria-label="Example key/value position"
            value={selected}
            onChange={(e) => setSelected(Number(e.target.value))}
          >
            {keys.map((_, i) => (
              <option key={i} value={i}>
                Position {i}
              </option>
            ))}
          </select>
        </label>
        <div className="wiki-scratch-controls">
          <Slider
            label="Example key feature 0"
            value={keys[selected][0]}
            onChange={(v) => editRow('key', 0, v)}
            min={-4}
            max={4}
            step={0.25}
          />
          <Slider
            label="Example key feature 1"
            value={keys[selected][1]}
            onChange={(v) => editRow('key', 1, v)}
            min={-4}
            max={4}
            step={0.25}
          />
          <Slider
            label="Example value feature 0"
            value={values[selected][0]}
            onChange={(v) => editRow('value', 0, v)}
            min={-4}
            max={4}
            step={0.25}
          />
          <Slider
            label="Example value feature 1"
            value={values[selected][1]}
            onChange={(v) => editRow('value', 1, v)}
            min={-4}
            max={4}
            step={0.25}
          />
        </div>
      </details>
      <div className="wiki-calculation-walk">
        <span className="eyebrow">FOLLOW THE MECHANISM</span>
        <div role="group" aria-label="Attention calculation stages">
          {stages.map((item, i) => (
            <button key={item.title} aria-pressed={stage === i} onClick={() => setStage(i)}>
              {i + 1} · {item.title}
            </button>
          ))}
        </div>
        <p>{stages[stage].text}</p>
      </div>
      <label className="wiki-scratch-toggle">
        <input type="checkbox" checked={causal} onChange={(e) => setCausal(e.target.checked)} />
        Apply the causal mask
      </label>
      {!causal && (
        <p className="wiki-scratch-caveat">
          Noncausal comparison: future keys are allowed in this scratch example. The underlying
          decoder remains causal.
        </p>
      )}
      <div className="wiki-table-scroll">
        <table className="wiki-attention-calculation">
          <caption>Q = {vector([x, y])} · head width Dh = 2</caption>
          <thead>
            <tr>
              <th>Key position</th>
              <th>K</th>
              <th>Q · K</th>
              <th>Score / √2</th>
              <th>Visibility</th>
              <th>Probability</th>
              <th>Weighted V</th>
            </tr>
          </thead>
          <tbody>
            {result.keys.map((key, i) => (
              <tr key={i} data-visible={result.visible[i]}>
                <th>{i}</th>
                <td>{vector(key)}</td>
                <td>{n(result.dots[i])}</td>
                <td>{n(result.scaled[i])}</td>
                <td>{result.visible[i] ? '✓ visible' : '× future'}</td>
                <td>
                  {result.probabilities[i] > 0 && (
                    <span
                      className="wiki-probability-bar"
                      style={{ width: `${result.probabilities[i] * 100}%` }}
                    />
                  )}
                  <span>{n(result.probabilities[i] * 100)}%</span>
                </td>
                <td>{vector(result.contributions[i])}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="wiki-calculation-grid">
        <Calculation
          label="Values at positions 0, 1, 2"
          value={result.values.map(vector).join(' · ')}
        />
        <Calculation label="Output: sum the weighted value vectors" value={vector(result.output)} />
      </div>
      <div className="wiki-pin-comparison">
        <button className="button" onClick={() => setBaseline(result)}>
          {baseline ? 'Replace pinned result' : 'Pin current result'}
        </button>
        {baseline && (
          <>
            <button className="button" onClick={() => setBaseline(undefined)}>
              Clear pinned result
            </button>
            <Calculation
              label="Probability changes since pin"
              value={vector(result.probabilities.map((v, i) => v - baseline.probabilities[i]))}
            />
            <Calculation
              label="Output changes since pin"
              value={vector(result.output.map((v, i) => v - baseline.output[i]))}
            />
            <p>
              Only this labelled synthetic row is compared. Pin it, then change a value while
              holding queries and keys fixed to separate attention weights from their contribution.
            </p>
          </>
        )}
      </div>
      <p className="wiki-scratch-insight">
        {result.visible.filter(Boolean).length} keys are visible. Excluded cells contribute exactly
        zero; including a future key changes the denominator for the entire row. The output has two
        features, not one token ID.
      </p>
    </>
  )
}

function RotationExample() {
  const [position, setPosition] = useState(1),
    [pair, setPair] = useState(0)
  const input: [number, number] = [1, 0.5],
    result = rotationExample(input, position, pair)
  const scale = 55,
    end = (values: number[]) => ({ x: 85 + values[0] * scale, y: 85 - values[1] * scale })
  const before = end(input),
    after = end(result.output)
  return (
    <>
      <h4>Rotate a pair; preserve its length</h4>
      <p>
        Adjacent features {2 * pair} and {2 * pair + 1} in a width-4 head use frequency
        10000^(−2j/4). Position changes the angle; pair index changes the frequency.
      </p>
      <div className="wiki-scratch-controls">
        <Slider
          label="Example absolute position"
          value={position}
          onChange={setPosition}
          min={0}
          max={16}
        />
        <Slider label="Example RoPE pair index" value={pair} onChange={setPair} min={0} max={1} />
      </div>
      <div className="wiki-rotation-example">
        <svg
          viewBox="0 0 170 170"
          role="img"
          aria-label={`Input vector ${vector(input)}, rotated vector ${vector(result.output)}; both lengths ${n(result.norm)}`}
        >
          <circle cx="85" cy="85" r={result.norm * scale} className="wiki-rotation-circle" />
          <path d="M10 85H160M85 10V160" className="wiki-rotation-axes" />
          <line x1="85" y1="85" x2={before.x} y2={before.y} className="wiki-vector-before" />
          <line x1="85" y1="85" x2={after.x} y2={after.y} className="wiki-vector-after" />
          <circle cx={before.x} cy={before.y} r="4" className="wiki-vector-before-dot" />
          <rect
            x={after.x - 4}
            y={after.y - 4}
            width="8"
            height="8"
            className="wiki-vector-after-dot"
          />
        </svg>
        <div>
          <Calculation label="○ input pair" value={vector(input)} />
          <Calculation label="□ rotated pair" value={vector(result.output)} />
          <Calculation
            label="Angle = position × frequency"
            value={`${position} × ${n(result.frequency)} = ${n(result.angle)} radians`}
          />
          <Calculation
            label="Pair length before / after"
            value={`${n(result.norm)} / ${n(Math.hypot(...result.output))}`}
          />
        </div>
      </div>
      <Calculation
        label="First output coordinate"
        value={`1 × cos(${n(result.angle)}) − 0.5 × sin(${n(result.angle)}) = ${n(result.output[0])}`}
      />
      <Calculation
        label="Second output coordinate"
        value={`1 × sin(${n(result.angle)}) + 0.5 × cos(${n(result.angle)}) = ${n(result.output[1])}`}
      />
      <p className="wiki-scratch-insight">
        At position 0 the rotation is the identity. The numerical profile rotates Q and K, using
        absolute position IDs; changing a cache slot never resets the position.
      </p>
    </>
  )
}

function NormalizationExample() {
  const [x, setX] = useState(3),
    [y, setY] = useState(4),
    [gain, setGain] = useState(1)
  const result = normalizationExample([x, y], gain)
  return (
    <>
      <h4>Normalize one feature row</h4>
      <p>
        The two coordinates share a denominator. Change either input and watch both outputs; the
        scratch gain is applied to both features for clarity.
      </p>
      <div className="wiki-scratch-controls">
        <Slider
          label="Example RMS input 0"
          value={x}
          onChange={setX}
          min={-4}
          max={4}
          step={0.25}
        />
        <Slider
          label="Example RMS input 1"
          value={y}
          onChange={setY}
          min={-4}
          max={4}
          step={0.25}
        />
        <Slider
          label="Example RMS gain"
          value={gain}
          onChange={setGain}
          min={0.5}
          max={2}
          step={0.1}
        />
      </div>
      <div className="wiki-calculation-grid">
        <Calculation label="1 · Square each coordinate" value={vector(result.squares)} />
        <Calculation
          label="2 · Average the squares"
          value={`(${n(result.squares[0])} + ${n(result.squares[1])}) / 2 = ${n(result.meanSquare)}`}
        />
        <Calculation
          label="3 · Add epsilon and take the root"
          value={`√(${n(result.meanSquare)} + 0.00001) = ${n(result.denominator)}`}
        />
        <Calculation label="4 · Divide and multiply by gain" value={vector(result.output)} />
      </div>
      <p className="wiki-scratch-insight">
        Output mean = {n((result.output[0] + result.output[1]) / 2)}. RMSNorm does not subtract the
        mean or produce probabilities. Epsilon keeps an all-zero input well-defined.
      </p>
    </>
  )
}

function GateExample() {
  const [gate, setGate] = useState(2),
    [up, setUp] = useState(-3)
  const result = gateExample(gate, up)
  return (
    <>
      <h4>A gate is a numerical multiplication</h4>
      <p>
        Explore one intermediate coordinate before the down projection. SiLU is the gate
        preactivation multiplied by its sigmoid; it is not a probability restricted to 0–1.
      </p>
      <div className="wiki-scratch-controls">
        <Slider
          label="Example gate preactivation"
          value={gate}
          onChange={setGate}
          min={-5}
          max={5}
          step={0.25}
        />
        <Slider
          label="Example up-branch feature"
          value={up}
          onChange={setUp}
          min={-4}
          max={4}
          step={0.25}
        />
      </div>
      <div className="wiki-calculation-grid">
        <Calculation label="Sigmoid(gate)" value={n(result.sigmoid)} />
        <Calculation label="SiLU(gate) = gate × sigmoid(gate)" value={n(result.silu)} />
        <Calculation label="Matching up coordinate" value={n(up)} />
        <Calculation
          label="Element-wise product"
          value={`${n(result.silu)} × ${n(up)} = ${n(result.product)}`}
        />
      </div>
      <p className="wiki-scratch-insight">
        There is no sum at this multiplication. The later down projection sums across intermediate
        features. A negative gate input can give a negative SiLU value.
      </p>
    </>
  )
}

function ProbabilityExample() {
  const [temperature, setTemperature] = useState(1)
  const scores = [0, 1, 2],
    result = probabilityExample(scores, temperature)
  return (
    <>
      <h4>Change temperature; watch probability move</h4>
      <p>
        These are three fictional vocabulary candidates with logits [0, 1, 2]. This tiny
        distribution is unrelated to either tokenizer or checkpoint.
      </p>
      <Slider
        label="Example vocabulary temperature"
        value={temperature}
        onChange={setTemperature}
        min={0.1}
        max={3}
        step={0.1}
      />
      <div className="wiki-calculation-grid">
        {result.probabilities.map((p, i) => (
          <Calculation
            key={i}
            label={`Candidate ${i} · logit ${scores[i]}`}
            value={`${n(p * 100)}%`}
          />
        ))}
        <Calculation
          label="Stable normalization"
          value={`row max ${n(result.maximum)} · exponential sum ${n(result.denominator)}`}
        />
      </div>
      <p className="wiki-scratch-insight">
        Candidate 2 remains the greedy maximum for every positive temperature here. Sampling
        probabilities change, but the logits and model parameters do not.
      </p>
    </>
  )
}

function CacheExample({
  config,
  symbols,
}: {
  config: ModelConfig
  symbols?: Record<string, number>
}) {
  const capacity = config.contextLength
  const [extent, setExtent] = useState(Math.min(capacity, Math.max(1, symbols?.S ?? 128)))
  const [queries, setQueries] = useState(Math.min(extent, Math.max(1, symbols?.T ?? 1)))
  const [policy, setPolicy] = useState<NonNullable<ModelConfig['kvCache']>>(
    () => config.kvCache ?? { mode: 'full' },
  )
  const [bytes, setBytes] = useState(config.dtypeBytes)
  const scratch = { ...config, kvCache: policy, dtypeBytes: bytes }
  const result = cacheExample(scratch, extent, Math.min(queries, extent))
  const count = result.retained
  const indices =
    count > 24
      ? [
          ...Array.from({ length: 8 }, (_, i) => i),
          ...Array.from({ length: 16 }, (_, i) => count - 16 + i),
        ]
      : Array.from({ length: count }, (_, i) => i)
  const shown = indices.map((i) => residentPositionAt(scratch, extent, i))
  return (
    <>
      <h4>Account for this configuration’s K/V</h4>
      <p>
        Starts from the active study: {config.nLayers} layers, Hkv = {config.nKvHeads}, Dh ={' '}
        {config.headDim}, {config.dtypeBytes} bytes per scalar, {cachePolicyLabel(config)}. Changing
        these scratch controls leaves the study unchanged.
      </p>
      <div className="wiki-scratch-selects">
        <label>
          Scratch retention
          <select
            aria-label="Scratch cache retention"
            value={policy.mode}
            onChange={(e) =>
              setPolicy(
                e.target.value === 'full'
                  ? { mode: 'full' }
                  : e.target.value === 'sliding-window'
                    ? { mode: 'sliding-window', windowSize: Math.min(capacity, 4) }
                    : {
                        mode: 'sparse',
                        windowSize: Math.min(capacity, 4),
                        stride: Math.min(capacity, 4),
                      },
              )
            }
          >
            <option value="full">Full prefix</option>
            <option value="sliding-window">Sliding window</option>
            <option value="sparse">Sparse anchors + recent</option>
          </select>
        </label>
        <label>
          Scratch bytes per scalar
          <select
            aria-label="Scratch scalar bytes"
            value={bytes}
            onChange={(e) => setBytes(Number(e.target.value) as 1 | 2 | 4)}
          >
            {[1, 2, 4].map((v) => (
              <option key={v} value={v}>
                {v} byte{v === 1 ? '' : 's'}
              </option>
            ))}
          </select>
        </label>
      </div>
      <div className="wiki-scratch-controls">
        <Slider
          label="Scratch processed extent S"
          value={extent}
          onChange={setExtent}
          min={1}
          max={capacity}
        />
        <Slider
          label="Scratch query count T"
          value={Math.min(queries, extent)}
          onChange={setQueries}
          min={1}
          max={extent}
        />
        {policy.mode !== 'full' && (
          <Slider
            label="Scratch retention window"
            value={policy.windowSize}
            onChange={(windowSize) => setPolicy({ ...policy, windowSize })}
            min={1}
            max={capacity}
          />
        )}
        {policy.mode === 'sparse' && (
          <Slider
            label="Scratch anchor stride"
            value={policy.stride}
            onChange={(stride) => setPolicy({ ...policy, stride })}
            min={2}
            max={capacity}
          />
        )}
      </div>
      <div className="wiki-calculation-grid">
        <Calculation label="Before this call · retained prefix P" value={String(result.prefix)} />
        <Calculation
          label="During attention · working U = P + T"
          value={`${result.prefix} + ${result.queries} = ${result.working}`}
        />
        <Calculation label="After this call · retained R" value={String(result.retained)} />
        <Calculation label="Logical persistent K/V payload" value={formatBytes(result.bytes)} />
      </div>
      <div className="wiki-position-strip" aria-label="Scratch retained absolute positions">
        {shown.map((p, i) => (
          <span key={p}>
            {count > 24 && i === 8 ? '… ' : ''}
            {p}
          </span>
        ))}
      </div>
      <Calculation
        label="2 K/V planes × layers × batch × KV heads × R × features × bytes"
        value={`2 × ${config.nLayers} × 1 × ${config.nKvHeads} × ${result.retained} × ${config.headDim} × ${bytes} = ${result.bytes.toLocaleString()} bytes`}
      />
      <p className="wiki-scratch-insight">
        At the same head width and retention: MHA-sized state = {formatBytes(result.mhaBytes)};
        MQA-sized state = {formatBytes(result.mqaBytes)}. These are payload comparisons, not
        trained-weight conversions. Trace history, position metadata and allocator overhead are
        excluded. Scratch policy: {cachePolicyLabel(scratch)}. Storage-size controls estimate
        payload only; they do not execute quantization.
      </p>
    </>
  )
}

function ProjectionExample() {
  const [x, setX] = useState(2),
    [y, setY] = useState(-1)
  const [weights, setWeights] = useState<[Pair, Pair]>([
    [3, 4],
    [5, -2],
  ])
  const [row, setRow] = useState(0),
    [column, setColumn] = useState(0)
  const result = projectionExample([x, y], weights)
  function editWeight(value: number) {
    setWeights(
      (old) =>
        old.map((r, i) =>
          i === row ? (r.map((v, j) => (j === column ? value : v)) as Pair) : r,
        ) as [Pair, Pair],
    )
  }
  return (
    <>
      <h4>Build an output feature from its contributions</h4>
      <p>
        This synthetic row-vector projection mixes two input features into two output features.
        Select a matrix cell to edit that weight, then inspect each output’s separate products and
        sum.
      </p>
      <div className="wiki-scratch-controls">
        <Slider
          label="Projection input feature 0"
          value={x}
          onChange={setX}
          min={-4}
          max={4}
          step={0.25}
        />
        <Slider
          label="Projection input feature 1"
          value={y}
          onChange={setY}
          min={-4}
          max={4}
          step={0.25}
        />
        <Slider
          label="Selected projection weight"
          value={weights[row][column]}
          onChange={editWeight}
          min={-6}
          max={6}
          step={0.25}
        />
      </div>
      <div className="wiki-toy-matrix" aria-label="Synthetic projection matrix">
        {weights.map((values, i) => (
          <div key={i}>
            {values.map((weight, j) => (
              <button
                key={j}
                aria-label={`Edit weight row ${i}, column ${j}`}
                aria-pressed={i === row && j === column}
                onClick={() => {
                  setRow(i)
                  setColumn(j)
                }}
              >
                <small>
                  input {i} → output {j}
                </small>
                {n(weight)}
              </button>
            ))}
          </div>
        ))}
      </div>
      <div className="wiki-calculation-grid">
        {result.products.map((products, j) => (
          <Calculation
            key={j}
            label={`Output feature ${j} · sum over input features`}
            value={`${n(x)} × ${n(weights[0][j])} + ${n(y)} × ${n(weights[1][j])} = ${n(products[0])} + ${n(products[1])} = ${n(result.output[j])}`}
          />
        ))}
      </div>
      <p className="wiki-scratch-insight">
        The input-feature axis is contracted; the output-feature axis remains. Editing one matrix
        cell affects its output column directly. This changes toy weights, not checkpoint
        parameters.
      </p>
    </>
  )
}

function ResidualExample() {
  const [input, setInput] = useState(2),
    [attention, setAttention] = useState(-0.5),
    [feedForward, setFeedForward] = useState(0.2),
    [skip, setSkip] = useState(true)
  const result = residualExample(input, attention, feedForward, skip)
  return (
    <>
      <h4>Keep the stream and add an update</h4>
      <p>
        Follow one synthetic coordinate through two additions. The branch updates are independent
        knobs here: this illustrates addition, not the full attention or feed-forward calculation.
      </p>
      <div className="wiki-scratch-controls">
        <Slider
          label="Residual input coordinate"
          value={input}
          onChange={setInput}
          min={-4}
          max={4}
          step={0.1}
        />
        <Slider
          label="Attention update coordinate"
          value={attention}
          onChange={setAttention}
          min={-3}
          max={3}
          step={0.1}
        />
        <Slider
          label="Feed-forward update coordinate"
          value={feedForward}
          onChange={setFeedForward}
          min={-3}
          max={3}
          step={0.1}
        />
      </div>
      <label className="wiki-scratch-toggle">
        <input type="checkbox" checked={skip} onChange={(e) => setSkip(e.target.checked)} />
        Include the identity paths
      </label>
      <div className="wiki-residual-path">
        <span>
          Incoming stream<strong>{n(input)}</strong>
        </span>
        <b>{skip ? '+ attention update' : 'replace with attention update'}</b>
        <span>
          After attention<strong>{n(result.afterAttention)}</strong>
        </span>
        <b>{skip ? '+ feed-forward update' : 'replace with feed-forward update'}</b>
        <span>
          Block output<strong>{n(result.output)}</strong>
        </span>
      </div>
      <p className="wiki-scratch-insight">
        {skip
          ? 'Setting both updates to zero preserves the incoming coordinate.'
          : 'With both identity paths removed, this simplified example no longer carries the original input to the output.'}{' '}
        This says nothing by itself about trained-model quality or recoverability of an arbitrary
        residual block.
      </p>
    </>
  )
}

function EmbeddingExample() {
  const [id, setId] = useState(1),
    [feature, setFeature] = useState(0)
  const [rows, setRows] = useState(() => [
    [0.2, -0.4, 0.7],
    [1, 0.5, -0.2],
    [-0.8, 0.3, 0.1],
    [0.4, 0.9, -0.6],
  ])
  function edit(value: number) {
    setRows((old) =>
      old.map((row, i) => (i === id ? row.map((v, j) => (j === feature ? value : v)) : row)),
    )
  }
  return (
    <>
      <h4>An ID chooses a row, not a magnitude</h4>
      <p>
        These four fictional IDs and three-feature rows belong only to this toy lookup. Select an
        ID, then choose a feature to edit its stored embedding.
      </p>
      <div className="wiki-scratch-controls">
        <Slider label="Toy vocabulary ID" value={id} onChange={setId} min={0} max={3} />
        <Slider
          label="Toy embedding feature"
          value={feature}
          onChange={setFeature}
          min={0}
          max={2}
        />
        <Slider
          label="Selected embedding value"
          value={rows[id][feature]}
          onChange={edit}
          min={-2}
          max={2}
          step={0.1}
        />
      </div>
      <div className="wiki-embedding-rows">
        {rows.map((row, i) => (
          <button
            key={i}
            aria-pressed={id === i}
            aria-label={`Gather fictional embedding ID ${i}`}
            onClick={() => setId(i)}
          >
            <span>ID {i}</span>
            <code>{vector(row)}</code>
          </button>
        ))}
      </div>
      <Calculation label="Gathered hidden row" value={vector(rows[id])} />
      <p className="wiki-scratch-insight">
        Choosing the same ID again retrieves the same parameter row. Later contextual differences
        arise from subsequent model operations, which this lookup example does not fabricate.
      </p>
    </>
  )
}

function HeadExample({ config }: { config: ModelConfig }) {
  const [head, setHead] = useState(0)
  const kv = kvHeadFor(config, head)
  return (
    <>
      <h4>Follow a query to its shared state</h4>
      <p>
        The active profile has {config.nQueryHeads} query heads and {config.nKvHeads} K/V heads.
        Each contiguous group contains {config.nQueryHeads / config.nKvHeads} queries.
      </p>
      <Slider
        label="Scratch query head"
        value={head}
        onChange={setHead}
        min={0}
        max={config.nQueryHeads - 1}
      />
      <div className="wiki-head-map">
        {Array.from({ length: config.nQueryHeads }, (_, h) => (
          <button
            key={h}
            aria-label={`Inspect query head ${h} mapping`}
            aria-pressed={h === head}
            onClick={() => setHead(h)}
          >
            Q{h}
            <span>
              → K{kvHeadFor(config, h)} / V{kvHeadFor(config, h)}
            </span>
          </button>
        ))}
      </div>
      <Calculation
        label="Contiguous group assignment"
        value={`floor(${head} / (${config.nQueryHeads} / ${config.nKvHeads})) = ${kv}`}
      />
      <p className="wiki-scratch-insight">
        Q{head} reads K{kv} and V{kv}, but keeps its own scores and probability row. Selecting a
        logical head does not allocate another cache.
      </p>
    </>
  )
}

export function WikiExploration({
  articleId,
  context,
  onRead,
}: {
  articleId: string
  context: WikiContext
  onRead: (id: string) => void
}) {
  const kind = explorationFor[articleId]
  const [revision, setRevision] = useState(0)
  if (!kind) return null
  const config = context.config ?? context.doc.config
  return (
    <section className="wiki-exploration" aria-label="Interactive worked example">
      <header>
        <span className="eyebrow">
          {kind === 'cache' || kind === 'heads'
            ? 'SCRATCH CALCULATION · ACTIVE PROFILE'
            : 'INTERACTIVE WORKED EXAMPLE · SYNTHETIC'}
        </span>
        <button aria-label="Reset worked example" onClick={() => setRevision(revision + 1)}>
          <RotateCcw size={13} />
          Reset
        </button>
      </header>
      <div key={`${articleId}-${revision}`}>
        {kind === 'adam-lab' || kind === 'muon-lab' ? (
          <WikiOptimizerLab kind={kind} />
        ) : kind === 'memory-comparison' ? (
          <AttentionMemoryComparison onRead={onRead} />
        ) : kind === 'attention' ? (
          <AttentionExample />
        ) : kind === 'rotation' ? (
          <RotationExample />
        ) : kind === 'normalization' ? (
          <>
            <NormalizationExample />
            <NormalizationComparison />
          </>
        ) : kind === 'gate' ? (
          <GateExample />
        ) : kind === 'probabilities' ? (
          <ProbabilityExample />
        ) : kind === 'cache' ? (
          <CacheExample config={config} symbols={context.symbols} />
        ) : kind === 'projection' ? (
          <ProjectionExample />
        ) : kind === 'residual' ? (
          <ResidualExample />
        ) : kind === 'embedding' ? (
          <EmbeddingExample />
        ) : kind === 'heads' ? (
          <HeadExample config={config} />
        ) : (
          <WikiLearningLab kind={kind} />
        )}
      </div>
      <footer>
        Independent teaching arithmetic. Does not edit the model or display checkpoint activations.
        The article’s primary references explain the underlying mechanism.
      </footer>
    </section>
  )
}
