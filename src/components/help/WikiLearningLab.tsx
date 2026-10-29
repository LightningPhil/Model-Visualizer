import { useEffect, useId, useState } from 'react'
import {
  regressionStep,
  positionComparison,
  featureMixture,
  interventionExample,
  preferenceExample,
  precisionExample,
  compareNormalizations,
  routedExample,
  type Intervention,
} from '../../help/wiki/learning-experiments'

const number = (v: number) =>
  Number.isFinite(v)
    ? Number(v.toFixed(6)).toLocaleString('en-GB', { maximumFractionDigits: 6 })
    : String(v)
const vector = (v: number[]) => `[${v.map(number).join(', ')}]`
export function Control({
  label,
  value,
  change,
  min,
  max,
  step = 1,
}: {
  label: string
  value: number
  change: (n: number) => void
  min: number
  max: number
  step?: number
}) {
  const id = useId(),
    [draft, setDraft] = useState(String(value))
  useEffect(() => setDraft(String(value)), [value])
  return (
    <div className="wiki-scratch-slider">
      <span>
        <label htmlFor={id}>{label}</label>
        <input
          aria-label={`Set ${label}`}
          type="number"
          value={draft}
          min={min}
          max={max}
          step={step}
          onChange={(e) => {
            setDraft(e.currentTarget.value)
            if (e.currentTarget.value && e.currentTarget.validity.valid)
              change(Number(e.currentTarget.value))
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
        onChange={(e) => change(Number(e.target.value))}
      />
    </div>
  )
}
export function Result({ label, value }: { label: string; value: string }) {
  return (
    <div className="wiki-calculation">
      <span>{label}</span>
      <code>{value}</code>
    </div>
  )
}

function LearningLoop() {
  const [x, setX] = useState(2),
    [target, setTarget] = useState(3),
    [weight, setWeight] = useState(0),
    [bias, setBias] = useState(0),
    [rate, setRate] = useState(0.1),
    [count, setCount] = useState(0)
  useEffect(() => setCount(0), [x, target, weight, bias, rate])
  const history = []
  let w = weight,
    b = bias
  for (let i = 0; i < count; i++) {
    const step = regressionStep(x, target, w, b, rate)
    history.push(step)
    w = step.nextWeight
    b = step.nextBias
  }
  const current = regressionStep(x, target, w, b, rate)
  return (
    <>
      <h4>Follow one example through an SGD update</h4>
      <p>
        A synthetic scalar predictor, not transformer training. The controls choose an initial
        state; each step updates both parameters from the same old gradient. Changing a control
        restarts the scratch trajectory.
      </p>
      <div className="wiki-scratch-controls">
        <Control label="Training input x" value={x} change={setX} min={-4} max={4} step={0.25} />
        <Control
          label="Training target y"
          value={target}
          change={setTarget}
          min={-4}
          max={4}
          step={0.25}
        />
        <Control
          label="Initial training weight"
          value={weight}
          change={setWeight}
          min={-2}
          max={2}
          step={0.1}
        />
        <Control
          label="Initial training bias"
          value={bias}
          change={setBias}
          min={-2}
          max={2}
          step={0.1}
        />
        <Control
          label="SGD learning rate"
          value={rate}
          change={setRate}
          min={0}
          max={2}
          step={0.05}
        />
      </div>
      <div className="wiki-calculation-grid">
        <Result
          label={`Current parameters · after ${count} steps`}
          value={`w = ${number(w)}, b = ${number(b)}`}
        />
        <Result
          label="1 · Forward prediction and error"
          value={`ŷ = ${number(current.prediction)}; e = ${number(current.error)}`}
        />
        <Result label="2 · Half squared error" value={number(current.loss)} />
        <Result
          label="3 · Backward gradients"
          value={`dL/dw = ${number(current.gradientWeight)}; dL/db = ${number(current.gradientBias)}`}
        />
        <Result
          label="Independent weight finite difference · h = 0.0001"
          value={number(current.finiteDifference)}
        />
        <Result
          label="4 · Proposed simultaneous update"
          value={`w′ = ${number(current.nextWeight)}; b′ = ${number(current.nextBias)}`}
        />
        <Result label="Loss after proposed step" value={number(current.nextLoss)} />
      </div>
      <div className="wiki-scenario-presets">
        <button disabled={count >= 12} onClick={() => setCount(count + 1)}>
          Take one SGD step
        </button>
        <button onClick={() => setCount(0)}>Restart training trajectory</button>
      </div>
      <p className="wiki-scratch-insight">
        {current.nextLoss > current.loss
          ? 'The proposed step increases this example’s loss: the learning rate overshoots.'
          : current.loss === 0
            ? 'This example is fitted; its gradient is zero.'
            : 'The proposed step reduces this example’s loss. That does not establish generalization.'}{' '}
        Up to twelve scratch updates; no checkpoint parameters change.
      </p>
      {history.length > 0 && (
        <ol className="wiki-lab-history" aria-label="Synthetic training history">
          {history.map((s, i) => (
            <li key={i}>
              Step {i + 1}: loss {number(s.loss)} → {number(s.nextLoss)}
            </li>
          ))}
        </ol>
      )}
    </>
  )
}
function Positions() {
  const [query, setQuery] = useState(1),
    [key, setKey] = useState(0),
    [slope, setSlope] = useState(0.5)
  const r = positionComparison(query, key, slope)
  return (
    <>
      <h4>Move position while holding content fixed</h4>
      <p>
        Two-feature content Q = [1, 0.5], K = [0.5, 1]. Addition rows assume identity Q/K
        projections. The table is fictional; sine/cosine and rotation use one frequency of one
        radian per position.
      </p>
      <div className="wiki-scratch-controls">
        <Control
          label="Comparison query position"
          value={query}
          change={setQuery}
          min={0}
          max={3}
        />
        <Control label="Comparison key position" value={key} change={setKey} min={0} max={3} />
        <Control
          label="ALiBi toy head slope"
          value={slope}
          change={setSlope}
          min={0}
          max={2}
          step={0.1}
        />
      </div>
      <p className="wiki-scratch-insight">
        {r.visible
          ? 'Causally visible: these raw comparisons may enter the row normalization.'
          : 'Future key: ALL methods exclude this key after the raw comparison. No positional method overrides the causal mask.'}
      </p>
      <div className="wiki-calculation-grid">
        <Result label="Content only · scaled dot product" value={number(r.content)} />
        <Result label="Fictional learned absolute addition · raw score" value={number(r.learned)} />
        <Result label="Sinusoidal addition · raw score" value={number(r.sinusoidal)} />
        <Result label="RoPE pair rotation · raw score" value={number(r.rotary)} />
        <Result label="ALiBi distance bias · raw score" value={number(r.alibi)} />
        <Result
          label="Fictional absolute rows selected"
          value={`Q: ${vector(r.queryEncoding)}; K: ${vector(r.keyEncoding)}`}
        />
      </div>
      <p>
        Try positions (1,0), then (2,1). The rotary and distance-bias comparisons preserve this
        pair’s relative displacement; the addition methods need not preserve the score. This does
        not extend the loaded model’s context.
      </p>
    </>
  )
}
function Features() {
  const [a, setA] = useState(1),
    [b, setB] = useState(0.5),
    [c, setC] = useState(0),
    [angle, setAngle] = useState(60),
    [view, setView] = useState(0)
  const r = featureMixture([a, b, c], (angle * Math.PI) / 180, (view * Math.PI) / 180)
  return (
    <>
      <h4>Three fictional features, two coordinates</h4>
      <p>
        Change feature strengths and direction overlap. A dot-product readout for F0 contains its
        own coefficient plus interference. Rotate both the display and readout together to change
        coordinates without changing that scalar result.
      </p>
      <div className="wiki-scratch-controls">
        <Control
          label="Fictional feature F0 strength"
          value={a}
          change={setA}
          min={0}
          max={1}
          step={0.1}
        />
        <Control
          label="Fictional feature F1 strength"
          value={b}
          change={setB}
          min={0}
          max={1}
          step={0.1}
        />
        <Control
          label="Fictional feature F2 strength"
          value={c}
          change={setC}
          min={0}
          max={1}
          step={0.1}
        />
        <Control
          label="F1 dictionary angle"
          value={angle}
          change={setAngle}
          min={0}
          max={180}
          step={15}
        />
        <Control
          label="Feature view rotation"
          value={view}
          change={setView}
          min={0}
          max={180}
          step={15}
        />
      </div>
      <svg
        className="wiki-feature-geometry"
        viewBox="-210 -210 420 420"
        role="img"
        aria-label="Synthetic feature directions and their summed hidden vector"
      >
        <line x1="-190" y1="0" x2="190" y2="0" className="axis" />
        <line x1="0" y1="-190" x2="0" y2="190" className="axis" />
        {r.directions.map((d, i) => (
          <g key={i}>
            <line x1="0" y1="0" x2={d[0] * 75} y2={-d[1] * 75} className={`feature f${i}`} />
            <text x={d[0] * 95} y={-d[1] * 95}>
              F{i}
            </text>
          </g>
        ))}
        <line x1="0" y1="0" x2={r.hidden[0] * 55} y2={-r.hidden[1] * 55} className="hidden" />
        <rect
          x={r.hidden[0] * 55 - 4}
          y={-r.hidden[1] * 55 - 4}
          width="8"
          height="8"
          className="hidden-tip"
        />
      </svg>
      <div className="wiki-calculation-grid">
        <Result label="Displayed hidden coordinates" value={vector(r.hidden)} />
        <Result label="Declared F0 coefficient" value={number(r.targetStrength)} />
        <Result label="F0 dot-product readout" value={number(r.readout)} />
        <Result label="Interference from other directions" value={number(r.interference)} />
      </div>
      <p>
        The labels are authored fictional features, not discovered concepts in a trained checkpoint.
        Dashed directions are dictionary vectors; the solid line with a square endpoint is their
        weighted sum.
      </p>
    </>
  )
}
function Interventions() {
  const [mode, setMode] = useState<Intervention>('clean'),
    [replacement, setReplacement] = useState(0)
  const r = interventionExample(mode, replacement)
  return (
    <>
      <h4>Change a path and recompute its consequence</h4>
      <p>
        Fixed synthetic graph: z = 3a − 2b. Clean hidden [1,1]; altered hidden [0,1]. Patching
        copies the first clean coordinate into the altered state. Ablation here removes the second
        coordinate from clean.
      </p>
      <div
        className="wiki-scenario-presets"
        role="group"
        aria-label="Synthetic intervention choices"
      >
        {(
          [
            ['clean', 'Clean baseline'],
            ['corrupted', 'Altered baseline'],
            ['ablate', 'Ablate negative path'],
            ['patch', 'Patch clean first coordinate'],
            ['replace', 'Replace first coordinate'],
          ] as [Intervention, string][]
        ).map(([value, label]) => (
          <button key={value} aria-pressed={mode === value} onClick={() => setMode(value)}>
            {label}
          </button>
        ))}
      </div>
      {mode === 'replace' && (
        <Control
          label="Replacement first coordinate"
          value={replacement}
          change={setReplacement}
          min={-3}
          max={3}
          step={0.25}
        />
      )}
      <div className="wiki-calculation-grid">
        <Result label="Intervened hidden state" value={vector(r.hidden)} />
        <Result label="Separate path contributions" value={vector(r.contributions)} />
        <Result label="Recomputed output logit" value={number(r.logit)} />
        <Result label="Two-outcome sigmoid probability" value={number(r.probability)} />
        <Result label="Logit change from clean baseline" value={number(r.deltaClean)} />
        <Result label="Logit change from altered baseline" value={number(r.deltaCorrupted)} />
      </div>
      <p>
        {mode === 'ablate'
          ? 'Removing the negative contribution increases the logit. Deletion is not guaranteed to reduce an output.'
          : mode === 'patch'
            ? 'The patch restores this toy’s clean result under this altered background. It is not evidence about a language concept.'
            : 'Read the changed tensor, intervention source and baseline before interpreting the delta.'}{' '}
        No trained-model activation is replaced.
      </p>
    </>
  )
}
function PostTraining() {
  const [gap, setGap] = useState(0),
    [reference, setReference] = useState(0),
    [beta, setBeta] = useState(0.5)
  const r = preferenceExample(gap, reference, beta)
  return (
    <>
      <h4>Demonstrations and preferences use different signals</h4>
      <p>
        This fictional universe has exactly two completions. A scalar policy gap determines their
        probabilities. The preferred completion is also the demonstration target. DPO additionally
        compares the policy gap with a fixed reference gap; no assistant or reward model is trained.
      </p>
      <div className="wiki-scratch-controls">
        <Control
          label="Policy completion log-probability gap"
          value={gap}
          change={setGap}
          min={-6}
          max={6}
          step={0.25}
        />
        <Control
          label="Reference completion log-probability gap"
          value={reference}
          change={setReference}
          min={-6}
          max={6}
          step={0.25}
        />
        <Control
          label="Preference beta"
          value={beta}
          change={setBeta}
          min={0.1}
          max={2}
          step={0.1}
        />
      </div>
      <div className="wiki-calculation-grid">
        <Result
          label="Policy probability of preferred completion"
          value={number(r.preferredProbability)}
        />
        <Result label="Demonstration negative log likelihood" value={number(r.demonstrationLoss)} />
        <Result label="Reference-adjusted DPO margin" value={number(r.margin)} />
        <Result label="DPO pair loss" value={number(r.dpoLoss)} />
        <Result label="DPO derivative with respect to policy gap" value={number(r.gradientGap)} />
      </div>
      <p>
        Change only the reference gap: the policy probability and demonstration loss stay fixed,
        while the preference objective changes. Real response probabilities are products of token
        probabilities, computed as sums in log space. Preference labels do not certify truth.
      </p>
    </>
  )
}
function Precision() {
  const [magnitude, setMagnitude] = useState(100000000),
    [logit, setLogit] = useState(1000)
  const r = precisionExample(magnitude, logit)
  return (
    <>
      <h4>Watch a small contribution disappear</h4>
      <label className="wiki-learning-select">
        Cancellation magnitude A
        <select
          aria-label="Cancellation magnitude"
          value={magnitude}
          onChange={(e) => setMagnitude(Number(e.target.value))}
        >
          {[1, 10000, 100000000, 1000000000000].map((v) => (
            <option key={v} value={v}>
              {v.toLocaleString()}
            </option>
          ))}
        </select>
      </label>
      <p>
        Each operand and addition is rounded to binary32 with Math.fround. The target real-number
        sum of A + 1 − A is 1; the JavaScript binary64 row is a separate higher-precision
        calculation.
      </p>
      <div className="wiki-calculation-grid">
        <Result label="A after binary32 storage rounding" value={number(r.stored)} />
        <Result label="Binary32 · add the unit first: (A + 1) − A" value={number(r.sequential32)} />
        <Result label="Binary32 · cancel first: (A − A) + 1" value={number(r.cancelFirst32)} />
        <Result label="JavaScript binary64 · (A + 1) − A" value={number(r.binary64)} />
      </div>
      <h4>Stable probabilities despite large logits</h4>
      <Control
        label="Large softmax logit"
        value={logit}
        change={setLogit}
        min={0}
        max={1000}
        step={10}
      />
      <div className="wiki-calculation-grid">
        <Result label="Naive exponentials for [z, z − 1]" value={vector(r.exponentials)} />
        <Result label="Naive normalized probabilities" value={vector(r.naive)} />
        <Result label="Max-shifted stable probabilities" value={vector(r.stable)} />
      </div>
      <p>
        Infinity and NaN are reported explicitly. The stable pair stays the same under a common
        finite shift. This is not FP16/BF16 emulation or a GPU benchmark.
      </p>
    </>
  )
}
export function NormalizationComparison() {
  const [offset, setOffset] = useState(0),
    [constant, setConstant] = useState(false)
  const r = compareNormalizations(constant ? [3, 3] : [3, 4], offset)
  return (
    <div className="wiki-learning-extension">
      <h4>Compare LayerNorm and RMSNorm on the same row</h4>
      <p>
        Shared gain 1, zero additive bias and epsilon 0.00001. Centered variance uses the population
        denominator. Changing the common offset exposes the difference between centering and
        root-mean-square scaling.
      </p>
      <Control
        label="Normalization common offset"
        value={offset}
        change={setOffset}
        min={-5}
        max={5}
        step={0.5}
      />
      <label className="wiki-scratch-toggle">
        <input type="checkbox" checked={constant} onChange={(e) => setConstant(e.target.checked)} />
        Use a constant feature row
      </label>
      <div className="wiki-calculation-grid">
        <Result label="Shared input row" value={vector(r.row)} />
        <Result
          label="Centered mean and variance"
          value={`mean = ${number(r.mean)}; variance = ${number(r.variance)}`}
        />
        <Result label="LayerNorm comparison output" value={vector(r.layer)} />
        <Result label="RMSNorm comparison output" value={vector(r.rootMeanSquare)} />
      </div>
    </div>
  )
}
function Experts() {
  const [input, setInput] = useState(2),
    [first, setFirst] = useState(1),
    [second, setSecond] = useState(0),
    [third, setThird] = useState(-1),
    [count, setCount] = useState(1)
  const r = routedExample(input, [first, second, third], count)
  return (
    <>
      <h4>Route to experts; distinguish stored from activated parameters</h4>
      <p>
        Three fictional scalar experts: E0(x)=x, E1(x)=−x+1, E2(x)=2x−1. Top-k scores select
        branches; weights renormalize over those branches. Equal scores choose the lower ID first.
        No shared experts, capacity limit or communication model is simulated.
      </p>
      <div className="wiki-scratch-controls">
        <Control
          label="Toy expert input"
          value={input}
          change={setInput}
          min={-3}
          max={3}
          step={0.25}
        />
        <Control label="Selected expert count" value={count} change={setCount} min={1} max={3} />
        <Control
          label="Expert zero router score"
          value={first}
          change={setFirst}
          min={-3}
          max={3}
          step={0.25}
        />
        <Control
          label="Expert one router score"
          value={second}
          change={setSecond}
          min={-3}
          max={3}
          step={0.25}
        />
        <Control
          label="Expert two router score"
          value={third}
          change={setThird}
          min={-3}
          max={3}
          step={0.25}
        />
      </div>
      <div className="wiki-expert-branches">
        {r.weights.map((weight, i) => (
          <div key={i} data-selected={r.selected.includes(i)}>
            <strong>
              E{i} · {r.selected.includes(i) ? 'selected' : 'inactive'}
            </strong>
            <span>Combine weight {number(weight)}</span>
            <code>
              {r.selected.includes(i)
                ? `output ${number(r.outputs[i]!)} → contribution ${number(r.contributions[i])}`
                : 'Branch not executed in the routed calculation'}
            </code>
          </div>
        ))}
      </div>
      <div className="wiki-calculation-grid">
        <Result label="Routed weighted output" value={number(r.output)} />
        <Result label="Dense reference · one 2x − 1 transformation" value={number(r.denseOutput)} />
        <Result
          label="Expert-only parameters · excludes router"
          value={`stored ${r.storedExpertParameters}; activated ${r.activeExpertParameters}; dense reference 2`}
        />
      </div>
      <p>
        The output difference does not compare model quality. Real experts are trained networks;
        router, shared-expert, dispatch and hardware costs must be counted separately.
      </p>
    </>
  )
}
export type LearningLabKind =
  | 'learning'
  | 'positions'
  | 'features'
  | 'interventions'
  | 'post-training'
  | 'precision'
  | 'experts'
export function WikiLearningLab({ kind }: { kind: LearningLabKind }) {
  return kind === 'learning' ? (
    <LearningLoop />
  ) : kind === 'positions' ? (
    <Positions />
  ) : kind === 'features' ? (
    <Features />
  ) : kind === 'interventions' ? (
    <Interventions />
  ) : kind === 'post-training' ? (
    <PostTraining />
  ) : kind === 'precision' ? (
    <Precision />
  ) : (
    <Experts />
  )
}
