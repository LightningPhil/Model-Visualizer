import { useEffect, useState } from 'react'
import { Control, Result } from './WikiLearningLab'
import {
  adamSequence,
  muonMatrixExample,
  type AdamSettings,
  type AdamMode,
  type Matrix2,
} from '../../help/wiki/optimizer-experiments'

const n = (v: number) => Number(v.toFixed(6)).toLocaleString('en-GB', { maximumFractionDigits: 6 })
const pair = (v: number[]) => `[${v.map(n).join(', ')}]`
const modes: [AdamMode, string][] = [
  ['adam', 'Plain Adam'],
  ['l2', 'Adam + coupled L2'],
  ['adamw', 'AdamW'],
]

function AdamLab() {
  const [gradients, setGradients] = useState([2, 2, -2, 0]),
    [step, setStep] = useState(0)
  const [settings, setSettings] = useState<AdamSettings>({
    initial: 2,
    rate: 0.1,
    beta1: 0.9,
    beta2: 0.999,
    epsilon: 1e-8,
    decay: 0.1,
  })
  useEffect(() => setStep(0), [gradients, settings])
  const traces = modes.map(([mode, label]) => ({
    mode,
    label,
    trace: adamSequence(gradients, settings, mode),
  }))
  function update(key: keyof AdamSettings, value: number) {
    setSettings((s) => ({ ...s, [key]: value }))
  }
  return (
    <>
      <h4>Follow moments and compare three separate parameter states</h4>
      <p>
        These four gradients are prescribed, not recomputed from a model as parameters diverge.
        Every supplied value—including zero—is a step. Each optimizer starts with the same scalar
        parameter and empty buffers; changing an assumption restarts the trace.
      </p>
      <div className="wiki-scratch-controls">
        <Control
          label="Optimizer initial parameter"
          value={settings.initial}
          change={(v) => update('initial', v)}
          min={-3}
          max={3}
          step={0.25}
        />
        <Control
          label="Optimizer learning rate"
          value={settings.rate}
          change={(v) => update('rate', v)}
          min={0}
          max={0.5}
          step={0.01}
        />
        <Control
          label="Optimizer decay coefficient"
          value={settings.decay}
          change={(v) => update('decay', v)}
          min={0}
          max={1}
          step={0.05}
        />
      </div>
      <div className="wiki-scratch-selects">
        <label>
          First-moment beta
          <select
            aria-label="Adam first moment beta"
            value={settings.beta1}
            onChange={(e) => update('beta1', Number(e.target.value))}
          >
            {[0, 0.5, 0.9, 0.99].map((v) => (
              <option key={v}>{v}</option>
            ))}
          </select>
        </label>
        <label>
          Second-moment beta
          <select
            aria-label="Adam second moment beta"
            value={settings.beta2}
            onChange={(e) => update('beta2', Number(e.target.value))}
          >
            {[0, 0.9, 0.99, 0.999].map((v) => (
              <option key={v}>{v}</option>
            ))}
          </select>
        </label>
        <label>
          Epsilon outside square root
          <select
            aria-label="Adam stabilizer epsilon"
            value={settings.epsilon}
            onChange={(e) => update('epsilon', Number(e.target.value))}
          >
            {[1e-8, 0.01, 1].map((v) => (
              <option key={v} value={v}>
                {v === 1e-8 ? '1e-8' : v}
              </option>
            ))}
          </select>
        </label>
      </div>
      <details className="wiki-edit-operands">
        <summary>Edit the four supplied gradients</summary>
        <div className="wiki-scratch-controls">
          {gradients.map((g, i) => (
            <Control
              key={i}
              label={`Supplied gradient ${i + 1}`}
              value={g}
              min={-4}
              max={4}
              step={0.25}
              change={(v) => setGradients((old) => old.map((value, j) => (j === i ? v : value)))}
            />
          ))}
        </div>
      </details>
      <div className="wiki-scenario-presets">
        <button onClick={() => setGradients([2, 2, -2, 0])}>Sign change then zero</button>
        <button onClick={() => setGradients([0, 0, 0, 0])}>Zero loss gradients</button>
        <button onClick={() => setGradients([2, 2, 2, 2])}>Constant gradient</button>
      </div>
      <div className="wiki-scenario-presets">
        <button disabled={step === 4} onClick={() => setStep(step + 1)}>
          Advance optimizer step
        </button>
        <button onClick={() => setStep(0)}>Restart optimizer trace</button>
      </div>
      <p className="wiki-scratch-insight">
        Showing optimizer step {step} of 4 · supplied loss gradients {pair(gradients)}. Plain Adam
        ignores the decay control; coupled L2 feeds it into moments; AdamW applies a separate
        shrinkage.
      </p>
      <div className="wiki-optimizer-cards">
        {traces.map(({ mode, label, trace }) => {
          const r = step ? trace[step - 1] : undefined
          return (
            <section key={mode} aria-label={`${label} scalar state`}>
              <h5>{label}</h5>
              <div className="wiki-calculation-grid">
                <Result
                  label="Parameter after this step"
                  value={n(r?.parameter ?? settings.initial)}
                />
                <Result
                  label="Gradient entering moment buffers"
                  value={r ? n(r.effectiveGradient) : 'No step yet'}
                />
                <Result
                  label="Raw first / second moments"
                  value={r ? pair([r.first, r.second]) : '[0, 0]'}
                />
                <Result
                  label="Bias-corrected first / second moments"
                  value={r ? pair([r.correctedFirst, r.correctedSecond]) : 'No step yet'}
                />
                <Result label="Adaptive subtraction" value={r ? n(r.adaptive) : '0'} />
                <Result label="Separate decay subtraction" value={r ? n(r.shrinkage) : '0'} />
              </div>
            </section>
          )
        })}
      </div>
      <p>
        The same prescribed gradient sequence isolates update rules. It is not a fair
        training-quality comparison, and no learned parameter changes. Try the zero-gradient preset
        from step zero to separate cold-start decay from historical momentum.
      </p>
    </>
  )
}
function Matrix({ title, matrix }: { title: string; matrix: Matrix2 }) {
  return (
    <div className="wiki-optimizer-matrix" role="group" aria-label={title}>
      <strong>{title}</strong>
      {matrix.map((row, i) => (
        <div key={i}>
          {row.map((v, j) => (
            <code key={j}>{n(v)}</code>
          ))}
        </div>
      ))}
    </div>
  )
}
function MuonLab() {
  const [first, setFirst] = useState(4),
    [second, setSecond] = useState(0.25),
    [oldFirst, setOldFirst] = useState(0),
    [oldSecond, setOldSecond] = useState(0)
  const [beta, setBeta] = useState(0.9),
    [steps, setSteps] = useState(5),
    [lookahead, setLookahead] = useState(true),
    [rate, setRate] = useState(0.02)
  const r = muonMatrixExample([first, second], [oldFirst, oldSecond], beta, steps, lookahead)
  const delta = r.output.map((row) => row.map((v) => -rate * v)) as Matrix2
  return (
    <>
      <h4>Inspect the matrix candidate and its finite polynomial transform</h4>
      <p>
        Restricted synthetic 2×2 family: gradient and old momentum share fixed left/right rotation
        bases (30° and −20°). Signed factors set their directional magnitudes. The transform uses
        the cited coefficients and binary64 arithmetic, not GPU bfloat16. The square shape
        multiplier is 1; this lab applies no weight decay.
      </p>
      <div className="wiki-scratch-controls">
        <Control
          label="Muon gradient factor 0"
          value={first}
          change={setFirst}
          min={-4}
          max={4}
          step={0.25}
        />
        <Control
          label="Muon gradient factor 1"
          value={second}
          change={setSecond}
          min={-4}
          max={4}
          step={0.25}
        />
        <Control
          label="Muon polynomial iterations"
          value={steps}
          change={setSteps}
          min={0}
          max={10}
        />
        <Control
          label="Muon momentum beta"
          value={beta}
          change={setBeta}
          min={0}
          max={0.99}
          step={0.01}
        />
        <Control
          label="Muon teaching step size"
          value={rate}
          change={setRate}
          min={0}
          max={0.1}
          step={0.01}
        />
      </div>
      <details className="wiki-edit-operands">
        <summary>Change the old momentum state</summary>
        <div className="wiki-scratch-controls">
          <Control
            label="Muon old momentum factor 0"
            value={oldFirst}
            change={setOldFirst}
            min={-2}
            max={2}
            step={0.25}
          />
          <Control
            label="Muon old momentum factor 1"
            value={oldSecond}
            change={setOldSecond}
            min={-2}
            max={2}
            step={0.25}
          />
        </div>
      </details>
      <label className="wiki-scratch-toggle">
        <input
          type="checkbox"
          checked={lookahead}
          onChange={(e) => setLookahead(e.target.checked)}
        />
        Use the reference Nesterov-like candidate
      </label>
      <div className="wiki-optimizer-matrices">
        <Matrix title="Supplied gradient matrix" matrix={r.gradient} />
        <Matrix title="Updated EMA momentum matrix" matrix={r.momentum} />
        <Matrix title="Candidate before normalization" matrix={r.candidate} />
        <Matrix title="Finite polynomial output" matrix={r.output} />
        <Matrix title="Ideal partial-polar target · separate reference" matrix={r.ideal} />
        <Matrix title="Proposed parameter delta · minus step size times output" matrix={delta} />
      </div>
      <Result label="Candidate Frobenius norm · epsilon is 1e-7" value={n(r.norm)} />
      <ol className="wiki-lab-history" aria-label="Muon singular-factor iteration history">
        {r.iterations.map((row) => (
          <li key={row.iteration}>
            Iteration {row.iteration}: signed factors {pair(row.factors)}; singular magnitudes{' '}
            {pair(row.magnitudes)}
          </li>
        ))}
      </ol>
      <p className="wiki-scratch-insight">
        Ideal nonzero target magnitudes: {pair(r.targetMagnitudes)}. The finite output is
        intentionally displayed separately: these coefficients do not guarantee exact unit singular
        values or monotonic convergence. Zero directions stay zero; no arbitrary full orthogonal
        extension is inserted.
      </p>
      <p>
        This is a matrix-update illustration, not a training benchmark or a transformation of the
        loaded model’s weights. Eligible parameter groups, rectangular scaling and auxiliary
        optimizers must be specified separately.
      </p>
    </>
  )
}
export function WikiOptimizerLab({ kind }: { kind: 'adam-lab' | 'muon-lab' }) {
  return kind === 'adam-lab' ? <AdamLab /> : <MuonLab />
}
