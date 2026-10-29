import { useMemo, useState } from 'react'
import { MathBlock } from '../Math'
import {
  defaultTraining,
  trainNetwork,
  trainingData,
  holdoutData,
  networkForward,
  parameterNames,
  targetFunction,
  type TrainingRecipe,
} from '../../model/study/training'
import { StudyRecordForm } from './StudyRecords'

const n = (v: number) => Number(v.toPrecision(5)).toString()
export function TrainingLab({ initial = defaultTraining }: { initial?: TrainingRecipe }) {
  const [recipe, setRecipe] = useState(initial),
    [sample, setSample] = useState(0),
    [parameter, setParameter] = useState(0)
  const run = useMemo(() => trainNetwork(recipe), [recipe]),
    data = trainingData(recipe.task),
    point = data[sample] ?? data[0],
    forward = networkForward(run.parameters, point.x),
    last = run.history.at(-1)!
  function change(change: Partial<TrainingRecipe>) {
    setRecipe((r) => ({ ...r, ...change, steps: 0 }))
  }
  const curve = Array.from({ length: 81 }, (_, i) => -1 + i / 40),
    sx = (x: number) => 30 + (x + 1) * 150,
    sy = (y: number) => 120 - Math.max(-2, Math.min(2, y)) * 45
  return (
    <section className="study-training" aria-label="Trainable network">
      <span className="eyebrow">SYNTHETIC DATA · 13 TRAINABLE PARAMETERS · LOCAL</span>
      <h3>Watch a small network learn</h3>
      <p>
        A 1 → 4 tanh → 1 network fits five visible examples. Every step recomputes the gradient on
        the full training set, then updates all parameters together. Holdout examples never supply
        gradients. These are actual computations from this network, independent of the transformer
        checkpoint.
      </p>
      <div className="study-controls">
        <label>
          Dataset
          <select
            aria-label="Training dataset"
            value={recipe.task}
            onChange={(e) => change({ task: e.target.value as TrainingRecipe['task'] })}
          >
            <option value="curve">Fit a smooth curve</option>
            <option value="extrapolate">Train only in the middle</option>
            <option value="noise">Fit noisy observations</option>
          </select>
        </label>
        <label>
          Optimizer
          <select
            aria-label="Network optimizer"
            value={recipe.optimizer}
            onChange={(e) => change({ optimizer: e.target.value as TrainingRecipe['optimizer'] })}
          >
            <option value="sgd">SGD</option>
            <option value="adam">Adam</option>
            <option value="adamw">AdamW</option>
          </select>
        </label>
        <label>
          Learning rate
          <input
            aria-label="Network learning rate"
            type="number"
            min={0.001}
            max={2}
            step={0.001}
            value={recipe.rate}
            onChange={(e) => {
              if (e.currentTarget.validity.valid && e.target.value)
                change({ rate: Number(e.target.value) })
            }}
          />
        </label>
        <label>
          Decay (AdamW only)
          <input
            aria-label="Network weight decay"
            type="number"
            min={0}
            max={1}
            step={0.01}
            disabled={recipe.optimizer !== 'adamw'}
            value={recipe.decay}
            onChange={(e) => {
              if (e.currentTarget.validity.valid && e.target.value)
                change({ decay: Number(e.target.value) })
            }}
          />
        </label>
        <label>
          Initialization seed
          <input
            aria-label="Network initialization seed"
            type="number"
            min={0}
            max={2147483647}
            step={1}
            value={recipe.seed}
            onChange={(e) => {
              if (e.currentTarget.validity.valid && e.target.value)
                change({ seed: Number(e.target.value) })
            }}
          />
        </label>
      </div>
      <div className="study-actions">
        <button
          disabled={recipe.steps >= 1000 || run.stopped}
          onClick={() => setRecipe((r) => ({ ...r, steps: r.steps + 1 }))}
        >
          Train one step
        </button>
        <button
          disabled={recipe.steps >= 1000 || run.stopped}
          onClick={() => setRecipe((r) => ({ ...r, steps: Math.min(1000, r.steps + 25) }))}
        >
          Train 25 steps
        </button>
        <button onClick={() => setRecipe((r) => ({ ...r, steps: 0 }))}>Reset network</button>
      </div>
      <p role="status">
        Step {run.completed} · training half-MSE {n(last.train)} · holdout half-MSE{' '}
        {n(last.holdout)}
        {run.stopped && ' · Stopped before a nonfinite update.'}
      </p>
      <svg
        viewBox="0 0 360 240"
        role="img"
        aria-label="Network predictions, target curve and five training samples"
      >
        <line x1="30" y1="120" x2="330" y2="120" stroke="currentColor" opacity=".3" />
        <polyline
          points={curve.map((x) => `${sx(x)},${sy(targetFunction(x))}`).join(' ')}
          fill="none"
          stroke="var(--secondary)"
          strokeDasharray="5 4"
        />
        <polyline
          points={curve
            .map((x) => `${sx(x)},${sy(networkForward(run.parameters, x).output)}`)
            .join(' ')}
          fill="none"
          stroke="var(--cyan)"
          strokeWidth="2"
        />
        {data.map((p, i) => (
          <circle key={i} cx={sx(p.x)} cy={sy(p.y)} r="4" fill="var(--amber)" />
        ))}
        <text x="30" y="232" fill="currentColor" fontSize="10">
          x: −1 … 1 · display y clipped to [−2, 2]
        </text>
      </svg>
      <p>
        Solid cyan: prediction. Dashed: noiseless target sin(πx). Dots: the five training
        observations. The holdout loss averages eight fixed noiseless points; it is a small
        diagnostic, not a guarantee of generalization.
      </p>
      <details>
        <summary>Inspect every training and holdout observation</summary>
        <div className="study-table-wrap">
          <table>
            <thead>
              <tr>
                <th>Set</th>
                <th>x</th>
                <th>Target</th>
                <th>Prediction</th>
                <th>Half squared error</th>
              </tr>
            </thead>
            <tbody>
              {[
                ...data.map((p) => ({ ...p, set: 'Training' })),
                ...holdoutData.map((p) => ({ ...p, set: 'Holdout' })),
              ].map((p, i) => {
                const prediction = networkForward(run.parameters, p.x).output
                return (
                  <tr key={i}>
                    <th>{p.set}</th>
                    <td>{p.x}</td>
                    <td>{n(p.y)}</td>
                    <td>{n(prediction)}</td>
                    <td>{n((prediction - p.y) ** 2 / 2)}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </details>
      <details>
        <summary>Every training and holdout measurement</summary>
        <div className="study-table-wrap">
          <table>
            <thead>
              <tr>
                <th>Step</th>
                <th>Training half-MSE</th>
                <th>Holdout half-MSE</th>
              </tr>
            </thead>
            <tbody>
              {run.history.map((h) => (
                <tr key={h.step}>
                  <td>{h.step}</td>
                  <td>{n(h.train)}</td>
                  <td>{n(h.holdout)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
      <details open>
        <summary>Forward values and backward gradients</summary>
        <label>
          Inspect training sample
          <select
            aria-label="Inspect training sample"
            value={sample}
            onChange={(e) => setSample(Number(e.target.value))}
          >
            {data.map((p, i) => (
              <option key={i} value={i}>
                x={p.x}, target={n(p.y)}
              </option>
            ))}
          </select>
        </label>
        <p>
          x = {point.x}; target = {n(point.y)}; hidden preactivations z = [
          {forward.z.map(n).join(', ')}]; tanh(z) = [{forward.h.map(n).join(', ')}]; prediction ={' '}
          {n(forward.output)}; error = {n(forward.output - point.y)}.
        </p>
        <MathBlock
          tex={String.raw`h_j=\tanh(xw_j+b_j),\quad \hat y=\sum_{j=0}^{3}v_jh_j+c,\quad L=\frac1{2N}\sum_{i=1}^{N}(\hat y_i-y_i)^2`}
        />
        <MathBlock
          tex={String.raw`\frac{\partial L}{\partial w_j}=\frac1N\sum_i(\hat y_i-y_i)v_j(1-h_{ij}^2)x_i`}
        />
        <p>
          The error flows backwards through the output weight v, then the tanh slope 1−h², then the
          input x. Bias gradients omit x. Gradients below average all five samples. “Last gradient”
          and moment buffers belong to the last update; “next gradient” is recomputed at the
          displayed parameters.
        </p>
        <label>
          Inspect parameter
          <select
            aria-label="Inspect network parameter"
            value={parameter}
            onChange={(e) => setParameter(Number(e.target.value))}
          >
            {parameterNames.map((p, i) => (
              <option key={p} value={i}>
                {p}
              </option>
            ))}
          </select>
        </label>
        <p>
          {parameterNames[parameter]} = {n(run.parameters[parameter])}; next gradient ={' '}
          {n(run.current.gradient[parameter])}; last gradient = {n(run.lastGradient[parameter])};
          last update Δ = {n(run.lastDelta[parameter])}.
        </p>
        {recipe.optimizer !== 'sgd' && (
          <p>
            Raw m = {n(run.m[parameter])}; raw v = {n(run.v[parameter])}. β₁=0.9, β₂=0.999, ε=10⁻⁸;
            bias correction uses the completed step. AdamW decays all 13 parameters, including
            biases, in this explicit teaching recipe.
          </p>
        )}
        <div className="study-table-wrap">
          <table>
            <thead>
              <tr>
                <th>Parameter</th>
                <th>Value</th>
                <th>Next gradient</th>
                <th>Last Δ</th>
              </tr>
            </thead>
            <tbody>
              {parameterNames.map((name, i) => (
                <tr key={name}>
                  <th>{name}</th>
                  <td>{n(run.parameters[i])}</td>
                  <td>{n(run.current.gradient[i])}</td>
                  <td>{n(run.lastDelta[i])}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p>
          Try the middle-only dataset and inspect the ends. Try SGD with rate 2 and compare each
          measured loss; a downhill local derivative does not guarantee a finite update improves it.
          Noise fitting and holdout behavior depend on the seed and step count.
        </p>
        <p>
          Derivatives:{' '}
          <a href="https://www.nature.com/articles/323533a0" target="_blank" rel="noreferrer">
            Rumelhart, Hinton & Williams
          </a>
          . Updates:{' '}
          <a href="https://arxiv.org/abs/1412.6980" target="_blank" rel="noreferrer">
            Adam
          </a>
          ,{' '}
          <a href="https://arxiv.org/abs/1711.05101" target="_blank" rel="noreferrer">
            AdamW
          </a>
          . This network and dataset are authored synthetic examples.
        </p>
      </details>
      <StudyRecordForm
        recipe={{ kind: 'training', settings: recipe }}
        title="Train a small network"
        question="How do the training and holdout errors change as this network learns?"
        evidence={{
          scope: 'Synthetic trainable network',
          summary: `Step ${run.completed}; training half-MSE ${last.train}; holdout half-MSE ${last.holdout}; fixed five-example full batch; JS Number arithmetic.`,
          values: run.parameters,
        }}
      />
    </section>
  )
}
