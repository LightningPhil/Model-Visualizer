import { useState } from 'react'
import { investigationQuestions, investigationResult } from '../../model/study/investigations'
import type { StudyRecipe } from '../../model/study/records'
import { TrainingLab } from './TrainingLab'
import { StudyRecordForm, StudyRecordLibrary } from './StudyRecords'

export function InvestigationQuestions({ onChoose }: { onChoose: (id: string) => void }) {
  return (
    <section className="study-questions">
      <span className="eyebrow">START WITH A QUESTION</span>
      <h3>What would you like to find out?</h3>
      <div>
        {investigationQuestions.map((q) => (
          <button key={q.id} onClick={() => onChoose(q.id)}>
            {q.title}
            <span>Predict · experiment · explain →</span>
          </button>
        ))}
        <button onClick={() => onChoose('training')}>
          What actually changes when a network learns?<span>Train a small network →</span>
        </button>
      </div>
    </section>
  )
}
export function Investigations({
  initial,
  onRead,
  onRestore,
}: {
  initial?: StudyRecipe
  onRead: (id: string) => void
  onRestore: (recipe: StudyRecipe) => void
}) {
  const [kind, setKind] = useState(
      initial?.kind === 'investigation'
        ? initial.investigation
        : initial?.kind === 'training'
          ? 'training'
          : 'values',
    ),
    [value, setValue] = useState(initial?.kind === 'investigation' ? initial.value : 0),
    [query, setQuery] = useState(initial?.kind === 'investigation' ? initial.query : 1),
    [length, setLength] = useState(initial?.kind === 'investigation' ? initial.length : 1024),
    [prediction, setPrediction] = useState(''),
    [observe, setObserve] = useState(false)
  const q = investigationQuestions.find((q) => q.id === kind) ?? investigationQuestions[0]
  const result = investigationResult(q.id, value, query, length),
    f = (v: number[]) => `[${v.map((x) => Number(x.toFixed(5))).join(', ')}]`
  return (
    <div className="study-desk">
      <h3>Investigate a mechanism</h3>
      <p>
        Make a prediction, vary one factor, and keep the evidence. No score or completion badge: the
        aim is a useful explanation you can test.
      </p>
      <InvestigationQuestions
        onChoose={(id) => {
          setKind(id)
          setObserve(false)
          setPrediction('')
        }}
      />
      {kind === 'training' ? (
        <TrainingLab
          key={JSON.stringify(initial)}
          initial={initial?.kind === 'training' ? initial.settings : undefined}
        />
      ) : (
        <section className="study-investigation" aria-label="Guided investigation">
          <h3>{q.title}</h3>
          <p className="eyebrow">CONTROLLED SYNTHETIC EXAMPLE</p>
          <h4>1. Predict</h4>
          <p>{q.instruction}</p>
          <label>
            Your prediction
            <textarea
              aria-label="Guided prediction"
              value={prediction}
              maxLength={4000}
              onChange={(e) => setPrediction(e.target.value)}
            />
          </label>
          <details>
            <summary>A hint</summary>
            <p>{q.hint}</p>
          </details>
          <h4>2. Change one factor</h4>
          {q.id === 'memory' ? (
            <label>
              Retained positions
              <input
                aria-label="Investigation retained positions"
                type="number"
                min={1}
                max={8192}
                step={1}
                value={length}
                onChange={(e) => {
                  if (e.currentTarget.validity.valid && e.target.value)
                    setLength(Number(e.target.value))
                }}
              />
            </label>
          ) : (
            <label>
              {q.id === 'values' ? 'Value row 1, feature 0' : 'Head B query feature 0'}
              <input
                aria-label="Investigation edited feature"
                type="number"
                min={-3}
                max={3}
                step={0.25}
                value={q.id === 'values' ? value : query}
                onChange={(e) => {
                  if (e.currentTarget.validity.valid && e.target.value)
                    (q.id === 'values' ? setValue : setQuery)(Number(e.target.value))
                }}
              />
            </label>
          )}
          <button className="button" onClick={() => setObserve(true)}>
            Observe the calculation
          </button>
          {observe && (
            <div className="study-observation" role="status">
              <h4>3. Observe</h4>
              <p>
                {result.label}: {f(result.baseline)} → {f(result.current)}
              </p>
              {result.probabilities.length > 0 && (
                <>
                  <p>Baseline probability row: {f(result.baseProbabilities!)}</p>
                  <p>Edited probability row: {f(result.probabilities)}</p>
                  <p>
                    Fixed keys: [[1,0],[0,1],[1,1]]; baseline values: [[2,0],[0,4],[2,2]]; scale √2;
                    all three positions visible.
                  </p>
                </>
              )}
              <h4>4. Explain</h4>
              <p>{q.explanation}</p>
              <button onClick={() => onRead(q.guide)}>Read the full explanation →</button>
              <StudyRecordForm
                key={q.id}
                title={q.title}
                question={q.title}
                recipe={{ kind: 'investigation', investigation: q.id, value, query, length }}
                evidence={{
                  scope: 'Controlled synthetic investigation',
                  summary: `Prediction before observation: ${prediction}\n${q.instruction}\n${result.label}: ${f(result.baseline)} → ${f(result.current)}. Probabilities: ${f(result.probabilities)}.`,
                  values: result.current,
                  baseline: result.baseline,
                }}
              />
            </div>
          )}
        </section>
      )}
      <StudyRecordLibrary onRestore={onRestore} />
    </div>
  )
}
