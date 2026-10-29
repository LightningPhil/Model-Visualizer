import { useRef, useState } from 'react'
import {
  downloadStudyLibrary,
  newStudyRecord,
  parseStudyLibrary,
  type StudyRecipe,
  type StudyRecord,
} from '../../model/study/records'
import { useStudyRecords } from './useStudyRecords'

export function StudyRecordForm({
  recipe,
  evidence,
  title,
  question: initialQuestion = '',
}: {
  recipe: StudyRecipe
  evidence: StudyRecord['evidence']
  title: string
  question?: string
}) {
  const library = useStudyRecords(),
    [question, setQuestion] = useState(initialQuestion),
    [prediction, setPrediction] = useState(''),
    [observation, setObservation] = useState(''),
    [conclusion, setConclusion] = useState(''),
    [status, setStatus] = useState('')
  const [savedRecord, setSavedRecord] = useState<StudyRecord>()
  return (
    <details className="study-record-form">
      <summary>Record this investigation</summary>
      <p>
        Saves the exact reproducible setup and the evidence shown here. Predictions and conclusions
        are your notes; saving does not mark an explanation as proven.
      </p>
      {(
        [
          ['Question', question, setQuestion],
          ['Prediction', prediction, setPrediction],
          ['Observation', observation, setObservation],
          ['Conclusion', conclusion, setConclusion],
        ] as const
      ).map(([label, value, set]) => (
        <label key={label}>
          {label}
          <textarea
            aria-label={`Investigation ${label.toLowerCase()}`}
            value={value}
            maxLength={label === 'Question' || label === 'Prediction' ? 4000 : 8000}
            onChange={(e) => set(e.target.value)}
          />
        </label>
      ))}
      <button
        className="button"
        onClick={() => {
          try {
            const record = newStudyRecord({
              title,
              question,
              prediction,
              observation,
              conclusion,
              recipe: structuredClone(recipe),
              evidence: structuredClone(evidence),
            })
            const persisted = library.save(record)
            setSavedRecord(record)
            setStatus(
              persisted
                ? 'Investigation saved with its setup and evidence.'
                : 'Investigation kept in this session. Export it to keep a durable copy.',
            )
          } catch (e) {
            setStatus(e instanceof Error ? e.message : 'Could not save investigation.')
          }
        }}
      >
        Save investigation
      </button>
      {(status || library.message) && (
        <p role="status">
          {status} {library.message}
        </p>
      )}
      {!library.writable && savedRecord && (
        <button onClick={() => downloadStudyLibrary([savedRecord])}>
          Export this session record
        </button>
      )}
    </details>
  )
}

export function StudyRecordLibrary({ onRestore }: { onRestore: (recipe: StudyRecipe) => void }) {
  const store = useStudyRecords(),
    file = useRef<HTMLInputElement>(null),
    [status, setStatus] = useState('')
  return (
    <section className="study-record-library" aria-label="Saved investigations">
      <h3>Saved investigations</h3>
      <p>
        Stored on this device, separately from model documents and notebooks. Export these records
        to share or back them up. Reopening reconstructs a setup; trained-model loading and
        inference still require their visible controls.
      </p>
      <div className="study-actions">
        <button onClick={() => downloadStudyLibrary(store.library.records)}>
          Export investigations
        </button>
        <button onClick={() => file.current?.click()}>Import investigations</button>
      </div>
      <input
        hidden
        ref={file}
        type="file"
        accept="application/json,.json"
        aria-label="Import investigation file"
        onChange={async (e) => {
          const f = e.target.files?.[0]
          e.target.value = ''
          if (!f) return
          try {
            if (f.size > 2_000_000) throw new Error('Investigation files must be under 2 MB.')
            const parsed = parseStudyLibrary(await f.text())
            const count = store.importRecords(parsed.records)
            setStatus(`Imported ${count} new investigations; existing IDs were retained.`)
          } catch (error) {
            setStatus(error instanceof Error ? error.message : 'Could not import investigations.')
          }
        }}
      />
      {(status || store.message) && <p role="status">{status || store.message}</p>}
      {!store.library.records.length && (
        <p>No saved investigations yet. Record an experiment below or from a numerical notebook.</p>
      )}
      {store.library.records.map((record) => (
        <details key={record.id}>
          <summary>
            {record.title} · {record.created.slice(0, 10)}
          </summary>
          <p className="eyebrow">{record.evidence.scope}</p>
          {(['question', 'prediction', 'observation', 'conclusion'] as const).map(
            (key) =>
              record[key] && (
                <p key={key}>
                  <b>{key}: </b>
                  {record[key]}
                </p>
              ),
          )}
          <p>
            <b>Captured evidence: </b>
            {record.evidence.summary}
          </p>
          {record.evidence.values && (
            <pre>
              {record.evidence.tensor} {JSON.stringify(record.evidence.shape)} · from cell{' '}
              {record.evidence.start ?? 0}
              {'\n'}
              {JSON.stringify(record.evidence.values)}
              {record.evidence.baseline &&
                `\nBaseline: ${JSON.stringify(record.evidence.baseline)}`}
            </pre>
          )}
          <div className="study-actions">
            <button onClick={() => onRestore(structuredClone(record.recipe))}>Reopen setup</button>
            <button onClick={() => downloadStudyLibrary([record])}>Export record</button>
            <button onClick={() => store.remove(record.id)}>Remove record</button>
          </div>
        </details>
      ))}
    </section>
  )
}
