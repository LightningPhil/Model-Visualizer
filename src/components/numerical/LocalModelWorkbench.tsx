import { useEffect, useMemo, useRef, useState } from 'react'
import { Cpu, Play, ShieldCheck } from 'lucide-react'
import type { ModelDocument } from '../../model/schema'
import type { NumericalTrace } from '../../model/numerics/engine'
import {
  checkpointStudySession,
  loadCheckpointSession,
  type CheckpointSession,
  trainedSettingsSchema,
} from '../../model/checkpoints/session'
import {
  StoriesTokenizer,
  storiesManifest,
  type TrainedRunSettings,
} from '../../model/checkpoints/stories'
import { NumericalWorkbench } from './NumericalWorkbench'
import { Dialog } from '../Dialog'
import './numerical.css'
import {
  headInterventionSchema,
  type HeadIntervention,
  type StudyRecipe,
} from '../../model/study/records'

type Reply = {
  id: number
  ready?: boolean
  trace?: NumericalTrace
  baseline?: NumericalTrace
  error?: string
  progress?: { count: number; text: string }
}
class LocalCheckpointClient {
  private worker = new Worker(
    new URL('../../model/checkpoints/checkpoint.worker.ts', import.meta.url),
    { type: 'module' },
  )
  private id = 0
  private stopped = false
  private pending = new Map<
    number,
    { resolve: (reply: Reply) => void; reject: (error: Error) => void }
  >()
  constructor(onProgress: (progress: { count: number; text: string }) => void) {
    this.worker.onmessage = (event: MessageEvent<Reply>) => {
      const entry = this.pending.get(event.data.id)
      if (!entry) return
      if (event.data.progress) {
        onProgress(event.data.progress)
        return
      }
      this.pending.delete(event.data.id)
      if (event.data.error) entry.reject(new Error(event.data.error))
      else entry.resolve(event.data)
    }
    this.worker.onerror = () => {
      this.stop('The local inference worker failed. Reload the checkpoint to retry.')
    }
  }
  request(message: object, transfer: Transferable[] = []) {
    const id = ++this.id
    return new Promise<Reply>((resolve, reject) => {
      if (this.stopped) {
        reject(new Error('Reload the checkpoint before running again.'))
        return
      }
      this.pending.set(id, { resolve, reject })
      this.worker.postMessage({ ...message, id }, transfer)
    })
  }
  stop(reason = 'Local model stopped.') {
    this.worker.terminate()
    this.stopped = true
    this.pending.forEach((p) => p.reject(new Error(reason)))
    this.pending.clear()
  }
}

export function LocalModelWorkbench({
  doc,
  reference,
  onSave,
  onClose,
  onTheme,
  notify,
  initialStudy,
  onRestoreStudy,
}: {
  doc: ModelDocument
  reference: ModelDocument
  onSave: (session: CheckpointSession) => void
  onClose: () => void
  onTheme: () => void
  notify: (message: string) => void
  initialStudy?: StudyRecipe
  onRestoreStudy?: (recipe: StudyRecipe) => void
}) {
  const [initial] = useState(() => loadCheckpointSession(doc))
  const compatibleRestore =
    initialStudy?.kind === 'trained' &&
    initialStudy.checkpoint === storiesManifest.id &&
    initialStudy.revision === storiesManifest.revision
      ? initialStudy
      : undefined
  const [session, setSession] = useState(() =>
    compatibleRestore
      ? { ...initial.session, settings: compatibleRestore.settings }
      : initial.session,
  )
  const [trace, setTrace] = useState<NumericalTrace>()
  const [baseline, setBaseline] = useState<NumericalTrace>()
  const [intervention, setIntervention] = useState<HeadIntervention>(
    compatibleRestore?.intervention ?? {
      mode: 'zero',
      layer: 0,
      head: 0,
      donorPrompt: 'Once upon a time, there was a dog.',
    },
  )
  const [tokenizer, setTokenizer] = useState<StoriesTokenizer>()
  const [busy, setBusy] = useState(''),
    [error, setError] = useState('')
  const client = useRef<LocalCheckpointClient | undefined>(undefined),
    request = useRef(0),
    mounted = useRef(true),
    saveRef = useRef(onSave)
  saveRef.current = onSave
  useEffect(() => {
    if (initial.writable) saveRef.current(session)
  }, [session, initial.writable])
  useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
      request.current++
      client.current?.stop()
    }
  }, [])
  const runtimeReference = useMemo(
    () => ({
      ...reference,
      sources: [
        ...reference.sources,
        {
          id: 'stories260k',
          title: 'Stories260K · checkpoint, tokenizer and training',
          url: storiesManifest.training,
          note: 'Original author’s training recipe and greedy sample.',
        },
        {
          id: 'llama2c',
          title: 'llama2.c · checkpoint and inference implementation',
          url: storiesManifest.implementation,
          note: 'Pinned reference for weight layout, tokenizer, RoPE, RMSNorm and GQA.',
        },
      ],
    }),
    [reference],
  )
  const update = (change: Partial<TrainedRunSettings>) =>
    setSession((s) => ({ ...s, settings: { ...s.settings, ...change } }))
  async function load() {
    const generation = ++request.current
    client.current?.stop()
    client.current = new LocalCheckpointClient((progress) => {
      if (mounted.current)
        setBusy(
          `Generated ${progress.count} token${progress.count === 1 ? '' : 's'} locally: ${progress.text || '(special token)'}`,
        )
    })
    setError('')
    setBusy('Reading the bundled checkpoint and tokenizer…')
    try {
      const buffers = await Promise.all(
        [storiesManifest.weights, storiesManifest.tokenizer].map(async (asset) => {
          const response = await fetch(
            `${import.meta.env.BASE_URL}models/stories260k/${asset.file}`,
          )
          if (!response.ok)
            throw new Error(`Could not load ${asset.file}: HTTP ${response.status}.`)
          const buffer = await response.arrayBuffer()
          if (buffer.byteLength !== asset.bytes) throw new Error(`Invalid ${asset.file} size.`)
          return buffer
        }),
      )
      if (!mounted.current || generation !== request.current) return
      setBusy('Verifying SHA-256 and loading trained weights…')
      const matchingTokenizer = new StoriesTokenizer(buffers[1].slice(0))
      await client.current.request({ weights: buffers[0], tokenizer: buffers[1] }, buffers)
      if (!mounted.current || generation !== request.current) return
      setTokenizer(matchingTokenizer)
      setBusy('')
    } catch (cause) {
      if (mounted.current && generation === request.current) {
        setBusy('')
        setError(cause instanceof Error ? cause.message : 'Could not load the local model.')
      }
    }
  }
  async function run(intervene = false) {
    if (!tokenizer || !client.current) return
    const generation = ++request.current
    setError('')
    try {
      const settings = trainedSettingsSchema.parse(session.settings),
        ids = tokenizer.encode(settings.prompt)
      if (ids.length > 48)
        throw new Error(
          `This inspection run accepts at most 48 prompt tokens (including BOS); this prompt has ${ids.length}. Shorten it to retain every activation.`,
        )
      setBusy('Computing trained-model activations locally…')
      const result = await client.current.request({
        settings,
        ...(compatibleRestore?.comparison ? { comparison: compatibleRestore.comparison } : {}),
        ...(intervene ? { intervention: headInterventionSchema.parse(intervention) } : {}),
      })
      if (mounted.current && generation === request.current && result.trace) {
        setTrace(result.trace)
        setBaseline(result.baseline)
        setBusy('')
      }
    } catch (cause) {
      if (mounted.current && generation === request.current) {
        setBusy('')
        setError(cause instanceof Error ? cause.message : 'Local generation failed.')
      }
    }
  }
  const controls = (
    <div className="trained-controls">
      <span className="eyebrow">LOCAL CHECKPOINT · TRAINED</span>
      <h3>Stories260K</h3>
      {compatibleRestore && (
        <p>
          Reopened saved setup. Load the checkpoint, then{' '}
          {compatibleRestore.intervention
            ? 'use “Run controlled intervention” below to replay its replacement.'
            : 'generate to reconstruct the captured run.'}
          {compatibleRestore.comparison
            ? ' The saved comparison is also reconstructed on an ordinary generation run.'
            : ''}
        </p>
      )}
      <p>
        A small TinyStories completion model. 260,032 learned parameters. Five layers; GQA 8 query /
        4 KV heads; tied embedding/output weights. Its language ability is limited.
      </p>
      <label>
        Prompt
        <textarea
          aria-label="Local model prompt"
          maxLength={600}
          value={session.settings.prompt}
          onChange={(e) => update({ prompt: e.target.value.replaceAll('\0', '') })}
        />
      </label>
      <p className="numeric-muted">
        Matching 512-token scored BPE · BOS included ·{' '}
        {tokenizer ? tokenizer.encode(session.settings.prompt).length : '…'} prompt tokens ·
        inspection limit 48
      </p>
      <label>
        New tokens
        <select
          aria-label="Local model output budget"
          value={session.settings.generate}
          onChange={(e) => update({ generate: Number(e.target.value) })}
        >
          {[1, 4, 8, 12, 16].map((n) => (
            <option key={n}>{n}</option>
          ))}
        </select>
      </label>
      <label>
        Selection
        <select
          aria-label="Local model selection"
          value={session.settings.sampling}
          onChange={(e) => update({ sampling: e.target.value as TrainedRunSettings['sampling'] })}
        >
          <option value="greedy">Greedy argmax</option>
          <option value="sample">Seeded categorical</option>
        </select>
      </label>
      <label>
        Sampling seed
        <input
          aria-label="Local model sampling seed"
          type="number"
          min={0}
          max={2147483647}
          value={session.settings.seed}
          onChange={(e) => {
            const seed = Number(e.target.value)
            if (Number.isInteger(seed) && seed >= 0 && seed <= 2147483647) update({ seed })
          }}
        />
      </label>
      <label>
        Temperature <b>{session.settings.temperature}</b>
        <input
          aria-label="Local model temperature"
          type="range"
          min={0.1}
          max={3}
          step={0.1}
          value={session.settings.temperature}
          onChange={(e) => update({ temperature: Number(e.target.value) })}
        />
      </label>
      <p>
        Architecture and full causal KV retention match the checkpoint. No architecture switches,
        MTP or chat template are applied.
      </p>
      {error && (
        <p className="trained-error" role="alert">
          {error}
        </p>
      )}
      {busy && <p role="status">{busy}</p>}
      <button className="button" disabled={!tokenizer || Boolean(busy)} onClick={() => void run()}>
        <Play size={14} />
        {trace ? 'Generate another trace' : 'Generate & inspect'}
      </button>
      <details className="trained-interventions">
        <summary>Experiment on a trained head</summary>
        <p>
          Replace one query head’s output vector at the final prompt position, before concatenation
          and Wo. Run the unchanged baseline and replacement with the same prompt and checkpoint.
          Both runs compute one next-token distribution, regardless of the generation budget above.
        </p>
        <label>
          Replacement
          <select
            aria-label="Trained intervention replacement"
            value={intervention.mode}
            onChange={(e) =>
              setIntervention((s) => ({ ...s, mode: e.target.value as HeadIntervention['mode'] }))
            }
          >
            <option value="zero">Zero ablation</option>
            <option value="patch">Patch from a donor prompt</option>
          </select>
        </label>
        <label>
          Layer
          <select
            aria-label="Trained intervention layer"
            value={intervention.layer}
            onChange={(e) => setIntervention((s) => ({ ...s, layer: Number(e.target.value) }))}
          >
            {[0, 1, 2, 3, 4].map((i) => (
              <option key={i}>{i}</option>
            ))}
          </select>
        </label>
        <label>
          Query head
          <select
            aria-label="Trained intervention head"
            value={intervention.head}
            onChange={(e) => setIntervention((s) => ({ ...s, head: Number(e.target.value) }))}
          >
            {[0, 1, 2, 3, 4, 5, 6, 7].map((i) => (
              <option key={i}>{i}</option>
            ))}
          </select>
        </label>
        {intervention.mode === 'patch' && (
          <label>
            Donor prompt
            <textarea
              aria-label="Trained donor prompt"
              maxLength={600}
              value={intervention.donorPrompt ?? ''}
              onChange={(e) =>
                setIntervention((s) => ({ ...s, donorPrompt: e.target.value.replaceAll('\0', '') }))
              }
            />
          </label>
        )}
        <p>
          A donor patch takes that head’s final-position vector from the donor run using the same
          checkpoint. Different token lengths mean different absolute positions; this is an explicit
          counterfactual, not a position-matched claim. Donor prompts have the same 48-token bound.
        </p>
        <button
          className="button"
          disabled={!tokenizer || Boolean(busy)}
          onClick={() => void run(true)}
        >
          Run controlled intervention
        </button>
      </details>
      {busy && (
        <button
          onClick={() => {
            request.current++
            client.current?.stop()
            client.current = undefined
            setBusy('')
            setTokenizer(undefined)
            setError('Stopped. Reload the checkpoint before another run.')
          }}
        >
          Stop local computation
        </button>
      )}
      <button className="button" disabled={Boolean(busy)} onClick={load}>
        <Cpu size={15} />
        {tokenizer ? 'Reload checkpoint' : 'Load trained model · 1.01 MiB'}
      </button>
      <details>
        <summary>Checkpoint provenance & arithmetic</summary>
        <p>
          MIT · bundled assets from the pinned author revision. SHA-256 verified before inference.
          Your prompt and activations stay in this browser. FP32 tensors with JavaScript Number
          accumulation; this reference kernel can differ slightly from C/GPU rounding. All weights
          and activations are retained for this bounded run. Playback replays the captured
          calculation; it does not slow the kernel itself.
        </p>
        <code>{storiesManifest.revision}</code>
        <p>Weights SHA-256</p>
        <code>{storiesManifest.weights.sha256}</code>
        <p>Tokenizer SHA-256</p>
        <code>{storiesManifest.tokenizer.sha256}</code>
        <a href={storiesManifest.source} target="_blank" rel="noreferrer">
          Checkpoint & tokenizer source ↗
        </a>
        <a href={storiesManifest.training} target="_blank" rel="noreferrer">
          Training recipe & reference output ↗
        </a>
      </details>
    </div>
  )
  if (trace && tokenizer)
    return (
      <NumericalWorkbench
        doc={doc}
        reference={runtimeReference}
        onClose={onClose}
        onTheme={onTheme}
        notify={notify}
        onRestoreStudy={onRestoreStudy}
        initialStudy={initialStudy}
        onSave={(study) =>
          setSession((s) => ({
            ...s,
            study: { layout: study.layout, bookmarks: study.bookmarks, notes: study.notes },
          }))
        }
        trained={{
          trace,
          baseline,
          clearBaseline: () => setBaseline(undefined),
          tokenizer,
          controls,
          session: checkpointStudySession(session),
          writable: initial.writable,
        }}
      />
    )
  return (
    <Dialog
      title="A real model. Inside your browser."
      className="local-model-modal"
      onClose={onClose}
    >
      <div className="trained-intro">
        <div>
          <Cpu size={40} />
          <span className="eyebrow">REAL WEIGHTS · LOCAL INFERENCE</span>
          <h2>
            Open the model.
            <br />
            Follow its calculations.
          </h2>
          <p>
            Generate a short completion, then travel through the exact attention, weights,
            intermediate activations and cached state behind each token.
          </p>
          <p>
            <ShieldCheck size={15} /> No account, service or GPU required. Model files are bundled;
            loading and generation are explicit.
          </p>
          <p>
            260K is deliberately tiny: this is a trained learning specimen, not a capable assistant.
            The original untrained laboratory stays separate.
          </p>
          {!initial.writable && (
            <p role="status">
              An unsupported saved checkpoint session is preserved. This temporary session will not
              overwrite it.
            </p>
          )}
        </div>
        {controls}
      </div>
    </Dialog>
  )
}
