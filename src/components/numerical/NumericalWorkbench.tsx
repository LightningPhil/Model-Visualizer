import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
  type Dispatch,
  type SetStateAction,
} from 'react'
import {
  ArrowLeft,
  ArrowRight,
  Bookmark,
  BookOpen,
  Check,
  ChevronLeft,
  ChevronRight,
  Download,
  FlaskConical,
  Pause,
  Play,
  RotateCcw,
  Settings2,
  Sun,
  Moon,
} from 'lucide-react'
import type { ModelDocument } from '../../model/schema'
import { dimensions, formatBytes } from '../../model/schema'
import { residentCount } from '../../model/attention'
import {
  latestBinding,
  type NumericalFrame,
  type NumericalSettings,
  type NumericalTrace,
} from '../../model/numerics/engine'
import {
  loadNumericalSession,
  numericalGraph,
  type NumericalSession,
} from '../../model/numerics/session'
import { unravel, tensorComparisonIssue, type NumericTensor } from '../../model/numerics/tensor'
import { explainCell, valueLabel, type OperandCell } from '../../model/numerics/explain'
import { Dialog } from '../Dialog'
import { GraphCanvas, type CameraFocus, type Selection } from '../GraphCanvas'
import { MathBlock } from '../Math'
import { CachePolicyControls } from '../CachePolicyControls'
import { TooltipSettingsButton } from '../help/TooltipProvider'
import { useWiki, useWikiFollow } from '../help/WikiProvider'
import { StudyRecordForm, StudyRecordLibrary } from '../help/StudyRecords'
import { RunComparison } from './RunComparison'
import type { StudyRecipe } from '../../model/study/records'
import { TensorView } from './TensorView'
import { CellCalculation } from './CellCalculation'
import { HeadObservatory } from './HeadObservatory'
import { ParameterLibrary } from './ParameterLibrary'
import { resolveNumericalExample } from '../../model/numerics/examples'
import { inspectConnection } from '../../model/numerics/inspection'
import { prepareTraceExport } from '../../model/numerics/trace-export'
import { useReferenceTrace } from './useReferenceTrace'
import './numerical.css'

const questions = [
  {
    title: 'How is one attention score made?',
    node: 'layer-0.attn.scores-0',
    index: 0,
    call: 0,
    description:
      'One query row meets one key column. Choose any result cell, then follow the products back to Q and K.',
  },
  {
    title: 'Why is the future invisible?',
    node: 'layer-0.attn.scale-0',
    index: 1,
    call: 0,
    description:
      'This query is at position 0. Key position 1 is in its future. The mask produces −∞ before softmax, which gives it exactly zero probability.',
    multiple: true,
  },
  {
    title: 'Where does a probability come from?',
    node: 'layer-0.attn.softmax-0',
    index: 0,
    call: 0,
    description:
      'Scores become a row-normalized distribution. Inspect the row maximum, exponentials and denominator, then compare another query.',
  },
  {
    title: 'How does RoPE rotate a vector?',
    node: 'layer-0.attn.ropeq',
    index: 12,
    call: 0,
    description:
      'Two features rotate together. Select a different token or feature pair to change the absolute position or frequency.',
  },
  {
    title: 'What does a gate actually do?',
    node: 'layer-0.swiglu.multiply',
    index: 0,
    call: 0,
    description:
      'The SiLU gate multiplies each up-branch feature individually. Follow both operands to see their separate projections.',
  },
  {
    title: 'What survives into the next token?',
    node: 'layer-0.attn.cachek',
    index: 0,
    call: 1,
    description:
      'The next decoder call reuses this layer’s retained K/V. Change retention in Run setup to compare full, sparse and sliding state.',
    checkpointDescription:
      'This checkpoint retains the full causal prefix. Each layer reuses its own K/V with absolute positions. A selected output ID enters the cache only when the next forward pass consumes it.',
  },
]
const chapters = [
  ['Input', 'tokens.lookup'],
  ['L0 · attention', 'layer-0.norm1'],
  ['L0 · feed-forward', 'layer-0.norm2'],
  ['L1 · attention', 'layer-1.norm1'],
  ['L1 · feed-forward', 'layer-1.norm2'],
  ['Vocabulary', 'output.lmhead'],
  ['Choose & loop', 'output.sample'],
]
function firstCell(
  trace: NumericalTrace,
  frame: NumericalFrame,
  payload = frame.outputs[0],
): OperandCell | undefined {
  const t = trace.tensors[payload]
  if (!t) return undefined
  const choice = frame.outputs.map((id) => trace.tensors[id]).find((t) => t.recipe.op === 'choice')
  return {
    tensor: t.id,
    index:
      t.recipe.op === 'softmax' && t.axes.at(-1) === 'V'
        ? (t.shape.at(-2)! - 1) * t.shape.at(-1)! + (choice?.data[0] ?? 0)
        : 0,
  }
}
function downloadTrace(trace: NumericalTrace, blob: Blob) {
  const url = URL.createObjectURL(blob),
    a = document.createElement('a')
  a.href = url
  a.download = trace.checkpoint
    ? `${trace.checkpoint.id}-trace.json`
    : `numerical-teaching-run-seed-${trace.settings.seed}.json`
  a.click()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}

type WorkbenchProps = {
  doc: ModelDocument
  reference: ModelDocument
  initialEntry?: string
  initialStudy?: StudyRecipe
  onRestoreStudy?: (recipe: StudyRecipe) => void
  onSave: (session: NumericalSession) => void
  onClose: () => void
  onTheme: () => void
  notify: (message: string) => void
  trained?: {
    trace: NumericalTrace
    baseline?: NumericalTrace
    clearBaseline?: () => void
    session: NumericalSession
    writable: boolean
    controls: ReactNode
    tokenizer: { piece: (id: number) => string; decode: (ids: number[]) => string }
  }
}

export function NumericalWorkbench(props: WorkbenchProps) {
  const [initial] = useState(() =>
    props.trained
      ? { session: props.trained.session, writable: props.trained.writable }
      : loadNumericalSession(props.doc),
  )
  const [session, setSession] = useState(() =>
    props.initialStudy?.kind === 'numerical'
      ? {
          ...initial.session,
          settings: props.initialStudy.settings,
          pinned: props.initialStudy.baseline,
          ...(props.initialStudy.settings.attention !== initial.session.settings.attention
            ? {
                layouts: {
                  ...initial.session.layouts,
                  [initial.session.settings.attention]: initial.session.layout,
                },
                layout: initial.session.layouts[props.initialStudy.settings.attention] ?? {
                  direction: initial.session.layout.direction,
                  positions: {},
                },
              }
            : {}),
        }
      : initial.session,
  )
  const current = useReferenceTrace(props.trained ? undefined : session.settings)
  const comparison = useReferenceTrace(props.trained ? undefined : session.pinned)
  const trace = props.trained?.trace ?? current.trace
  if (!trace)
    return (
      <Dialog title="The mathematics, alive." className="numerical-modal" onClose={props.onClose}>
        <div className="numerical-worker-status" role="status">
          {current.error ?? 'Computing the reference run in a background worker…'}
          {current.error && (
            <button className="button" onClick={current.retry}>
              Retry run
            </button>
          )}
        </div>
      </Dialog>
    )
  return (
    <NumericalWorkbenchView
      {...props}
      initial={initial}
      session={session}
      setSession={setSession}
      trace={trace}
      baseline={comparison.trace}
      computing={current.busy || comparison.busy}
      computationError={comparison.error}
      retryComparison={comparison.retry}
    />
  )
}

function NumericalWorkbenchView({
  doc,
  reference,
  initialEntry,
  initialStudy,
  onRestoreStudy,
  onSave,
  onClose,
  onTheme,
  notify: onNotify,
  trained,
  initial,
  session,
  setSession,
  trace,
  baseline: referenceBaseline,
  computing,
  computationError,
  retryComparison,
}: WorkbenchProps & {
  initial: { session: NumericalSession; writable: boolean }
  session: NumericalSession
  setSession: Dispatch<SetStateAction<NumericalSession>>
  trace: NumericalTrace
  baseline?: NumericalTrace
  computing: boolean
  computationError?: string
  retryComparison: () => void
}) {
  const openWiki = useWiki()
  const [capturedBaseline, setCapturedBaseline] = useState<NumericalTrace>()
  const baseline = trained?.baseline ?? capturedBaseline ?? referenceBaseline
  const saveRef = useRef(onSave)
  saveRef.current = onSave
  const [cursor, setCursor] = useState(0),
    [playing, setPlaying] = useState(false),
    [delay, setDelay] = useState(700),
    [follow, setFollow] = useState(false)
  const [selection, setSelection] = useState<Selection>(null),
    [cell, setCell] = useState<OperandCell>(),
    [back, setBack] = useState<OperandCell[]>([])
  const [focus, setFocus] = useState<CameraFocus>(),
    [ready, setReady] = useState(false),
    [setup, setSetup] = useState(false),
    [stepFilter, setStepFilter] = useState(''),
    [lesson, setLesson] = useState('')
  const [mobilePane, setMobilePane] = useState<'flow' | 'data' | 'journey'>('flow'),
    [notesOpen, setNotesOpen] = useState(false),
    [notice, setNotice] = useState('')
  function notify(message: string) {
    setNotice(message)
    onNotify(message)
  }
  useEffect(() => {
    if (!notice) return
    const timer = setTimeout(() => setNotice(''), 5000)
    return () => clearTimeout(timer)
  }, [notice])
  const previousTrace = useRef(trace),
    initialCameraDone = useRef(false)
  const frame = trace.frames[Math.min(cursor, trace.frames.length - 1)]
  const semanticGraph = useMemo(
    () =>
      numericalGraph(
        reference,
        trace.settings,
        { direction: 'RIGHT', positions: {} },
        trained ? trace.config : undefined,
      ),
    [reference, trace.settings, trained ? trace.config : undefined],
  )
  const graph = useMemo(
    () => ({ ...semanticGraph, layout: session.layout }),
    [semanticGraph, session.layout],
  )
  const completed = useMemo(
    () => new Set(trace.frames.slice(0, cursor).flatMap((f) => f.nodeIds)),
    [trace, cursor],
  )
  const currentCell =
    cell && trace.tensors[cell.tensor] && trace.tensors[cell.tensor].createdAt <= cursor
      ? cell
      : firstCell(trace, frame)
  const tensor = currentCell ? trace.tensors[currentCell.tensor] : undefined
  const selectedNode =
    selection?.type === 'node' ? graph.nodes.find((n) => n.id === selection.id) : undefined
  const selectedEdge =
    selection?.type === 'edge' ? graph.edges.find((e) => e.id === selection.id) : undefined
  const edgeInspection = useMemo(
    () => (selectedEdge ? inspectConnection(trace, selectedEdge, cursor) : undefined),
    [trace, selectedEdge, cursor],
  )
  const binding =
    selection?.type === 'node'
      ? latestBinding(trace, selection.id, cursor)
      : selectedEdge
        ? edgeInspection?.record
        : frame
  const candidates = useMemo(
    () =>
      !selection && tensor?.role === 'parameter'
        ? [tensor]
        : (
            edgeInspection?.available ?? [
              ...new Set([...(binding?.outputs ?? []), ...(binding?.inputs ?? [])]),
            ]
          )
            .map((id) => trace.tensors[id])
            .filter((t) => t.createdAt <= cursor),
    [binding, trace, cursor, selectedEdge, selection, tensor],
  )
  const pending = Boolean(selection && !binding)
  const S = frame.queryPositions.at(-1)! + 1,
    P = residentCount(trace.config, Math.max(0, S - frame.queryLength))
  const symbols = useMemo(
    () => ({
      ...dimensions(trace.config),
      T: frame.queryLength,
      S,
      R: residentCount(trace.config, S),
      P,
      U: P + frame.queryLength,
    }),
    [trace.config, frame.queryLength, S, P],
  )
  const [exporting, setExporting] = useState(false)
  const exportJob = useRef<ReturnType<typeof prepareTraceExport> | undefined>(undefined)
  useEffect(
    () => () => {
      exportJob.current?.cancel()
      exportJob.current = undefined
    },
    [],
  )
  async function exportTrace() {
    if (exportJob.current) return
    setExporting(true)
    let job: ReturnType<typeof prepareTraceExport> | undefined
    try {
      job = prepareTraceExport(trace)
      exportJob.current = job
      const blob = await job.promise
      downloadTrace(trace, blob)
    } catch (error) {
      if (error instanceof Error && error.name !== 'AbortError') notify(error.message)
    } finally {
      if (exportJob.current === job) {
        exportJob.current = undefined
        setExporting(false)
      }
    }
  }
  const update = (change: Partial<NumericalSession>) => setSession((s) => ({ ...s, ...change }))
  const changeSettings = (change: Partial<NumericalSettings>) => {
    const next = { ...session.settings, ...change }
    update({
      settings: next,
      ...(change.attention && change.attention !== session.settings.attention
        ? {
            layouts: { ...session.layouts, [session.settings.attention]: session.layout },
            layout: session.layouts[next.attention] ?? {
              direction: session.layout.direction,
              positions: {},
            },
          }
        : {}),
    })
  }
  useEffect(() => {
    if (initial.writable) saveRef.current(session)
  }, [session])
  useEffect(() => {
    setCursor(0)
    setPlaying(false)
    setSelection(null)
    setCell(undefined)
    setBack([])
    setLesson('')
    if (previousTrace.current !== trace) {
      previousTrace.current = trace
      setFocus(undefined)
    }
  }, [trace])
  useEffect(() => {
    if (!ready) {
      setPlaying(false)
      return
    }
    if (initialCameraDone.current) return
    initialCameraDone.current = true
    const requested = focus
    if (requested) {
      const camera = requestAnimationFrame(() => setFocus({ ...requested, revision: Date.now() }))
      return () => cancelAnimationFrame(camera)
    }
  }, [ready, focus])
  useEffect(() => {
    if (!initialEntry) return
    jump(
      resolveNumericalExample(initialEntry, trace.config),
      0,
      undefined,
      'A numerical foundations example with its own small dimensions and seeded, untrained parameters. These are not activations from the architecture you were viewing.',
    )
    // This is the opening destination; subsequent experiment edits reset playback.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialEntry])
  useEffect(() => {
    if (
      (initialStudy?.kind !== 'numerical' && initialStudy?.kind !== 'trained') ||
      !initialStudy.node
    )
      return
    const restoreNode = initialStudy.node
    const at = trace.frames.findIndex(
      (f) => f.call === (initialStudy.call ?? 0) && f.nodeIds.includes(restoreNode),
    )
    if (at < 0) return
    jump(restoreNode, initialStudy.call ?? 0)
    if (initialStudy.tensor && trace.tensors[initialStudy.tensor]) {
      setCursor(Math.max(at, trace.tensors[initialStudy.tensor].createdAt))
      setCell({
        tensor: initialStudy.tensor,
        index: Math.min(
          initialStudy.index ?? 0,
          trace.tensors[initialStudy.tensor].data.length - 1,
        ),
      })
    }
  }, [initialStudy])
  useEffect(() => {
    if (!playing) return
    if (computing) {
      setPlaying(false)
      return
    }
    if (cursor >= trace.frames.length - 1) {
      setPlaying(false)
      return
    }
    const timer = setTimeout(() => seek(cursor + 1, false), delay)
    return () => clearTimeout(timer)
    // The playback cursor owns the timer. Inspection deliberately pauses it.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing, cursor, trace, delay, computing])
  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if (
        (event.target as Element)?.closest(
          'input,textarea,select,button,a,[contenteditable=true]',
        ) ||
        event.ctrlKey ||
        event.metaKey ||
        event.altKey
      )
        return
      if (event.key === 'ArrowRight') {
        event.preventDefault()
        seek(cursor + 1)
      } else if (event.key === 'ArrowLeft') {
        event.preventDefault()
        seek(cursor - 1)
      } else if (event.key === ' ' && ready) {
        event.preventDefault()
        setPlaying((p) => !p)
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
    // The current cursor and readiness own these shortcuts.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cursor, ready, follow, trace])
  function seek(index: number, pause = true) {
    const next = Math.max(0, Math.min(trace.frames.length - 1, index)),
      f = trace.frames[next]
    if (pause) setPlaying(false)
    setCursor(next)
    setSelection(null)
    setCell(firstCell(trace, f))
    setBack([])
    setLesson('')
    if (follow) setFocus({ id: f.nodeIds[0], revision: Date.now(), preserveZoom: true })
  }
  function inspect(next: OperandCell, push = true) {
    setPlaying(false)
    if (push && currentCell) setBack((s) => [...s.slice(-15), currentCell])
    setCell({
      ...next,
      index: Math.max(0, Math.min(next.index, trace.tensors[next.tensor].data.length - 1)),
    })
    setMobilePane('data')
  }
  function inspectParameter(next: OperandCell) {
    setSelection(null)
    setBack([])
    setNotesOpen(false)
    setLesson(
      trained
        ? 'A trained parameter from the verified checkpoint. Values stay fixed across decoder calls. Tied weights are identified explicitly.'
        : 'A seeded, untrained parameter tensor. Its values stay fixed across decoder calls; each layer has its own matrices.',
    )
    inspect(next, false)
    const owner = trace.frames.find((f) => f.inputs.includes(next.tensor))
    if (follow && owner)
      setFocus({ id: owner.nodeIds[0], revision: Date.now(), preserveZoom: true })
  }
  function select(selected: Selection) {
    setPlaying(false)
    setSelection(selected)
    setBack([])
    setLesson('')
    if (!selected) return
    const edge =
      selected.type === 'edge' ? graph.edges.find((e) => e.id === selected.id) : undefined
    const record =
      selected.type === 'node'
        ? latestBinding(trace, selected.id, cursor)
        : edge
          ? (latestBinding(trace, edge.source, cursor) ?? latestBinding(trace, edge.target, cursor))
          : undefined
    const payload = edge
      ? inspectConnection(trace, edge, cursor).preferred
      : (record?.bindings[selected.id]?.[0] ?? record?.outputs[0])
    setCell(payload && record ? firstCell(trace, record, payload) : undefined)
    setMobilePane('data')
  }
  function jump(node: string, call = frame.call, index?: number, description = '') {
    const at = trace.frames.findIndex((f) => f.call === call && f.nodeIds.includes(node))
    if (at < 0) {
      notify('This operation is outside the current output budget. Increase it in Run setup.')
      return
    }
    seek(at)
    setSelection({ type: 'node', id: node })
    setLesson(description)
    setFocus({ id: node, revision: Date.now() })
    const payload = trace.frames[at].bindings[node]?.[0] ?? trace.frames[at].outputs[0]
    if (payload)
      setCell({
        tensor: payload,
        index: Math.min(
          index ?? firstCell(trace, trace.frames[at], payload)?.index ?? 0,
          trace.tensors[payload].data.length - 1,
        ),
      })
    setMobilePane('data')
  }
  function bookmark() {
    if (session.bookmarks.length >= 24) {
      notify('This session has 24 bookmarks. Remove one before adding another.')
      return
    }
    const node = selection?.type === 'node' ? selection.id : frame.nodeIds[0],
      record = selection?.type === 'node' ? latestBinding(trace, node, cursor) : frame,
      call = record?.call ?? frame.call,
      id = `${call}/${node}`
    if (session.bookmarks.some((b) => b.id === id)) {
      notify('This operation is already bookmarked.')
      return
    }
    update({
      bookmarks: [
        ...session.bookmarks,
        { id, title: record?.title ?? frame.title, node, call, note: '' },
      ],
    })
    notify('Bookmarked this operation in the saved lab session.')
  }
  const filtered = trace.frames.filter(
    (f) =>
      f.call === frame.call &&
      `${f.title} ${f.nodeIds.join(' ')}`.toLowerCase().includes(stepFilter.toLowerCase()),
  )
  const sources = selectedNode?.sourceIds ?? ['attention', 'gqa']
  const tensorPlace = tensor?.id.match(/^call(\d+)\/l(\d+)\//),
    tensorRow =
      tensor && currentCell
        ? (unravel(currentCell.index, tensor.shape)[tensor.axes.indexOf('T')] ?? 0)
        : 0
  const wikiContext = {
    doc: reference,
    graph,
    selection:
      selection ?? (frame.nodeIds[0] ? { type: 'node' as const, id: frame.nodeIds[0] } : undefined),
    symbols,
    inspection:
      tensor && currentCell
        ? {
            name: tensor.name,
            index: currentCell.index,
            coordinate: unravel(currentCell.index, tensor.shape),
            axes: tensor.axes,
            value: Number.isFinite(tensor.data[currentCell.index])
              ? tensor.data[currentCell.index]
              : String(tensor.data[currentCell.index]),
            explanation: explainCell(trace, tensor, currentCell.index).explanation,
            tex: explainCell(trace, tensor, currentCell.index).tex,
          }
        : undefined,
    config: trace.config,
    mode: trained ? ('trained' as const) : ('untrained' as const),
    currentGraphOnly: true,
    onRestoreStudy,
    onExperiment: (entry: string) => jump(resolveNumericalExample(entry, trace.config)),
    onNavigate: (target: import('../../help/wiki/catalog').WikiTarget) => {
      if (target.type === 'graph') return
      select({ type: target.type, id: target.id })
      const node =
        target.type === 'node' ? target.id : graph.edges.find((e) => e.id === target.id)?.target
      if (node) setFocus({ id: node, revision: Date.now() })
      setMobilePane('data')
    },
  }
  useWikiFollow(wikiContext)
  const recordRecipe: StudyRecipe =
    trained && trace.checkpoint && trace.tokenization
      ? {
          kind: 'trained',
          checkpoint: trace.checkpoint.id,
          node: selection?.type === 'node' ? selection.id : frame.nodeIds[0],
          call: frame.call,
          tensor: currentCell?.tensor,
          index: currentCell?.index,
          revision: trace.checkpoint.revision,
          ...(baseline?.tokenization
            ? {
                comparison: {
                  settings: {
                    prompt: baseline.tokenization.prompt,
                    generate: baseline.settings.generate,
                    seed: baseline.settings.seed,
                    sampling: baseline.settings.sampling,
                    temperature: baseline.settings.temperature,
                  },
                  ...(baseline.intervention
                    ? {
                        intervention: {
                          mode: baseline.intervention.mode,
                          layer: baseline.intervention.layer,
                          head: baseline.intervention.head,
                          donorPrompt: baseline.intervention.donorPrompt,
                        },
                      }
                    : {}),
                },
              }
            : {}),
          settings: {
            prompt: trace.tokenization.prompt,
            generate: trace.settings.generate,
            seed: trace.settings.seed,
            sampling: trace.settings.sampling,
            temperature: trace.settings.temperature,
          },
          ...(trace.intervention
            ? {
                intervention: {
                  mode: trace.intervention.mode,
                  layer: trace.intervention.layer,
                  head: trace.intervention.head,
                  donorPrompt: trace.intervention.donorPrompt,
                },
              }
            : {}),
        }
      : {
          kind: 'numerical',
          settings: trace.settings,
          ...(baseline ? { baseline: baseline.settings } : {}),
          node: selection?.type === 'node' ? selection.id : frame.nodeIds[0],
          call: frame.call,
          tensor: currentCell?.tensor,
          index: currentCell?.index,
        }
  return (
    <Dialog
      title="The mathematics, alive."
      className={`numerical-modal${trained ? ' trained-modal' : ''}`}
      onClose={onClose}
    >
      <div className="numerical-topline">
        <span className="numerical-badge">
          <FlaskConical size={12} />{' '}
          {trained ? 'LOCAL · TRAINED · Stories260K' : 'NUMERICAL · UNTRAINED'}
        </span>
        <span>
          {trace.config.nLayers} layers · D {trace.config.dModel} · Hq {trace.config.nQueryHeads} ·
          Hkv {trace.config.nKvHeads} · Dh {trace.config.headDim} · F {trace.config.ffnDim} · V{' '}
          {trace.config.vocabSize}
        </span>
        <button
          aria-label={exporting ? 'Preparing export…' : 'Export trace'}
          disabled={exporting || computing}
          onClick={() => void exportTrace()}
        >
          <Download size={13} />{' '}
          <span className="compact-action-label">
            {exporting ? 'Preparing export…' : 'Export trace'}
          </span>
        </button>
        <TooltipSettingsButton />
        <button
          aria-label="Explain this"
          onClick={() => {
            setPlaying(false)
            openWiki(wikiContext)
          }}
        >
          <BookOpen size={14} /> <span className="compact-action-label">Explain this</span>
        </button>
        <button aria-label="Toggle numerical theme" onClick={onTheme}>
          {doc.ui.theme === 'dark' ? <Sun size={14} /> : <Moon size={14} />}
        </button>
        <button
          aria-label="Open numerical run setup"
          aria-pressed={setup}
          onClick={() => {
            setSetup(!setup)
            setMobilePane('journey')
          }}
        >
          <Settings2 size={14} /> Run setup
        </button>
      </div>
      {(computing || computationError) && (
        <div className="numerical-worker-status" role="status" data-testid="numerical-computing">
          {computationError ??
            'Computing the updated run in background workers. The microscope still shows the previous completed run.'}
          {computationError && (
            <button className="button" onClick={retryComparison}>
              Retry baseline
            </button>
          )}
        </div>
      )}
      {!initial.writable && (
        <p className="numerical-version-note">
          An unsupported saved lab session is preserved. This temporary session will not overwrite
          it.
        </p>
      )}
      {notice && (
        <div className="numerical-notice" role="status">
          {notice}
          <button aria-label="Dismiss laboratory notification" onClick={() => setNotice('')}>
            ×
          </button>
        </div>
      )}
      <div className="numerical-mobile-tabs">
        <button aria-pressed={mobilePane === 'journey'} onClick={() => setMobilePane('journey')}>
          Journey
        </button>
        <button aria-pressed={mobilePane === 'flow'} onClick={() => setMobilePane('flow')}>
          Flow
        </button>
        <button aria-pressed={mobilePane === 'data'} onClick={() => setMobilePane('data')}>
          Inspect
        </button>
      </div>
      <div className={`numerical-body pane-${mobilePane}`}>
        <aside className="numerical-journey" aria-label="Numerical learning journey">
          {setup && trained ? (
            <div className="numerical-setup">
              {trained.controls}
              <button className="button" disabled={computing} onClick={() => setSetup(false)}>
                Explore the run <ArrowRight size={13} />
              </button>
            </div>
          ) : setup ? (
            <div className="numerical-setup">
              <h3>Make a small experiment.</h3>
              <p>
                A separate foundations reference profile, with seeded untrained weights. It does not
                execute the loaded architecture. All values are calculated locally.
              </p>
              <label>
                Parameter seed
                <input
                  aria-label="Numerical parameter seed"
                  type="number"
                  min={0}
                  max={2147483647}
                  value={session.settings.seed}
                  onChange={(e) => {
                    const seed = Number(e.target.value)
                    if (Number.isInteger(seed) && seed >= 0 && seed <= 2147483647)
                      changeSettings({ seed })
                  }}
                />
              </label>
              <label>
                Output tokens
                <select
                  aria-label="Numerical output budget"
                  value={session.settings.generate}
                  onChange={(e) => changeSettings({ generate: Number(e.target.value) })}
                >
                  {[1, 2, 3, 4, 8].map((n) => (
                    <option key={n}>{n}</option>
                  ))}
                </select>
              </label>
              <label>
                Head sharing
                <select
                  aria-label="Numerical head sharing"
                  value={session.settings.attention}
                  onChange={(e) =>
                    changeSettings({ attention: e.target.value as NumericalSettings['attention'] })
                  }
                >
                  {['MHA', 'GQA', 'MQA'].map((v) => (
                    <option key={v}>{v}</option>
                  ))}
                </select>
              </label>
              <label>
                Selection
                <select
                  aria-label="Numerical token selection"
                  value={session.settings.sampling}
                  onChange={(e) =>
                    changeSettings({ sampling: e.target.value as NumericalSettings['sampling'] })
                  }
                >
                  <option value="greedy">Greedy argmax</option>
                  <option value="sample">Seeded sampling</option>
                </select>
              </label>
              <label>
                Temperature <b>{session.settings.temperature}</b>
                <input
                  type="range"
                  aria-label="Numerical temperature"
                  min={0.1}
                  max={3}
                  step={0.1}
                  value={session.settings.temperature}
                  onChange={(e) => changeSettings({ temperature: Number(e.target.value) })}
                />
              </label>
              <CachePolicyControls
                config={{ ...trace.config, kvCache: session.settings.kvCache }}
                onChange={(kvCache) => changeSettings({ kvCache })}
              />
              <button className="button" disabled={computing} onClick={() => setSetup(false)}>
                Explore the run <ArrowRight size={13} />
              </button>
              <ParameterLibrary trace={trace} onInspect={inspectParameter} />
            </div>
          ) : (
            <>
              <div className="journey-intro">
                <span className="eyebrow">A WALK THROUGH THE MODEL</span>
                <p>Follow a value. Open an operation. Ask why.</p>
              </div>
              <label className="numerical-call">
                Decoder call
                <select
                  aria-label="Numerical decoder call"
                  value={frame.call}
                  onChange={(e) =>
                    seek(trace.frames.findIndex((f) => f.call === Number(e.target.value)))
                  }
                >
                  {Array.from({ length: new Set(trace.frames.map((f) => f.call)).size }, (_, i) => (
                    <option key={i} value={i}>
                      {i === 0 ? 'Prefill · input in parallel' : `Decode ${i} · one new query`}
                    </option>
                  ))}
                </select>
              </label>
              <nav className="numerical-chapters">
                {(trained
                  ? [
                      ['Input', 'tokens.lookup'],
                      ...Array.from({ length: trace.config.nLayers }, (_, l) => [
                        [`L${l} · attention`, `layer-${l}.norm1`],
                        [`L${l} · feed-forward`, `layer-${l}.norm2`],
                      ]).flat(),
                      ['Vocabulary', 'output.lmhead'],
                      ['Choose & loop', 'output.sample'],
                    ]
                  : chapters
                ).map(([title, node]) => (
                  <button key={node} disabled={computing} onClick={() => jump(node)}>
                    <span />
                    {title}
                    <ChevronRight size={12} />
                  </button>
                ))}
              </nav>
              <details className="numerical-questions" open>
                <summary>Explore a question</summary>
                {questions.map((q) => (
                  <button
                    key={q.node}
                    data-tooltip-title={q.title}
                    data-tooltip-detail={
                      trained && q.checkpointDescription ? q.checkpointDescription : q.description
                    }
                    disabled={
                      computing ||
                      (q.multiple && trace.settings.inputIds.length < 2) ||
                      q.call >= new Set(trace.frames.map((f) => f.call)).size
                    }
                    onClick={() =>
                      jump(
                        q.node,
                        q.call,
                        q.index,
                        trained && q.checkpointDescription
                          ? q.checkpointDescription
                          : q.description,
                      )
                    }
                  >
                    {q.title}
                    <ArrowRight size={11} />
                  </button>
                ))}
              </details>
              <ParameterLibrary trace={trace} onInspect={inspectParameter} />
              <div className="numerical-step-heading">
                <span className="eyebrow">EVERY OPERATION</span>
                <small>{filtered.length}</small>
              </div>
              <input
                className="numerical-step-search"
                aria-label="Find a numerical operation"
                placeholder="Find an operation…"
                value={stepFilter}
                onChange={(e) => setStepFilter(e.target.value)}
              />
              <div className="numerical-steps">
                {filtered.map((f) => (
                  <button
                    key={f.id}
                    className={f.id === cursor ? 'active' : f.id < cursor ? 'visited' : ''}
                    aria-current={f.id === cursor ? 'step' : undefined}
                    onClick={() => seek(f.id)}
                  >
                    <span>{f.id < cursor ? <Check size={10} /> : f.id + 1}</span>
                    <span>
                      {f.title}
                      <small>
                        {f.layer !== undefined ? `L${f.layer} · ` : ''}
                        {f.phase}
                      </small>
                    </span>
                  </button>
                ))}
              </div>
            </>
          )}
        </aside>
        <main className="numerical-stage">
          {trained && (
            <div className="trained-text" aria-label="Trained model text">
              <div>
                <span className="eyebrow">PROMPT · CHECKPOINT TOKENIZER</span>
                <p>{trace.tokenization?.prompt || <em>Beginning of sequence</em>}</p>
              </div>
              <div>
                <span className="eyebrow">COMPLETION · CONFIRMED AT THIS STEP</span>
                <p>
                  {trained.tokenizer
                    .decode(frame.confirmed.map((p) => Number(p.id.slice(1))))
                    .slice(trained.tokenizer.decode(trace.settings.inputIds).length) || (
                    <em>Waiting for generated tokens…</em>
                  )}
                </p>
              </div>
            </div>
          )}
          <div className="numerical-stream">
            <div>
              <span className="eyebrow">INPUT IDS</span>
              <div>
                {trace.settings.inputIds.map((id, i) => (
                  <label key={i} data-tooltip-detail={trained?.tokenizer.piece(id)}>
                    <small>p{i}</small>
                    <input
                      aria-label={`Numerical input token ${i}`}
                      type="number"
                      min={0}
                      max={trace.config.vocabSize - 1}
                      value={id}
                      readOnly={Boolean(trained)}
                      onChange={(e) => {
                        const value = Number(e.target.value)
                        if (Number.isInteger(value) && value >= 0 && value < 256)
                          changeSettings({
                            inputIds: session.settings.inputIds.map((t, j) =>
                              j === i ? value : t,
                            ),
                          })
                      }}
                    />
                    {trained && (
                      <span className="trained-token-piece">
                        {trained.tokenizer.piece(id).replaceAll(' ', '␠').replaceAll('\n', '↵')}
                      </span>
                    )}
                  </label>
                ))}
                {!trained && (
                  <>
                    <button
                      aria-label="Add numerical input token"
                      disabled={session.settings.inputIds.length === 8}
                      onClick={() =>
                        changeSettings({ inputIds: [...session.settings.inputIds, 11] })
                      }
                    >
                      +
                    </button>
                    <button
                      aria-label="Remove numerical input token"
                      disabled={session.settings.inputIds.length === 1}
                      onClick={() =>
                        changeSettings({ inputIds: session.settings.inputIds.slice(0, -1) })
                      }
                    >
                      −
                    </button>
                  </>
                )}
              </div>
            </div>
            <ArrowRight size={17} />
            <div>
              <span className="eyebrow">CONFIRMED OUTPUT</span>
              <div>
                {frame.confirmed.slice(trace.settings.inputIds.length).map((p, i) => (
                  <span
                    className="numerical-output-token"
                    key={i}
                    data-tooltip-detail={trained?.tokenizer.piece(Number(p.id.slice(1)))}
                  >
                    {p.id}
                    {trained && (
                      <span className="trained-token-piece">
                        {trained.tokenizer
                          .piece(Number(p.id.slice(1)))
                          .replaceAll(' ', '␠')
                          .replaceAll('\n', '↵')}
                      </span>
                    )}
                    <small>out {i}</small>
                  </span>
                ))}
                {!frame.generated && (
                  <span className="numeric-muted">Waiting for vocabulary selection</span>
                )}
              </div>
            </div>
          </div>
          <div className="numerical-step-card">
            <div>
              <span className="execution-phase">
                {frame.phase} · call {frame.call}
              </span>
              <span>
                {cursor + 1} / {trace.frames.length}
              </span>
            </div>
            <h3>{frame.title}</h3>
            <p>{frame.description}</p>
            <div className="numerical-call-shapes">
              <code>T {symbols.T}</code>
              <code>working U {symbols.U}</code>
              <code>target R {symbols.R}</code>
              <button onClick={bookmark}>
                <Bookmark size={12} /> Bookmark
              </button>
            </div>
          </div>
          <div className="numerical-cache-snapshot" aria-label="Current numerical cache state">
            <span>STORED K/V</span>
            {frame.cachePositions.map((positions, l) => (
              <button
                key={l}
                onClick={() => jump(`layer-${l}.attn.cachek`)}
                data-tooltip-detail={`Inspect layer ${l} cache; ${positions.length} retained positions`}
              >
                <b>L{l}</b>
                <span>
                  {Array.from({ length: S }, (_, p) => (
                    <i
                      key={p}
                      data-stored={positions.includes(p)}
                      data-tooltip-detail={`Position ${p}: ${positions.includes(p) ? 'stored' : 'absent'}`}
                    >
                      {p}
                    </i>
                  ))}
                </span>
                <small>
                  {positions.length} positions × {trace.config.nKvHeads} KV heads
                  {' · '}
                  {formatBytes(
                    2 * positions.length * trace.config.nKvHeads * trace.config.headDim * 4,
                  )}
                </small>
              </button>
            ))}
          </div>
          <div className="numerical-canvas" inert={computing}>
            <GraphCanvas
              graph={graph}
              symbols={symbols}
              selection={selection}
              showShapes
              symbolic={false}
              theme={doc.ui.theme}
              onSelect={select}
              onOpen={() => {}}
              onReady={setReady}
              onLayout={(positions, direction, reset) =>
                setSession((s) =>
                  s.settings.attention !== trace.settings.attention
                    ? s
                    : {
                        ...s,
                        layout: {
                          ...s.layout,
                          positions,
                          direction: direction ?? s.layout.direction,
                          ...(reset ? { viewport: undefined } : {}),
                        },
                      },
                )
              }
              onRestore={(snapshot) =>
                setSession((s) =>
                  s.settings.attention !== trace.settings.attention
                    ? s
                    : { ...s, layout: snapshot.layout },
                )
              }
              onViewport={(viewport) =>
                setSession((s) =>
                  s.settings.attention !== trace.settings.attention
                    ? s
                    : { ...s, layout: { ...s.layout, viewport } },
                )
              }
              notify={notify}
              execution={{ active: frame.nodeIds, completed, revision: cursor, playing }}
              focus={focus}
            />
          </div>
          <div className="numerical-playback">
            <div className="numerical-playback-buttons">
              <button aria-label="Reset numerical playback" onClick={() => seek(0)}>
                <RotateCcw size={15} />
              </button>
              <button
                aria-label="Previous numerical operation"
                disabled={computing || cursor === 0}
                onClick={() => seek(cursor - 1)}
              >
                <ChevronLeft size={18} />
              </button>
              <button
                className="button numerical-play"
                disabled={computing || !ready || cursor === trace.frames.length - 1}
                onClick={() => setPlaying(!playing)}
              >
                {playing ? <Pause size={15} /> : <Play size={15} />}{' '}
                {playing ? 'Pause' : ready ? 'Play' : 'Preparing flow…'}
              </button>
              <button
                aria-label="Next numerical operation"
                disabled={computing || cursor === trace.frames.length - 1}
                onClick={() => seek(cursor + 1)}
              >
                <ChevronRight size={18} />
              </button>
              <button
                aria-label="Finish numerical playback"
                onClick={() => seek(trace.frames.length - 1)}
              >
                <ArrowRight size={16} />
              </button>
              <select
                aria-label="Numerical playback speed"
                value={delay}
                onChange={(e) => setDelay(Number(e.target.value))}
              >
                <option value={1200}>Slow</option>
                <option value={700}>Normal</option>
                <option value={250}>Fast</option>
              </select>
              <label>
                <input
                  type="checkbox"
                  checked={follow}
                  onChange={(e) => setFollow(e.target.checked)}
                />{' '}
                Follow flow
              </label>
            </div>
            <input
              aria-label="Numerical execution timeline"
              type="range"
              min={0}
              max={trace.frames.length - 1}
              value={cursor}
              onChange={(e) => seek(Number(e.target.value))}
            />
            <div>
              <small>
                Pan and zoom freely; Follow flow keeps your zoom. Click a node or connection to
                inspect its tensors.
              </small>
              {
                <button
                  onClick={() => {
                    if (trained) setCapturedBaseline(trace)
                    else update({ pinned: structuredClone(trace.settings) })
                    notify('Pinned this run as the comparison baseline.')
                  }}
                >
                  Pin baseline
                </button>
              }
              {baseline && (
                <button
                  onClick={() => {
                    setCapturedBaseline(undefined)
                    trained?.clearBaseline?.()
                    update({ pinned: undefined })
                  }}
                >
                  Clear baseline
                </button>
              )}
            </div>
          </div>
        </main>
        <aside className="numerical-inspector" aria-label="Numerical tensor inspector">
          {baseline && (
            <RunComparison
              trace={trace}
              baseline={baseline}
              piece={trained ? (id) => trained.tokenizer.piece(id) : undefined}
              onInspect={(at, tensor, index) => {
                seek(at)
                setCell({ tensor, index })
                setFocus({ id: trace.frames[at].nodeIds[0], revision: Date.now() })
              }}
            />
          )}
          <div className="numeric-inspector-heading">
            <span className="eyebrow">TENSOR MICROSCOPE</span>
            <button
              aria-label="Open numerical lab notes"
              aria-pressed={notesOpen}
              onClick={() => setNotesOpen(!notesOpen)}
            >
              <Bookmark size={13} /> Notes
            </button>
          </div>
          {notesOpen ? (
            <div className="numerical-notes">
              <h3>Your experiment notebook.</h3>
              <StudyRecordForm
                recipe={recordRecipe}
                title={`${trained ? 'Trained' : 'Untrained'} · ${frame.title}`}
                evidence={{
                  scope: trained
                    ? 'Verified trained checkpoint · captured run'
                    : 'Seeded untrained reference · captured run',
                  tensor: tensor?.id,
                  shape: tensor?.shape,
                  axes: tensor?.axes,
                  start: currentCell?.index ?? 0,
                  values: tensor
                    ? Array.from(
                        tensor.data.slice(currentCell?.index ?? 0, (currentCell?.index ?? 0) + 64),
                        (v) => (Number.isFinite(v) ? v : String(v)),
                      )
                    : undefined,
                  baseline:
                    tensor &&
                    baseline?.tensors[tensor.id] &&
                    !tensorComparisonIssue(tensor, baseline.tensors[tensor.id])
                      ? Array.from(
                          baseline.tensors[tensor.id].data.slice(
                            currentCell?.index ?? 0,
                            (currentCell?.index ?? 0) + 64,
                          ),
                          (v) => (Number.isFinite(v) ? v : String(v)),
                        )
                      : undefined,
                  summary: `${frame.title}, call ${frame.call}. Selected tensor slice contains at most 64 consecutive cells, not a whole activation capture. ${trace.intervention ? JSON.stringify(trace.intervention) : 'No internal intervention.'} Output IDs: ${trace.outputIds.join(', ')}. ${trace.arithmetic}`,
                }}
              />
              {onRestoreStudy && <StudyRecordLibrary onRestore={onRestoreStudy} />}
              <textarea
                aria-label="Numerical laboratory notes"
                maxLength={24000}
                value={session.notes}
                placeholder="What surprised you? Which values changed?"
                onChange={(e) => update({ notes: e.target.value })}
              />
              <h4>Saved operations</h4>
              {session.bookmarks.length === 0 && (
                <p>Bookmark an operation to return to it across sessions.</p>
              )}
              {session.bookmarks.map((b) => (
                <div className="numerical-bookmark" key={b.id}>
                  <button
                    onClick={() => {
                      jump(b.node, b.call)
                      setNotesOpen(false)
                    }}
                  >
                    {b.title}
                    <small>
                      Call {b.call} · {b.node}
                    </small>
                  </button>
                  <textarea
                    aria-label={`Note for ${b.title}`}
                    maxLength={2000}
                    value={b.note}
                    onChange={(e) =>
                      update({
                        bookmarks: session.bookmarks.map((old) =>
                          old.id === b.id ? { ...old, note: e.target.value } : old,
                        ),
                      })
                    }
                  />
                  <button
                    onClick={() =>
                      update({ bookmarks: session.bookmarks.filter((old) => old.id !== b.id) })
                    }
                  >
                    Remove bookmark
                  </button>
                </div>
              ))}
            </div>
          ) : (
            <>
              {back.length > 0 && (
                <button
                  className="numeric-provenance-back"
                  onClick={() => {
                    const previous = back.at(-1)!
                    setBack((s) => s.slice(0, -1))
                    setCell(previous)
                  }}
                >
                  <ArrowLeft size={13} /> Back through the calculation
                </button>
              )}
              <h2>
                {selectedNode?.title ??
                  selectedEdge?.label ??
                  tensor?.name ??
                  'Choose an operation.'}
              </h2>
              {binding && selection && (
                <p className="numeric-record-label">
                  Recorded call {binding.call} · {binding.phase}
                  {binding.call !== frame.call ? ' · previous decoder call' : ''}
                </p>
              )}
              {lesson && <p className="numerical-lesson">{lesson}</p>}
              {(selectedNode || selectedEdge) && (
                <details className="numeric-architecture">
                  <summary>
                    {selectedEdge ? 'About this connection' : 'Architecture & primary sources'}
                  </summary>
                  <p>{selectedNode?.description ?? selectedEdge?.description}</p>
                  {(selectedNode?.equation ?? selectedEdge?.equation) && (
                    <MathBlock tex={(selectedNode?.equation ?? selectedEdge?.equation)!} />
                  )}
                  <p>
                    {selectedEdge
                      ? 'The recorded source outputs and available destination operands are shown below. Static edge shapes can describe an operation result; tensor axes show the recorded payload explicitly.'
                      : ''}
                  </p>
                  {reference.sources
                    .filter((s) => sources.includes(s.id))
                    .map((s) => (
                      <a key={s.id} href={s.url} target="_blank" rel="noreferrer">
                        {s.title} ↗
                      </a>
                    ))}
                </details>
              )}
              {pending ? (
                <div className="numeric-empty">
                  <h3>This operation hasn’t run yet.</h3>
                  <p>The timeline only exposes values computed by this step.</p>
                  <button
                    className="button"
                    onClick={() =>
                      jump(selection!.type === 'node' ? selection!.id : selectedEdge!.target)
                    }
                  >
                    Run to this operation <ArrowRight size={12} />
                  </button>
                </div>
              ) : (
                <>
                  {!!candidates.length && (
                    <div className="numeric-tensor-picker">
                      {candidates.map((t) => (
                        <button
                          key={t.id}
                          aria-pressed={tensor?.id === t.id}
                          onClick={() => inspect({ tensor: t.id, index: 0 }, false)}
                        >
                          <small>
                            {t.role === 'parameter'
                              ? 'WEIGHT'
                              : binding?.outputs.includes(t.id)
                                ? 'OUTPUT'
                                : 'INPUT'}
                          </small>
                          <span>{t.name}</span>
                          <code>{t.shape.join('×')}</code>
                        </button>
                      ))}
                    </div>
                  )}
                  {tensor && currentCell ? (
                    <>
                      <div className="numeric-current-tensor">
                        <span>{tensor.name}</span>
                        <small>
                          {tensor.role} · {tensor.recipe.op}
                        </small>
                      </div>
                      <TensorView
                        key={`tensor/${tensor.id}`}
                        tensor={tensor}
                        index={currentCell.index}
                        onCell={(index) => {
                          setPlaying(false)
                          setCell({ tensor: tensor.id, index })
                        }}
                        comparison={baseline?.tensors[tensor.id]}
                      />
                      {tensor.recipe.op === 'softmax' && tensor.axes.at(-1) === 'V' && (
                        <ProbabilityStrip
                          tensor={tensor}
                          index={currentCell.index}
                          onSelect={(index) => setCell({ tensor: tensor.id, index })}
                          piece={trained?.tokenizer.piece.bind(trained.tokenizer)}
                        />
                      )}
                      <CellCalculation
                        key={`proof/${tensor.id}`}
                        trace={trace}
                        tensor={tensor}
                        index={currentCell.index}
                        onOperand={inspect}
                        sources={reference.sources}
                      />
                      {tensorPlace && (
                        <HeadObservatory
                          trace={trace}
                          cursor={cursor}
                          call={Number(tensorPlace[1])}
                          layer={Number(tensorPlace[2])}
                          row={tensorRow}
                          onInspect={inspect}
                        />
                      )}
                      <p className="numeric-arithmetic">{trace.arithmetic}</p>
                    </>
                  ) : (
                    <p className="numeric-empty">
                      Choose an operation or tensor to inspect its actual values and arithmetic.
                    </p>
                  )}
                </>
              )}
            </>
          )}
        </aside>
      </div>
    </Dialog>
  )
}

function ProbabilityStrip({
  tensor,
  index,
  onSelect,
  piece,
}: {
  tensor: NumericTensor
  index: number
  onSelect: (index: number) => void
  piece?: (id: number) => string
}) {
  const width = tensor.shape.at(-1)!,
    row = Math.floor(index / width),
    values = Array.from(tensor.data.subarray(row * width, (row + 1) * width)),
    top = values
      .map((p, id) => ({ p, id }))
      .sort((a, b) => b.p - a.p)
      .slice(0, 8),
    max = top[0].p
  return (
    <section className="numeric-probabilities">
      <span className="eyebrow">VOCABULARY · TOP 8 OF {width}</span>
      {top.map((t) => (
        <button key={t.id} onClick={() => onSelect(row * width + t.id)}>
          <span data-tooltip-detail={piece?.(t.id)}>
            #{t.id}
            {piece ? ` · ${piece(t.id)}` : ''}
          </span>
          <i style={{ width: `${(t.p / max) * 100}%` }} />
          <b>{valueLabel(t.p * 100, 3)}%</b>
        </button>
      ))}
    </section>
  )
}
