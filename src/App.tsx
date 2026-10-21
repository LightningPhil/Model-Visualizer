import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  ArrowDownToLine,
  ArrowRight,
  ArrowUpFromLine,
  ArrowUpRight,
  BookOpen,
  Box,
  Check,
  ChevronDown,
  ChevronRight,
  Command,
  Cpu,
  Database,
  FileJson,
  FlaskConical,
  CodeXml as Github,
  HelpCircle,
  Layers3,
  Menu,
  Moon,
  PanelRightClose,
  PanelRightOpen,
  Search,
  SlidersHorizontal,
  Sun,
  Network,
  Play,
  X,
} from 'lucide-react'
import starter from '../examples/transformer-foundations.model.json'
import {
  cacheBytes,
  dimensions,
  formatBytes,
  parseDocument,
  type ModelConfig,
  type ModelDocument,
  type ModelGraph,
} from './model/schema'
import { demoTokenizer } from './model/tokenizer'
import { exportDocument, importDocument, loadWorkspace, saveWorkspace } from './model/storage'
import { GraphCanvas, type Selection } from './components/GraphCanvas'
import { Inspector } from './components/Inspector'
import { Logo, graphIcons } from './components/Icons'
import { AttentionLab, CacheLab, MemoryLab, MtpLab, TokenizerLab } from './components/Labs'
import { Dialog } from './components/Dialog'
import { addWholeModelView, instanceOf, isWholeModel } from './model/whole-model'
import { useSimulation } from './components/useSimulation'
import { SimulationPanel, SimulationStrip } from './components/SimulationPanel'
import { configurationLayouts } from './model/configuration-layouts'
import { upgradeCachePolicies } from './model/policy-upgrade'
import { upgradeTeachingProvenance } from './model/teaching-provenance'
import { cachePolicyLabel, residentCount, kvHeadFor } from './model/attention'
import { CachePolicyControls } from './components/CachePolicyControls'
import { TeachingModelControls } from './components/TeachingModelControls'
import { TooltipSettingsButton } from './components/help/TooltipProvider'
import { updateTeachingConfig } from './model/teaching-config'
import { createGraphCompiler } from './model/graph-compiler'
import { NUMERICAL_SESSION_KEY } from './model/numerics/identity'
import type { NumericalSession } from './model/numerics/session'
import { CHECKPOINT_SESSION_KEY } from './model/checkpoints/identity'
import type { CheckpointSession } from './model/checkpoints/session'

const LocalModelWorkbench = lazy(() =>
  import('./components/numerical/LocalModelWorkbench').then((module) => ({
    default: module.LocalModelWorkbench,
  })),
)

const NumericalWorkbench = lazy(() =>
  import('./components/numerical/NumericalWorkbench').then((module) => ({
    default: module.NumericalWorkbench,
  })),
)

const referenceDocument = parseDocument(starter)
const upgrade = (doc: ModelDocument) =>
  addWholeModelView(
    upgradeTeachingProvenance(upgradeCachePolicies(doc, referenceDocument), referenceDocument),
    referenceDocument,
  )

const labels: Record<string, string> = {
  overview: 'Architecture overview',
  tokenization: 'Tokenization',
  block: 'Transformer block',
  attention: 'Attention & heads',
  cache: 'KV cache',
  feedforward: 'Feed-forward network',
  output: 'Output & decoding',
  memory: 'N-gram memory',
  mtp: 'Multi-token prediction',
  generation: 'Generation & verification',
  'whole-model': 'Whole model',
  'attention-heads': 'Split attention heads',
  'whole-model-heads': 'Whole model · split heads',
}
const labGraphs = new Set([
  'tokenization',
  'attention',
  'attention-heads',
  'cache',
  'memory',
  'mtp',
])

export default function App() {
  const [doc, setDoc] = useState<ModelDocument>(() => upgrade(loadWorkspace() ?? referenceDocument))
  const [simulating, setSimulating] = useState(false)
  const [diagramReady, setDiagramReady] = useState(false)
  const [focus, setFocus] = useState<{ id: string; revision: number } | undefined>()
  const [selection, setSelection] = useState<Selection>({ type: 'node', id: 'decoder' })
  const [showShapes, setShowShapes] = useState(true)
  const [symbolic, setSymbolic] = useState(false)
  const [decode, setDecode] = useState(false)
  const [selectedToken, setSelectedToken] = useState(4)
  const [inspectorOpen, setInspectorOpen] = useState(true)
  const [mobileNav, setMobileNav] = useState(false)
  const [labOpen, setLabOpen] = useState(true)
  const [modal, setModal] = useState<
    'guide' | 'search' | 'config' | 'numerical' | 'local-model' | null
  >(null)
  const [numericalEntry, setNumericalEntry] = useState<string>()
  const [guideTab, setGuideTab] = useState<'guide' | 'sources' | 'format'>('guide')
  const [query, setQuery] = useState('')
  const [toast, setToast] = useState('')
  const [saved, setSaved] = useState(true)
  const [documentRevision, setDocumentRevision] = useState(0)
  const documentEpoch = useRef(0)
  const [selectedLayer, setSelectedLayer] = useState(0)
  const [selectedHead, setSelectedHead] = useState(0)
  const fileInput = useRef<HTMLInputElement>(null)
  const saveNumericalSession = useCallback(
    (session: NumericalSession) =>
      setDoc((current) => ({
        ...current,
        extensions: { ...current.extensions, [NUMERICAL_SESSION_KEY]: session },
      })),
    [],
  )
  const toastTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)
  const saveCheckpointSession = useCallback(
    (session: CheckpointSession) =>
      setDoc((current) => ({
        ...current,
        extensions: { ...current.extensions, [CHECKPOINT_SESSION_KEY]: session },
      })),
    [],
  )

  const authoredGraph = doc.graphs.find((g) => g.id === doc.ui.activeGraph)!
  const compileGraph = useMemo(createGraphCompiler, [])
  const graph = useMemo(
    () => compileGraph(doc, authoredGraph),
    [doc.graphs, doc.config, authoredGraph, compileGraph],
  )
  const wholeModel = doc.graphs.find(isWholeModel)
  const whole = isWholeModel(authoredGraph)
  const splitHeads = graph.id === 'whole-model-heads' || graph.id === 'attention-heads'
  const sim = useSimulation(doc.config, simulating && whole, splitHeads)
  const simulationFrame = sim.frame
  const encoded = useMemo(() => demoTokenizer.encode(doc.ui.prompt), [doc.ui.prompt])
  const tokenizerSupported =
    doc.tokenizer.adapter === demoTokenizer.id &&
    doc.config.vocabSize >= demoTokenizer.vocabularySize
  const tokenCount = tokenizerSupported ? Math.max(1, encoded.tokens.length) : 12
  const symbols = useMemo(
    () =>
      simulationFrame
        ? {
            ...dimensions(doc.config),
            T: simulationFrame.queryLength,
            S: Math.max(simulationFrame.contextLength, ...simulationFrame.cacheLengths),
            R: Math.max(...simulationFrame.cachePositions.map((p) => p.length)),
            P: residentCount(
              doc.config,
              Math.max(0, Math.max(...simulationFrame.cacheLengths) - simulationFrame.queryLength),
            ),
            U:
              residentCount(
                doc.config,
                Math.max(
                  0,
                  Math.max(...simulationFrame.cacheLengths) - simulationFrame.queryLength,
                ),
              ) + simulationFrame.queryLength,
          }
        : dimensions(doc.config, tokenCount, decode),
    [doc.config, tokenCount, decode, simulationFrame],
  )
  const currentToken = Math.min(selectedToken, Math.max(0, encoded.tokens.length - 1))
  const hasLab = labGraphs.has(graph.id)
  const showLab = hasLab && labOpen && tokenizerSupported

  const notify = useCallback((message: string) => {
    setToast(message)
    clearTimeout(toastTimer.current)
    toastTimer.current = setTimeout(() => setToast(''), 4500)
  }, [])

  useEffect(() => {
    document.documentElement.dataset.theme = doc.ui.theme
    document
      .querySelector('meta[name="theme-color"]')
      ?.setAttribute('content', doc.ui.theme === 'dark' ? '#090e19' : '#f5f7fc')
    const timer = setTimeout(() => setSaved(saveWorkspace(doc)), 300)
    return () => clearTimeout(timer)
  }, [doc])

  useEffect(() => {
    function handleKey(event: KeyboardEvent) {
      if (
        event.target instanceof HTMLInputElement ||
        event.target instanceof HTMLTextAreaElement ||
        event.target instanceof HTMLSelectElement
      )
        return
      if (event.key === '/' && !modal) {
        event.preventDefault()
        setModal('search')
      }
    }
    window.addEventListener('keydown', handleKey)
    return () => {
      window.removeEventListener('keydown', handleKey)
      clearTimeout(toastTimer.current)
    }
  }, [modal])

  const openGraph = useCallback((id: string) => {
    setDoc((current) => ({ ...current, ui: { ...current.ui, activeGraph: id } }))
    setSelection(null)
    setMobileNav(false)
    setLabOpen(true)
    setSimulating(false)
    setDiagramReady(false)
    setFocus(undefined)
  }, [])

  function updateGraph(id: string, change: (graph: ModelGraph) => ModelGraph) {
    // Camera animations and worker completions from a replaced document can
    // finish before React unmounts its old canvas. Scope writes to the document
    // instance that supplied this callback, even when both graph IDs match.
    if (documentEpoch.current !== documentRevision) return
    setDoc((current) =>
      documentEpoch.current !== documentRevision
        ? current
        : {
            ...current,
            graphs: current.graphs.map((g) => (g.id === id ? change(g) : g)),
          },
    )
  }

  async function loadFile(file: File | undefined) {
    if (!file) return
    try {
      const imported = upgrade(await importDocument(file))
      documentEpoch.current++
      setDoc(imported)
      setDocumentRevision(documentEpoch.current)
      setFocus(undefined)
      setDiagramReady(false)
      setSelection(null)
      setDecode(false)
      setSimulating(false)
      notify(`Opened ${imported.name}. Layouts and notes restored.`)
    } catch (error) {
      notify(
        `Could not open model: ${error instanceof Error ? error.message.slice(0, 220) : 'Invalid JSON document.'}`,
      )
    }
    if (fileInput.current) fileInput.current.value = ''
  }

  function setConfig(changes: Partial<ModelConfig>) {
    try {
      const candidate = configurationLayouts(doc, {
        ...doc,
        config: updateTeachingConfig(doc.config, changes),
      })
      const next = parseDocument(candidate)
      setDoc(next)
    } catch (error) {
      notify(error instanceof Error ? error.message : 'Invalid configuration.')
    }
  }

  function setAttention(attention: ModelConfig['attention']) {
    setConfig({ attention })
  }

  const searchResults = useMemo(
    () =>
      doc.graphs
        .flatMap((g) => compileGraph(doc, g).nodes.map((n) => ({ graph: g, node: n })))
        .filter(({ node }) =>
          `${node.title} ${node.subtitle} ${node.description}`
            .toLowerCase()
            .includes(query.toLowerCase()),
        )
        .slice(0, 40),
    [doc.graphs, doc.config, query],
  )
  const layer = Math.min(selectedLayer, doc.config.nLayers - 1)
  function focusHead(head: number) {
    setSelectedHead(head)
    if (!splitHeads) return
    const id = `${whole ? `layer-${layer}.attn.` : ''}scores-${head}`
    sim.setPlaying(false)
    setSelection({ type: 'node', id })
    setFocus({ id, revision: Date.now() })
  }
  const labProps = {
    config: doc.config,
    encoded,
    selectedToken: currentToken,
    onToken: setSelectedToken,
    decode,
    layer,
    onLayer: setSelectedLayer,
    onCachePolicy: (kvCache: ModelConfig['kvCache']) => setConfig({ kvCache }),
    onHead: focusHead,
    head: selectedHead,
  }

  return (
    <div
      className={`app ${inspectorOpen ? 'with-inspector' : ''} ${simulating ? 'with-simulator' : ''}`}
    >
      <header className="topbar">
        <div className="brand">
          <button
            className="icon-button mobile-menu"
            aria-label="Open navigation"
            onClick={() => setMobileNav(!mobileNav)}
          >
            <Menu size={20} />
          </button>
          <span className="brand-mark">
            <Logo />
          </span>
          <div>
            <strong>
              MODEL<span>VISUALIZER</span>
            </strong>
            <small>INSIDE THE ARCHITECTURE</small>
          </div>
          <span className="version">α 0.1</span>
        </div>
        <div className="project-breadcrumb">
          <Box size={14} />
          <span>{doc.name}</span>
          <ChevronRight size={12} />
          <span className="breadcrumb-current">Workbench</span>
        </div>
        <div className="topbar-actions">
          <TooltipSettingsButton />
          <button
            className="icon-button search-button"
            aria-label="Search components"
            data-tooltip-detail="Search components (/)"
            onClick={() => setModal('search')}
          >
            <Search size={17} />
          </button>
          <button
            className="icon-button"
            aria-label={doc.ui.theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
            data-tooltip-detail="Toggle theme"
            onClick={() =>
              setDoc({
                ...doc,
                ui: { ...doc.ui, theme: doc.ui.theme === 'dark' ? 'light' : 'dark' },
              })
            }
          >
            {doc.ui.theme === 'dark' ? <Sun size={17} /> : <Moon size={17} />}
          </button>
          <span className="toolbar-divider" />
          <button className="button import-button" onClick={() => fileInput.current?.click()}>
            <ArrowUpFromLine size={14} />
            <span>Open</span>
          </button>
          <button
            className="button export-button"
            onClick={() => {
              exportDocument(doc)
              notify('Exported model, all layouts, and notebook as readable JSON.')
            }}
          >
            <ArrowDownToLine size={14} />
            <span>Export model</span>
          </button>
          <input
            ref={fileInput}
            type="file"
            accept=".json,application/json"
            hidden
            aria-label="Open model JSON"
            onChange={(event) => void loadFile(event.target.files?.[0])}
          />
        </div>
      </header>

      <aside className={`sidebar ${mobileNav ? 'mobile-open' : ''}`} aria-label="Model navigation">
        <div className="sidebar-content">
          <div className="workspace-label">
            <span className="eyebrow">YOUR WORKSPACE</span>
            <span className="live-dot" />
          </div>
          <button className="model-switcher" onClick={() => setModal('config')}>
            <span className="model-avatar">
              <Cpu size={19} />
            </span>
            <span>
              <strong>{doc.name}</strong>
              <small>Decoder-only · teaching model</small>
            </span>
            <ChevronDown size={13} />
          </button>
          <div className="journey-navigation">
            {wholeModel && (
              <>
                <button
                  className={`journey-button ${whole && !simulating ? 'active' : ''}`}
                  onClick={() => openGraph(wholeModel.id)}
                >
                  <Network size={17} />
                  <span>
                    Whole model<small>Every layer on one canvas</small>
                  </span>
                </button>
                <button
                  className={`journey-button ${simulating ? 'active' : ''}`}
                  onClick={() => {
                    openGraph(whole ? graph.id : wholeModel.id)
                    setSimulating(true)
                    setInspectorOpen(true)
                  }}
                >
                  <Play size={16} />
                  <span>
                    Run simulation<small>Follow the execution</small>
                  </span>
                </button>
              </>
            )}
            <button
              className="journey-button numerical-entry"
              onClick={() => {
                sim.setPlaying(false)
                setNumericalEntry(undefined)
                setMobileNav(false)
                setModal('numerical')
              }}
            >
              <FlaskConical size={17} />
              <span>
                Tensor laboratory<small>Run the mathematics</small>
              </span>
            </button>
            <button
              className="journey-button numerical-entry"
              onClick={() => {
                sim.setPlaying(false)
                setMobileNav(false)
                setModal('local-model')
              }}
            >
              <Cpu size={17} />
              <span>
                Local trained model<small>Real weights · real activations</small>
              </span>
            </button>
          </div>
          <div className="nav-section-title">
            <span className="eyebrow">EXPLORE THE MODEL</span>
            <small>
              {String(doc.graphs.filter((g) => !isWholeModel(g)).length).padStart(2, '0')}
            </small>
          </div>
          <nav>
            {doc.graphs
              .filter((g) => !isWholeModel(g))
              .map((g, i) => {
                const Icon = graphIcons[g.id] ?? Layers3
                return (
                  <button
                    className={`nav-item ${graph.id === g.id ? 'active' : ''}`}
                    data-tooltip-data-tooltip-detail={labels[g.id] ?? g.title}
                    data-tooltip-detail={g.description}
                    key={g.id}
                    aria-current={graph.id === g.id ? 'page' : undefined}
                    onClick={() => openGraph(g.id)}
                  >
                    <Icon size={16} />
                    <span>{labels[g.id] ?? g.title}</span>
                    <small>{String(i + 1).padStart(2, '0')}</small>
                  </button>
                )
              })}
          </nav>
          <div className="config-summary">
            <div className="section-heading">
              <h3>MODEL CONFIGURATION</h3>
              <button
                className="icon-button"
                data-tooltip-detail="Edit configuration"
                aria-label="Edit model configuration"
                onClick={() => setModal('config')}
              >
                <SlidersHorizontal size={13} />
              </button>
            </div>
            <div className="config-grid">
              <span>
                Layers<strong>{doc.config.nLayers}</strong>
              </span>
              <span>
                Hidden dim<strong>{doc.config.dModel.toLocaleString()}</strong>
              </span>
              <span>
                Query heads<strong>{doc.config.nQueryHeads}</strong>
              </span>
              <span>
                KV heads<strong>{doc.config.nKvHeads}</strong>
              </span>
            </div>
            <div className="attention-type-label">Attention architecture</div>
            <div className="segmented">
              {(['MHA', 'GQA', 'MQA'] as const).map((value) => (
                <button
                  key={value}
                  aria-pressed={doc.config.attention === value}
                  onClick={() => setAttention(value)}
                >
                  {value}
                </button>
              ))}
            </div>
            <span className="config-hint">
              {doc.config.nQueryHeads / doc.config.nKvHeads} query head
              {doc.config.nQueryHeads / doc.config.nKvHeads === 1 ? '' : 's'} per KV head
            </span>
            <button className="cache-policy-shortcut" onClick={() => setModal('config')}>
              KV cache · {cachePolicyLabel(doc.config)}
            </button>
          </div>
        </div>
        <div className="sidebar-bottom">
          <button className="guide-link" onClick={() => setModal('guide')}>
            <BookOpen size={16} />
            <span>
              The field guide<small>Concepts, sources & model format</small>
            </span>
            <ArrowUpRight size={13} />
          </button>
          <div className="local-status">
            <span className="live-dot" />
            {saved
              ? 'Workspace saved on this device'
              : 'Browser storage unavailable · export to save'}
          </div>
        </div>
      </aside>
      {mobileNav && (
        <button
          className="nav-backdrop"
          aria-label="Close navigation"
          onClick={() => setMobileNav(false)}
        />
      )}

      <main className="main-workspace">
        <div className="workspace-heading">
          <div>
            <div className="workspace-crumb">
              <span>MODEL ATLAS</span>
              <ChevronRight size={11} />
              <span>{labels[graph.id] ?? graph.title}</span>
            </div>
            <h1>{graph.title}</h1>
            <p>{graph.subtitle}</p>
          </div>
          <div className="workspace-heading-actions">
            <span className="teaching-badge">
              <FlaskConical size={12} />
              Educational
            </span>
            <button
              className="icon-button"
              data-tooltip-detail="Toggle inspector"
              aria-label="Toggle inspector"
              aria-pressed={inspectorOpen}
              onClick={() => setInspectorOpen(!inspectorOpen)}
            >
              {inspectorOpen ? <PanelRightClose size={18} /> : <PanelRightOpen size={18} />}
            </button>
          </div>
        </div>
        {simulating ? (
          <SimulationStrip sim={sim} />
        ) : (
          <section className="prompt-panel" aria-label="Prompt and tokenization">
            <div className="prompt-input-row">
              <label htmlFor="prompt">
                <Command size={13} />
                PROMPT
              </label>
              <input
                id="prompt"
                value={doc.ui.prompt}
                maxLength={240}
                placeholder="Type a prompt to inspect its tokens…"
                onChange={(event) =>
                  setDoc({ ...doc, ui: { ...doc.ui, prompt: event.target.value } })
                }
                spellCheck={false}
              />
              <span className="token-count">
                {tokenizerSupported ? encoded.tokens.length : '—'}
                <small>tokens</small>
              </span>
            </div>
            {tokenizerSupported ? (
              <div className="token-ribbon">
                <span className="ribbon-caption">TOKEN IDS</span>
                <div className="token-items">
                  {encoded.tokens.map((token, i) => (
                    <button
                      className={`token-chip token-color-${i % 3} ${currentToken === i ? 'selected-token' : ''}`}
                      key={`${i}-${token.id}`}
                      data-tooltip-detail={`Position ${i} · ID ${token.id} · bytes [${token.bytes.join(', ')}]`}
                      aria-label={`Token ${i}: ${token.label}, ID ${token.id}`}
                      onClick={() => setSelectedToken(i)}
                    >
                      <span>{token.label}</span>
                      <small>{token.id}</small>
                    </button>
                  ))}
                  {encoded.tokens.length === 0 && (
                    <span className="muted">
                      Enter text. Shapes currently use T = 1 as a reference.
                    </span>
                  )}
                </div>
                <button
                  className="tokenizer-open"
                  data-tooltip-detail="Open tokenizer lab"
                  aria-label="Open tokenizer lab"
                  onClick={() => openGraph('tokenization')}
                >
                  <ArrowUpRight size={15} />
                </button>
              </div>
            ) : (
              <div className="unsupported-tokenizer">
                <HelpCircle size={13} />
                Adapter “{doc.tokenizer.adapter}” is unavailable. Graph shapes use reference T = 12;
                token labs are disabled.
              </div>
            )}
          </section>
        )}
        <div className="graph-toolbar">
          {splitHeads && (
            <label className="whole-model-jump">
              Head
              <select
                aria-label="Inspect query head"
                value={Math.min(selectedHead, doc.config.nQueryHeads - 1)}
                onChange={(e) => focusHead(Number(e.target.value))}
              >
                <option value="">Choose a head…</option>
                {Array.from({ length: doc.config.nQueryHeads }, (_, h) => (
                  <option key={h} value={h}>
                    Q{h} → K{kvHeadFor(doc.config, h)} / V{kvHeadFor(doc.config, h)}
                  </option>
                ))}
              </select>
            </label>
          )}
          {(whole || ['attention', 'attention-heads'].includes(graph.id)) &&
            doc.graphs.some((g) => g.id === 'attention-heads') && (
              <button
                className="head-detail-button"
                aria-pressed={splitHeads}
                onClick={() => {
                  const playing = simulating
                  openGraph(
                    whole
                      ? splitHeads
                        ? wholeModel!.id
                        : 'whole-model-heads'
                      : splitHeads
                        ? 'attention'
                        : 'attention-heads',
                  )
                  setSimulating(playing)
                }}
              >
                {splitHeads ? 'Batched attention' : 'Split attention heads'}
              </button>
            )}
          {whole && (
            <label className="whole-model-jump">
              Jump to
              <select
                aria-label="Jump to model stage"
                value=""
                onChange={(event) => {
                  const node = graph.nodes.find((n) => instanceOf(n)?.stage === event.target.value)
                  if (node) {
                    const layer = instanceOf(node)?.layer
                    if (layer !== undefined) setSelectedLayer(layer)
                    sim.setPlaying(false)
                    setFocus({ id: node.id, revision: Date.now() })
                  }
                }}
              >
                <option value="">Choose a stage…</option>
                {[...new Set(graph.nodes.map((n) => instanceOf(n)?.stage).filter(Boolean))].map(
                  (stage) => (
                    <option key={stage}>{stage}</option>
                  ),
                )}
              </select>
            </label>
          )}
          {!simulating && (
            <>
              <div className="segmented inference-mode">
                <button aria-pressed={!decode} onClick={() => setDecode(false)}>
                  Prefill
                </button>
                <button aria-pressed={decode} onClick={() => setDecode(true)}>
                  Decode step
                </button>
              </div>
              {['block', 'attention', 'attention-heads', 'feedforward', 'cache'].includes(
                graph.id,
              ) && (
                <label className="layer-selector">
                  ℓ
                  <select
                    aria-label="Active decoder layer"
                    value={layer}
                    onChange={(event) => setSelectedLayer(Number(event.target.value))}
                  >
                    {Array.from({ length: doc.config.nLayers }, (_, i) => (
                      <option key={i} value={i}>
                        {i}
                      </option>
                    ))}
                  </select>
                </label>
              )}
              <span className="inference-caption">
                {decode
                  ? `T = 1 · S = ${symbols.S} · one new token`
                  : `T = S = ${symbols.T} · prompt in parallel`}
              </span>
            </>
          )}
          {simulating && (
            <span className="inference-caption simulation-caption">
              {simulationFrame?.phase.toUpperCase() ?? 'READY'} · {graph.nodes.length} components ·{' '}
              {graph.edges.length} connections
            </span>
          )}
          <div className="shape-options">
            <label className="toggle-label">
              <input
                type="checkbox"
                checked={showShapes}
                onChange={(e) => setShowShapes(e.target.checked)}
              />
              <span className="mini-toggle" />
              Dimensions
            </label>
            <button
              className={`symbolic-button ${symbolic ? 'active' : ''}`}
              data-tooltip-detail="Toggle symbolic dimensions"
              aria-label="Toggle symbolic dimensions"
              aria-pressed={symbolic}
              onClick={() => setSymbolic(!symbolic)}
            >
              ƒx
            </button>
          </div>
        </div>
        <div className="graph-region">
          <GraphCanvas
            key={`${doc.id}-${documentRevision}`}
            graph={graph}
            symbols={symbols}
            selection={selection}
            showShapes={showShapes}
            symbolic={symbolic}
            theme={doc.ui.theme}
            onSelect={setSelection}
            onOpen={openGraph}
            notify={notify}
            execution={
              simulating
                ? {
                    active: simulationFrame?.nodeIds ?? [],
                    completed: sim.completed,
                    revision: sim.cursor,
                    playing: sim.playing,
                  }
                : undefined
            }
            focus={
              simulating && sim.follow && simulationFrame
                ? { id: simulationFrame.nodeIds[0], revision: sim.cursor }
                : focus
            }
            onReady={simulating ? setDiagramReady : undefined}
            onLayout={(positions, direction, resetViewport) =>
              updateGraph(graph.id, (g) => ({
                ...g,
                edges: direction
                  ? g.edges.map((e) => ({ ...e, sourceSide: undefined, targetSide: undefined }))
                  : g.edges,
                layout: {
                  ...g.layout,
                  positions,
                  ...(direction ? { direction } : {}),
                  ...(resetViewport ? { viewport: undefined } : {}),
                },
              }))
            }
            onViewport={(viewport) =>
              updateGraph(graph.id, (g) => ({ ...g, layout: { ...g.layout, viewport } }))
            }
            onRestore={(snapshot) =>
              updateGraph(graph.id, (g) => ({
                ...g,
                layout: snapshot.layout,
                edges: g.edges.map((edge) => {
                  const previous = snapshot.edges.find((e) => e.id === edge.id)
                  return {
                    ...edge,
                    sourceSide: previous?.sourceSide,
                    targetSide: previous?.targetSide,
                  }
                }),
              }))
            }
          />
        </div>
        {hasLab && tokenizerSupported && (
          <section
            className={`interactive-lab ${showLab ? 'lab-open' : ''}`}
            aria-label="Interactive lab"
          >
            <button
              className="lab-toggle"
              aria-expanded={showLab}
              onClick={() => setLabOpen(!labOpen)}
            >
              <FlaskConical size={14} />
              <span>
                {['attention', 'attention-heads'].includes(graph.id)
                  ? 'Attention head explorer'
                  : graph.id === 'cache'
                    ? 'KV tensor explorer'
                    : graph.id === 'memory'
                      ? 'N-gram address explorer'
                      : graph.id === 'mtp'
                        ? 'Prediction alignment'
                        : 'Byte-pair merge explorer'}
              </span>
              <span className="lab-toggle-right">
                INTERACTIVE
                <ChevronDown className={showLab ? 'rotate' : ''} size={14} />
              </span>
            </button>
            {showLab &&
              (encoded.tokens.length === 0 ? (
                <div className="empty-lab">Enter a prompt to explore this pathway.</div>
              ) : graph.id === 'tokenization' ? (
                <TokenizerLab key={doc.ui.prompt} {...labProps} />
              ) : ['attention', 'attention-heads'].includes(graph.id) ? (
                <AttentionLab {...labProps} />
              ) : graph.id === 'cache' ? (
                <CacheLab {...labProps} />
              ) : graph.id === 'memory' ? (
                <MemoryLab {...labProps} />
              ) : (
                <MtpLab {...labProps} />
              ))}
          </section>
        )}
        <footer className="workspace-footer">
          <div className="graph-legend">
            <span>
              <i className="legend-data" />
              Data
            </span>
            <span>
              <i className="legend-compute" />
              Compute
            </span>
            <span>
              <i className="legend-memory" />
              Memory
            </span>
            <span>
              <i className="legend-output" />
              Output
            </span>
          </div>
          <span className="footer-right">
            <Check size={11} />
            Local-first<span>·</span>JSON v{doc.schemaVersion}
          </span>
        </footer>
      </main>
      {simulating
        ? inspectorOpen && (
            <SimulationPanel
              sim={sim}
              config={doc.config}
              onCachePolicy={(kvCache) => setConfig({ kvCache })}
              ready={diagramReady}
              onClose={() => {
                sim.setPlaying(false)
                setInspectorOpen(false)
              }}
            />
          )
        : inspectorOpen && (
            <Inspector
              graph={graph}
              layer={layer}
              doc={doc}
              selection={selection}
              symbols={symbols}
              symbolic={symbolic}
              onOpen={openGraph}
              onNotes={(notes) => setDoc({ ...doc, ui: { ...doc.ui, notes } })}
              notify={notify}
              onNumerical={(entry) => {
                sim.setPlaying(false)
                setNumericalEntry(entry)
                setModal('numerical')
              }}
            />
          )}
      {toast && (
        <div className="toast" role="status">
          <Check size={16} />
          <span>{toast}</span>
          <button aria-label="Dismiss notification" onClick={() => setToast('')}>
            <X size={14} />
          </button>
        </div>
      )}

      {modal === 'numerical' && (
        <Suspense
          fallback={
            <Dialog title="Opening the tensor laboratory…" onClose={() => setModal(null)}>
              <p className="modal-description">Preparing the local numerical model.</p>
            </Dialog>
          }
        >
          <NumericalWorkbench
            doc={doc}
            reference={referenceDocument}
            initialEntry={numericalEntry}
            onSave={saveNumericalSession}
            onClose={() => setModal(null)}
            onTheme={() =>
              setDoc((current) => ({
                ...current,
                ui: { ...current.ui, theme: current.ui.theme === 'dark' ? 'light' : 'dark' },
              }))
            }
            notify={notify}
          />
        </Suspense>
      )}
      {modal === 'local-model' && (
        <Suspense
          fallback={
            <Dialog title="Opening the local model…" onClose={() => setModal(null)}>
              <p>Preparing checkpoint controls. No inference has started.</p>
            </Dialog>
          }
        >
          <LocalModelWorkbench
            doc={doc}
            reference={referenceDocument}
            onSave={saveCheckpointSession}
            onClose={() => setModal(null)}
            notify={notify}
            onTheme={() =>
              setDoc((current) => ({
                ...current,
                ui: { ...current.ui, theme: current.ui.theme === 'dark' ? 'light' : 'dark' },
              }))
            }
          />
        </Suspense>
      )}
      {modal === 'config' && (
        <Dialog title="Shape the teaching model." onClose={() => setModal(null)}>
          <p className="modal-description">
            Change the architecture and watch every tensor dimension update. No model weights are
            loaded.
          </p>
          <TeachingModelControls document={doc} onChange={setConfig} />
          <CachePolicyControls config={doc.config} onChange={(kvCache) => setConfig({ kvCache })} />
          <div className="configuration-result">
            <Cpu size={20} />
            <div>
              <strong>D = Hq × Dh = {doc.config.dModel.toLocaleString()}</strong>
              <span>
                {doc.config.attention} · {doc.config.nKvHeads} KV heads · vocabulary{' '}
                {doc.config.vocabSize}
              </span>
            </div>
            <Database size={19} />
            <div>
              <strong>{formatBytes(cacheBytes(doc.config, doc.config.contextLength))}</strong>
              <span>logical KV state at context capacity</span>
            </div>
          </div>
          <div className="modal-footer">
            <span>Changes save automatically on this device.</span>
            <button className="button export-button" onClick={() => setModal(null)}>
              Back to the model
              <ArrowRight size={14} />
            </button>
          </div>
        </Dialog>
      )}

      {modal === 'search' && (
        <Dialog
          title="Find a component."
          onClose={() => {
            setModal(null)
            setQuery('')
          }}
          className="search-modal"
        >
          <div className="search-field">
            <Search size={18} />
            <input
              autoFocus
              aria-label="Search model components"
              placeholder="Try ‘dot product’, ‘RoPE’, ‘cache’…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
          </div>
          <div className="search-results">
            {searchResults.map(({ graph: g, node }) => (
              <button
                key={`${g.id}-${node.id}`}
                onClick={() => {
                  openGraph(g.id)
                  setSelection({ type: 'node', id: node.id })
                  setFocus({ id: node.id, revision: Date.now() })
                  setInspectorOpen(true)
                  setModal(null)
                  setQuery('')
                }}
              >
                <span className={`search-icon category-${node.category}`}>
                  <Box size={16} />
                </span>
                <span>
                  <strong>{node.title}</strong>
                  <small>
                    {labels[g.id] ?? g.title} · {node.subtitle}
                  </small>
                </span>
                <ArrowUpRight size={14} />
              </button>
            ))}
            {searchResults.length === 0 && (
              <p>No matching components. Try a different operation or name.</p>
            )}
          </div>
        </Dialog>
      )}

      {modal === 'guide' && (
        <Dialog
          title="A field guide to the model."
          onClose={() => setModal(null)}
          className="guide-modal"
        >
          <div className="guide-tabs">
            {(['guide', 'sources', 'format'] as const).map((value) => (
              <button
                key={value}
                className={guideTab === value ? 'active' : ''}
                onClick={() => setGuideTab(value)}
              >
                {value === 'guide'
                  ? 'Getting oriented'
                  : value === 'sources'
                    ? 'Research library'
                    : 'The model format'}
              </button>
            ))}
          </div>
          <div className="guide-content">
            {guideTab === 'guide' ? (
              <>
                <p className="guide-lead">{doc.description}</p>
                <p>{doc.documentation}</p>
                <div className="guide-callout">
                  <FlaskConical size={20} />
                  <div>
                    <strong>Explore structure. Inspect numerical reference values.</strong>
                    <p>
                      The BPE tokenizer runs locally. Tensor dimensions and cache byte counts are
                      calculated. Study heatmaps and graphical MTP choices are illustrative. The
                      separate tensor laboratory executes a small transformer with seeded, untrained
                      weights and explains its actual tensor values. No trained language model or
                      inference service is used.
                    </p>
                    <button
                      className="numerical-example-button"
                      onClick={() => {
                        setNumericalEntry(undefined)
                        setModal('numerical')
                      }}
                    >
                      <FlaskConical size={15} /> Open tensor laboratory <ArrowUpRight size={14} />
                    </button>
                  </div>
                </div>
                <h3>Ways to explore</h3>
                <ol>
                  <li>Select a component to inspect its inputs, outputs, weights and LaTeX.</li>
                  <li>Select a connection to see the mathematical operation it represents.</li>
                  <li>Open a component with an expand icon to explore the graph inside.</li>
                  <li>
                    Try a numerical example, then select a matrix cell and follow its operands.
                  </li>
                </ol>
                <h3>Make a view yours</h3>
                <p>
                  Drag nodes, pan and zoom. Auto-layout rearranges the current graph; the direction
                  button switches between horizontal and vertical flow. Every graph keeps its own
                  layout. Export the model to keep a portable copy, including your notebook.
                </p>
                <h3>What comes next</h3>
                <p>
                  The foundation is a dense, decoder-only model. Separate future definitions can
                  describe MoE routing, MLA, more advanced sparse patterns, encoder–decoder models,
                  alternative positional encodings and custom tokenizers. Each needs an
                  architecture-specific graph and sources.
                </p>
              </>
            ) : guideTab === 'sources' ? (
              <>
                {doc.sources.map((source) => (
                  <a
                    className="research-source"
                    href={source.url}
                    key={source.id}
                    target="_blank"
                    rel="noreferrer"
                  >
                    <BookOpen size={17} />
                    <span>
                      <strong>{source.title}</strong>
                      <p>{source.note}</p>
                    </span>
                    <ArrowUpRight size={15} />
                  </a>
                ))}
              </>
            ) : (
              <>
                <div className="format-title">
                  <FileJson size={29} />
                  <div>
                    <h3>Readable. Portable. Extensible.</h3>
                    <p>A versioned JSON document with a published JSON Schema.</p>
                  </div>
                </div>
                <p>
                  Model semantics and layout travel together, but occupy separate fields. An agent
                  can edit explanations, equations and graph structure without depending on UI code.
                </p>
                <pre>
                  {JSON.stringify(
                    {
                      schemaVersion: '1.0',
                      name: 'My model',
                      config: { dModel: 512 },
                      graphs: [
                        {
                          id: 'attention',
                          nodes: ['components + tensors + LaTeX'],
                          edges: ['operations + tensor shapes'],
                          layout: { positions: {}, direction: 'RIGHT' },
                        },
                      ],
                      sources: ['primary research'],
                      ui: { notes: 'My documentation' },
                    },
                    null,
                    2,
                  )}
                </pre>
                <p>
                  This is a structural sketch. Export a complete document to start editing. Import
                  checks the schema, graph links, source references, head-sharing rules and
                  dimension expressions before replacing your workspace.
                </p>
                <p>
                  Use <code>extensions</code> for namespaced metadata. Unknown tokenizer adapters
                  keep graph inspection available and disable token labs until an adapter is
                  registered.
                </p>
                <button className="button export-button" onClick={() => exportDocument(doc)}>
                  <ArrowDownToLine size={14} />
                  Download complete example
                </button>
              </>
            )}
          </div>
          <div className="modal-footer">
            <a
              href="https://github.com/LightningPhil/Model-Visualizer"
              target="_blank"
              rel="noreferrer"
            >
              <Github size={14} />
              Model-Visualizer
            </a>
            <span>Research checked · 20 October 2026</span>
          </div>
        </Dialog>
      )}
    </div>
  )
}
