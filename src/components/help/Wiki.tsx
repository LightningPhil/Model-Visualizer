import { useEffect, useId, useMemo, useRef, useState, type ReactNode } from 'react'
import {
  ArrowLeft,
  ArrowRight,
  ArrowUpRight,
  BookOpen,
  Bookmark,
  Check,
  ChevronRight,
  Compass,
  FileText,
  Home,
  FlaskConical,
  List,
  Search,
  X,
} from 'lucide-react'
import { Dialog } from '../Dialog'
import { MathBlock } from '../Math'
import { shapeLabel } from '../../model/schema'
import {
  buildWikiEntries,
  buildWikiSearchIndex,
  conceptsFor,
  entryId,
  learningPaths,
  modeLabels,
  searchWiki,
  wikiArticles,
  wikiCategories,
  modelExamplesFor,
  suggestWikiQueries,
  type WikiContext,
  type WikiEntry,
} from '../../help/wiki/catalog'
import './wiki.css'
import { WikiExploration } from './WikiExploration'
import { WikiGlossary, WikiModelConnections, WikiReadingCollection } from './WikiCollections'
import { useWikiReading } from './useWikiReading'
import { readingKey } from '../../help/wiki/reading'
import { glossary } from '../../help/wiki/glossary'
import { experimentEntries, explorationFor } from '../../help/wiki/explorations'
import { reasoningQuestions } from '../../help/wiki/questions'
import { perspectives } from '../../help/wiki/perspectives'
import type { ReadingMode } from '../../help/wiki/perspective-types'
import { Investigations, InvestigationQuestions } from './Investigations'
import type { StudyRecipe } from '../../model/study/records'
import './study.css'

const termLinks: Record<string, string> = {
  'KV cache': 'kv-cache',
  'K/V state': 'kv-cache',
  'K/V cache': 'kv-cache',
  RMSNorm: 'rmsnorm',
  LayerNorm: 'rmsnorm',
  RoPE: 'rope',
  SwiGLU: 'swiglu',
  SiLU: 'swiglu',
  softmax: 'softmax',
  logits: 'logits',
  GQA: 'head-sharing',
  MQA: 'head-sharing',
  MHA: 'head-sharing',
  FlashAttention: 'storage-kernels',
  PagedAttention: 'storage-kernels',
  'linear attention': 'linear-attention',
  'KV quantization': 'cache-quantization',
  'prefix sharing': 'prefix-sharing',
  backpropagation: 'learning-loop',
  ALiBi: 'position-methods',
  superposition: 'feature-representations',
  'activation patching': 'causal-interventions',
  DPO: 'post-training',
  AdamW: 'adamw',
  Adam: 'adam',
  Muon: 'muon',
  'floating-point': 'numerical-precision',
  'residual stream': 'residual',
  'residual addition': 'residual',
  embedding: 'embeddings',
  BPE: 'tokenization',
  'cross-entropy': 'training-objective',
  MTP: 'mtp',
  MLA: 'mla',
  'mixture-of-experts': 'mixture-of-experts',
  prefill: 'prefill-decode',
  temperature: 'sampling',
  'tensor laboratory': 'tensor-laboratory',
  greedy: 'sampling',
  'categorical sampling': 'sampling',
}
const terms = Object.keys(termLinks).sort((a, b) => b.length - a.length)
const termPattern = new RegExp(
  `\\b(${terms.map((term) => term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|')})\\b`,
  'gi',
)
const termIds = new Map(Object.entries(termLinks).map(([term, id]) => [term.toLowerCase(), id]))

function LinkedText({
  text,
  current,
  onRead,
}: {
  text: string
  current: string
  onRead: (id: string) => void
}) {
  const parts: ReactNode[] = []
  let offset = 0
  for (const match of text.matchAll(termPattern)) {
    const index = match.index!
    if (index > offset) parts.push(text.slice(offset, index))
    const id = termIds.get(match[0].toLowerCase())!
    parts.push(
      id === current ? (
        match[0]
      ) : (
        <button className="wiki-inline-link" key={index} onClick={() => onRead(id)}>
          {match[0]}
        </button>
      ),
    )
    offset = index + match[0].length
  }
  parts.push(text.slice(offset))
  return <>{parts}</>
}

function excerpt(entry: WikiEntry, query: string) {
  const queryWords = query
    .toLowerCase()
    .split(/\s+/)
    .filter((word) => word.length > 3)
  const paragraphs = entry.sections.flatMap((s) => s.paragraphs)
  const matching = paragraphs.find((p) => queryWords.some((word) => p.toLowerCase().includes(word)))
  const text = matching ?? entry.summary
  const position = Math.max(
    0,
    queryWords.map((word) => text.toLowerCase().indexOf(word)).find((index) => index >= 0) ?? 0,
  )
  const start = Math.max(0, position - 70)
  return `${start ? '…' : ''}${text.slice(start, start + 230)}${text.length > start + 230 ? '…' : ''}`
}

export function Wiki({
  context,
  onClose,
  docked = false,
  canDock = false,
  onDock,
}: {
  context: WikiContext
  onClose: () => void
  docked?: boolean
  canDock?: boolean
  onDock?: () => void
}) {
  const entries = useMemo(() => buildWikiEntries(context), [context])
  const searchIndex = useMemo(() => buildWikiSearchIndex(entries), [entries])
  const entryMap = useMemo(() => new Map(entries.map((entry) => [entry.id, entry])), [entries])
  const selected = context.selection ? context.selection : undefined
  const contextId = entryId({
    graphId: context.graph.id,
    type: selected?.type ?? 'graph',
    id: selected?.id ?? context.graph.id,
  })
  const [history, setHistory] = useState({ ids: [context.initialArticle ?? contextId], index: 0 })
  const [query, setQuery] = useState('')
  const [category, setCategory] = useState('All topics')
  const [browse, setBrowse] = useState(false)
  const [followSelection, setFollowSelection] = useState(false)
  const [study, setStudy] = useState<StudyRecipe | undefined>(context.initialStudy)
  const [studyRevision, setStudyRevision] = useState(0)
  useEffect(() => {
    if (docked && followSelection) read(contextId)
  }, [contextId, docked, followSelection])
  function restoreStudy(recipe: StudyRecipe) {
    if (recipe.kind === 'training' || recipe.kind === 'investigation') {
      setStudy(recipe)
      setStudyRevision((r) => r + 1)
      read('investigations')
    } else if (context.onRestoreStudy) {
      context.onRestoreStudy(recipe)
      close()
    }
  }
  function chooseInvestigation(id: string) {
    setStudy(
      id === 'training'
        ? undefined
        : {
            kind: 'investigation',
            investigation: id as 'values' | 'sharing' | 'memory',
            value: 0,
            query: 1,
            length: 1024,
          },
    )
    if (id === 'training')
      setStudy({
        kind: 'training',
        settings: { seed: 7, task: 'curve', optimizer: 'adam', rate: 0.03, decay: 0.01, steps: 0 },
      })
    setStudyRevision((r) => r + 1)
    read('investigations')
  }
  const [limit, setLimit] = useState(60)
  const [readingMode, setReadingMode] = useState<ReadingMode>('introduction')
  const modeHistory = useRef(new Map<string, ReadingMode>())
  const reading = useWikiReading(context.doc.id)
  const positionTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)
  const contentRef = useRef<HTMLDivElement>(null)
  const searchRef = useRef<HTMLInputElement>(null)
  const scrollPositions = useRef(new Map<string, number>())
  const prefix = useId().replace(/:/g, '')
  const id = history.ids[history.index]
  const article = entryMap.get(id)
  const perspective = article?.origin === 'concept' ? perspectives[id] : undefined
  const activeMode = perspective ? readingMode : 'technical'
  const positionKey = `${id}|${activeMode}`
  const sections = perspective
    ? activeMode === 'introduction'
      ? perspective.introduction
      : activeMode === 'technical'
        ? [...article!.sections, ...perspective.technical]
        : []
    : (article?.sections ?? [])
  const matches = useMemo(
    () => (article ? modelExamplesFor(article, entries, context) : []),
    [article, entries, context],
  )
  const questions = reasoningQuestions[id] ?? []
  const learningPath = learningPaths.find((path) => path.articles.includes(id))
  const pathIndex = learningPath?.articles.indexOf(id) ?? -1
  const experimentEntry = experimentEntries[id]
  const saved = reading.library.bookmarks.some(
    (place) => place.key === readingKey(id, context.doc.id),
  )
  const searching = query.trim().length > 0 || category !== 'All topics'
  const results = useMemo(
    () => searchWiki(entries, query, category, searchIndex),
    [entries, query, category, searchIndex],
  )
  const contextEntry = entryMap.get(contextId)
  const suggested = contextEntry?.related.length
    ? contextEntry.related
    : conceptsFor(undefined, context.graph.id)
  useEffect(() => {
    contentRef.current?.querySelector<HTMLElement>('h3')?.focus({ preventScroll: true })
  }, [id])
  useEffect(() => {
    if (article)
      reading.remember(article, scrollPositions.current.get(positionKey) ?? 0, activeMode)
    return () => clearTimeout(positionTimer.current)
  }, [id, activeMode])
  function rememberCurrent() {
    clearTimeout(positionTimer.current)
    if (article && !searching && contentRef.current)
      reading.remember(article, contentRef.current.scrollTop, activeMode)
  }
  function close() {
    rememberCurrent()
    onClose()
  }
  const read = (next: string, resume = false, preferredMode?: ReadingMode) => {
    rememberCurrent()
    if (contentRef.current && !searching)
      scrollPositions.current.set(positionKey, contentRef.current.scrollTop)
    modeHistory.current.set(id, readingMode)
    const place = resume
      ? [...reading.library.recent, ...reading.library.bookmarks].find(
          (p) => p.key === readingKey(next, context.doc.id),
        )
      : undefined
    const storedMode = place?.readingMode
    const nextMode =
      preferredMode ??
      (storedMode === 'introduction' || storedMode === 'technical' || storedMode === 'explore'
        ? storedMode
        : readingMode === 'explore'
          ? 'introduction'
          : readingMode)
    setReadingMode(nextMode)
    setHistory((old) => ({ ids: [...old.ids.slice(0, old.index + 1), next], index: old.index + 1 }))
    setQuery('')
    setCategory('All topics')
    setBrowse(false)
    contentRef.current?.scrollTo({ top: 0 })
    if (resume) {
      if (place) {
        scrollPositions.current.set(`${next}|${nextMode}`, place.scroll)
        requestAnimationFrame(() => contentRef.current?.scrollTo({ top: place.scroll }))
      }
    }
  }
  const travel = (direction: number) => {
    rememberCurrent()
    if (contentRef.current && !searching)
      scrollPositions.current.set(positionKey, contentRef.current.scrollTop)
    modeHistory.current.set(id, readingMode)
    const index = history.index + direction
    if (index < 0 || index >= history.ids.length) return
    setHistory({ ...history, index })
    const nextMode = modeHistory.current.get(history.ids[index]) ?? 'introduction'
    setReadingMode(nextMode)
    setQuery('')
    setCategory('All topics')
    setBrowse(false)
    requestAnimationFrame(() =>
      contentRef.current?.scrollTo({
        top: scrollPositions.current.get(`${history.ids[index]}|${nextMode}`) ?? 0,
      }),
    )
  }
  function changeMode(mode: ReadingMode) {
    rememberCurrent()
    if (contentRef.current) scrollPositions.current.set(positionKey, contentRef.current.scrollTop)
    modeHistory.current.set(id, mode)
    setReadingMode(mode)
    requestAnimationFrame(() =>
      contentRef.current?.scrollTo({ top: scrollPositions.current.get(`${id}|${mode}`) ?? 0 }),
    )
  }
  const jump = (section: string) =>
    document.getElementById(`${prefix}-${section}`)?.scrollIntoView({
      block: 'start',
      behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches
        ? 'instant'
        : 'smooth',
    })
  const link = (target: string, label?: string) => {
    const entry = entryMap.get(target)
    return entry ? (
      <button key={target} onClick={() => read(target)}>
        {label ?? entry.title}
        <ChevronRight size={13} />
      </button>
    ) : null
  }
  const linked = (text: string) => <LinkedText text={text} current={id} onRead={read} />
  const words = article
    ? sections
        .flatMap((s) => s.paragraphs)
        .join(' ')
        .split(/\s+/).length
    : 0

  return (
    <Dialog
      title="The model wiki"
      className={`wiki-modal${docked ? ' wiki-docked' : ''}`}
      modal={!docked}
      onClose={close}
      onKeyDown={(event) => {
        // A nested reader must not step/play the numerical trace or open the
        // architecture search through a global shortcut behind its modal.
        event.stopPropagation()
        if (docked && event.key === 'Escape') {
          event.preventDefault()
          close()
          return
        }
        if (
          event.key === '/' &&
          !(event.target as HTMLElement).closest('input,textarea,[contenteditable=true]') &&
          !event.ctrlKey &&
          !event.metaKey &&
          !event.altKey
        ) {
          event.preventDefault()
          searchRef.current?.focus()
        }
      }}
    >
      <div className="wiki-dock-controls">
        <button disabled={!canDock} onClick={onDock}>
          {docked ? 'Expand reader' : 'Dock beside model'}
        </button>
        {docked && (
          <label>
            <input
              type="checkbox"
              checked={followSelection}
              onChange={(e) => setFollowSelection(e.target.checked)}
            />{' '}
            Follow selection
          </label>
        )}
      </div>
      <div className="wiki-toolbar">
        <div className="wiki-history">
          <button aria-label="Wiki back" disabled={history.index === 0} onClick={() => travel(-1)}>
            <ArrowLeft size={16} />
          </button>
          <button
            aria-label="Wiki forward"
            disabled={history.index === history.ids.length - 1}
            onClick={() => travel(1)}
          >
            <ArrowRight size={16} />
          </button>
          <button aria-label="Wiki home" onClick={() => read('home')}>
            <Home size={16} />
          </button>
        </div>
        <label className="wiki-search">
          <Search size={17} />
          <input
            ref={searchRef}
            aria-label="Search the model wiki"
            placeholder="Search concepts, questions, symbols, model components…"
            value={query}
            onChange={(event) => {
              setQuery(event.target.value)
              setLimit(60)
            }}
          />
          {query && (
            <button aria-label="Clear wiki search" onClick={() => setQuery('')}>
              <X size={14} />
            </button>
          )}
        </label>
        <button
          className="wiki-browse-button"
          aria-label="Browse wiki topics"
          aria-expanded={browse}
          onClick={() => setBrowse(!browse)}
        >
          <List size={17} />
          Topics
        </button>
      </div>
      <div className="wiki-shell">
        <nav className={`wiki-sidebar ${browse ? 'is-open' : ''}`} aria-label="Wiki topics">
          <span className="eyebrow">YOUR CURRENT STUDY</span>
          <button
            className={`wiki-context-link ${id === contextId && !searching ? 'active' : ''}`}
            onClick={() => read(contextId)}
          >
            <Compass size={16} />
            <span>
              {contextEntry?.title ?? context.graph.title}
              <small>Explain this view{selected ? ' / selection' : ''}</small>
            </span>
          </button>
          <button
            className={`wiki-nav-home ${id === 'home' && !searching ? 'active' : ''}`}
            onClick={() => read('home')}
          >
            <BookOpen size={15} />
            Learning paths & all topics
          </button>
          <button
            className={`wiki-nav-home ${id === 'glossary' ? 'active' : ''}`}
            onClick={() => read('glossary')}
          >
            <FileText size={15} />
            Glossary & current symbols
          </button>
          <button
            className={`wiki-nav-home ${id === 'saved' ? 'active' : ''}`}
            onClick={() => read('saved')}
          >
            <Bookmark size={15} />
            Saved & recent reading
            <span className="wiki-saved-count">{reading.library.bookmarks.length}</span>
          </button>
          <button
            className={`wiki-nav-home ${id === 'investigations' ? 'active' : ''}`}
            onClick={() => read('investigations')}
          >
            <FlaskConical size={15} />
            Investigations & records
          </button>
          {wikiCategories.map((name) => (
            <div className="wiki-nav-group" key={name}>
              <span className="eyebrow">{name}</span>
              {wikiArticles
                .filter((entry) => entry.category === name)
                .map((entry) => (
                  <button
                    className={entry.id === id && !searching ? 'active' : ''}
                    key={entry.id}
                    onClick={() => read(entry.id)}
                  >
                    {entry.title}
                  </button>
                ))}
            </div>
          ))}
          <button
            className="wiki-document-browse"
            onClick={() => {
              setCategory('Model document')
              setQuery('')
              setBrowse(false)
              contentRef.current?.scrollTo({ top: 0 })
            }}
          >
            <FileText size={15} />
            Browse model document
          </button>
          <button className="wiki-document-browse" onClick={() => read('document:overview')}>
            <BookOpen size={15} />
            Model documentation & sources
          </button>
        </nav>
        <div
          className="wiki-reading-pane"
          ref={contentRef}
          onScroll={(event) => {
            if (!article || searching) return
            clearTimeout(positionTimer.current)
            const scroll = event.currentTarget.scrollTop
            positionTimer.current = setTimeout(
              () => reading.remember(article, scroll, activeMode),
              350,
            )
          }}
        >
          <div className="wiki-mode">
            <span className="tiny-square" />
            {modeLabels[context.mode]}
          </div>
          {docked && context.inspection && (
            <section className="wiki-live-cell">
              <span className="eyebrow">SELECTED NUMERICAL CELL</span>
              <p>
                {context.inspection.name} · {context.inspection.axes.join(' × ')} [
                {context.inspection.coordinate.join(', ')}] ={' '}
                <code>{context.inspection.value}</code>
              </p>
              <details>
                <summary>Explain this value</summary>
                <p>{context.inspection.explanation}</p>
                <MathBlock tex={context.inspection.tex} />
              </details>
            </section>
          )}
          {searching ? (
            <div className="wiki-search-page">
              <span className="eyebrow">SEARCH THE WHOLE LIBRARY</span>
              <h3>{query.trim() ? `Results for “${query.trim()}”` : category}</h3>
              <div className="wiki-result-controls">
                <label>
                  Collection
                  <select
                    aria-label="Wiki collection"
                    value={category}
                    onChange={(event) => {
                      setCategory(event.target.value)
                      setLimit(60)
                    }}
                  >
                    {['All topics', ...wikiCategories, 'Glossary', 'Model document'].map((name) => (
                      <option key={name}>{name}</option>
                    ))}
                  </select>
                </label>
                <span role="status">
                  {results.length} result{results.length === 1 ? '' : 's'}
                </span>
              </div>
              {results.length ? (
                <div className="wiki-results">
                  {results.slice(0, limit).map((entry) => (
                    <button key={entry.id} onClick={() => read(entry.id)}>
                      <span className="wiki-result-type">
                        {entry.origin === 'document'
                          ? 'MODEL DOCUMENT'
                          : entry.category.toUpperCase()}
                      </span>
                      <strong>
                        {entry.title}
                        <ArrowUpRight size={16} />
                      </strong>
                      <p>{excerpt(entry, query)}</p>
                      {entry.target && (
                        <small>
                          {entry.target.graphId} · {entry.target.type} · {entry.target.id}
                        </small>
                      )}
                    </button>
                  ))}
                </div>
              ) : (
                <div className="wiki-empty">
                  <Search size={24} />
                  <h4>No matching articles yet</h4>
                  <p>
                    Try a shorter term such as “mask”, “cache”, “RoPE”, “Hkv” or a component name.
                    Search includes authored model explanations and connections.
                  </p>
                  {suggestWikiQueries(query).length > 0 && (
                    <div className="wiki-search-suggestions">
                      <span>Did you mean</span>
                      {suggestWikiQueries(query).map((suggestion) => (
                        <button key={suggestion} onClick={() => setQuery(suggestion)}>
                          {suggestion}
                        </button>
                      ))}
                    </div>
                  )}
                  <button
                    className="button"
                    onClick={() => {
                      setQuery('')
                      setCategory('All topics')
                    }}
                  >
                    Clear filters
                  </button>
                </div>
              )}
              {results.length > limit && (
                <button className="button" onClick={() => setLimit(limit + 60)}>
                  Show more results
                </button>
              )}
            </div>
          ) : id === 'investigations' ? (
            <Investigations
              key={studyRevision}
              initial={study}
              onRead={read}
              onRestore={restoreStudy}
            />
          ) : id === 'glossary' ? (
            <WikiGlossary context={context} onRead={read} />
          ) : id === 'saved' ? (
            <WikiReadingCollection
              library={reading.library}
              entries={entryMap}
              modelId={context.doc.id}
              status={reading.status}
              onRead={read}
              onRemove={reading.remove}
            />
          ) : article ? (
            <div className="wiki-article-layout">
              <article className="wiki-article" aria-label={article.title}>
                <div className="wiki-breadcrumb">
                  <button onClick={() => read('home')}>Wiki</button>
                  <ChevronRight size={12} />
                  <span>{article.category}</span>
                </div>
                <header className="wiki-article-header">
                  <div className="wiki-article-meta">
                    <span>
                      {article.origin === 'document'
                        ? 'FROM YOUR MODEL DOCUMENT'
                        : article.origin === 'glossary'
                          ? 'GLOSSARY DEFINITION'
                          : 'CONCEPT GUIDE'}
                    </span>
                    <span>{Math.max(1, Math.ceil(words / 200))} min read</span>
                  </div>
                  <h3 tabIndex={-1}>{article.title}</h3>
                  {article.origin !== 'glossary' && activeMode !== 'introduction' && (
                    <p className="wiki-lead">{article.summary}</p>
                  )}
                  {perspective && (
                    <>
                      <div
                        className="wiki-reading-modes"
                        role="tablist"
                        aria-label="Explanation level"
                      >
                        {(
                          [
                            'introduction',
                            'technical',
                            ...(explorationFor[id] ? ['explore'] : []),
                          ] as ReadingMode[]
                        ).map((mode) => (
                          <button
                            key={mode}
                            role="tab"
                            id={`${prefix}-mode-${mode}`}
                            aria-controls={`${prefix}-reading-content`}
                            aria-selected={activeMode === mode}
                            tabIndex={activeMode === mode ? 0 : -1}
                            onClick={() => changeMode(mode)}
                            onKeyDown={(event) => {
                              if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key))
                                return
                              event.preventDefault()
                              const modes: ReadingMode[] = [
                                'introduction',
                                'technical',
                                ...(explorationFor[id] ? ['explore' as const] : []),
                              ]
                              const index = modes.indexOf(mode)
                              const next =
                                event.key === 'Home'
                                  ? modes[0]
                                  : event.key === 'End'
                                    ? modes.at(-1)!
                                    : modes[
                                        (index +
                                          (event.key === 'ArrowRight' ? 1 : modes.length - 1)) %
                                          modes.length
                                      ]
                              changeMode(next)
                              document.getElementById(`${prefix}-mode-${next}`)?.focus()
                            }}
                          >
                            {mode === 'introduction'
                              ? 'Introduction'
                              : mode === 'technical'
                                ? 'Technical detail'
                                : 'Interactive lab'}
                          </button>
                        ))}
                      </div>
                      <p className="wiki-mode-explanation">
                        {activeMode === 'introduction'
                          ? 'Purpose, mechanism and a second way to picture it. No equations required.'
                          : activeMode === 'technical'
                            ? 'Equations, axes, derivations, assumptions and implementation boundaries.'
                            : 'Change a labelled teaching example, follow the calculation and compare controlled interventions.'}
                      </p>
                    </>
                  )}
                  <div className="wiki-article-actions">
                    {['learning-loop', 'adam', 'adamw'].includes(id) && (
                      <button onClick={() => chooseInvestigation('training')}>
                        Train a small network
                      </button>
                    )}
                    <button
                      aria-label={saved ? 'Unsave article' : 'Save article'}
                      aria-pressed={saved}
                      onClick={() =>
                        reading.toggle(article, contentRef.current?.scrollTop ?? 0, activeMode)
                      }
                    >
                      <Bookmark size={14} />
                      {saved ? 'Saved' : 'Save article'}
                    </button>
                    {explorationFor[id] && (
                      <button
                        onClick={() => {
                          changeMode('explore')
                          requestAnimationFrame(() => jump('exploration'))
                        }}
                      >
                        Explore a worked example
                        <ArrowRight size={13} />
                      </button>
                    )}
                    {experimentEntry && context.onExperiment && (
                      <button
                        className="wiki-run-example"
                        onClick={() => {
                          rememberCurrent()
                          context.onExperiment!(experimentEntry)
                          onClose()
                        }}
                      >
                        <FlaskConical size={14} />
                        {context.currentGraphOnly
                          ? 'Inspect in this run'
                          : 'Open numerical example'}
                      </button>
                    )}
                  </div>
                  {!reading.status.startsWith('Saved') && (
                    <p className="wiki-reading-status" role="status">
                      {reading.status}
                    </p>
                  )}
                  {experimentEntry && context.onExperiment && (
                    <p className="wiki-experiment-scope">
                      {context.currentGraphOnly
                        ? 'Seeks to this operation in the existing captured run. Parameters and run settings stay unchanged.'
                        : 'Opens the separate tiny foundations profile with seeded, untrained parameters. It does not execute an imported model.'}
                    </p>
                  )}
                  {article.target && (
                    <div className="wiki-document-context">
                      <FileText size={15} />
                      <code>
                        {article.target.graphId} / {article.target.id}
                      </code>
                      {context.onNavigate && (
                        <button
                          onClick={() => {
                            context.onNavigate!(article.target!)
                            close()
                          }}
                        >
                          View on canvas
                          <ArrowUpRight size={13} />
                        </button>
                      )}
                    </div>
                  )}
                </header>
                {article.origin === 'document' && (
                  <>
                    <p className="wiki-authored-note">
                      This explanation belongs to <strong>{context.doc.name}</strong>. Linked
                      concept guides describe the supported foundations profile; an imported model
                      may use different mathematics.
                    </p>
                    {article.related.length > 0 && (
                      <div className="wiki-deeper">
                        <span className="eyebrow">GO DEEPER · PURPOSE, MATH & EXAMPLES</span>
                        {article.related.map((target) => {
                          const concept = entryMap.get(target)
                          return concept ? (
                            <button key={target} onClick={() => read(target)}>
                              <span>
                                <strong>{concept.title}</strong>
                                <small>{concept.summary}</small>
                              </span>
                              <ArrowRight size={15} />
                            </button>
                          ) : null
                        })}
                      </div>
                    )}
                  </>
                )}
                {article.prerequisites.length > 0 && (
                  <div className="wiki-prerequisites">
                    <span>Useful first</span>
                    <div>{article.prerequisites.map((target) => link(target))}</div>
                  </div>
                )}
                {article.origin === 'concept' && (
                  <WikiModelConnections
                    key={id}
                    matches={matches}
                    context={context}
                    onRead={read}
                    onNavigate={(entry) => {
                      context.onNavigate?.(entry.target!)
                      close()
                    }}
                  />
                )}
                <div
                  id={`${prefix}-reading-content`}
                  role={perspective ? 'tabpanel' : undefined}
                  aria-labelledby={perspective ? `${prefix}-mode-${activeMode}` : undefined}
                  className={`wiki-perspective-content is-${activeMode}`}
                >
                  {sections.map((section) => (
                    <section
                      key={section.id}
                      id={`${prefix}-${section.id}`}
                      className="wiki-section"
                    >
                      <h4>
                        {section.title}
                        <button
                          aria-label={`Jump to section: ${section.title}`}
                          onClick={() => jump(section.id)}
                        >
                          #
                        </button>
                      </h4>
                      {section.paragraphs.map((paragraph, i) => (
                        <p key={i}>{linked(paragraph)}</p>
                      ))}
                      {section.equation && <MathBlock tex={section.equation} />}
                      {section.bullets && (
                        <ul>
                          {section.bullets.map((item) => (
                            <li key={item}>{linked(item)}</li>
                          ))}
                        </ul>
                      )}
                    </section>
                  ))}
                  {explorationFor[id] && (
                    <div
                      id={`${prefix}-exploration`}
                      className="wiki-exploration-anchor"
                      hidden={activeMode !== 'explore'}
                    >
                      <WikiExploration key={id} articleId={id} context={context} onRead={read} />
                    </div>
                  )}
                  {questions.length > 0 && activeMode !== 'introduction' && (
                    <section id={`${prefix}-questions`} className="wiki-section wiki-reasoning">
                      <h4>Questions worth working through</h4>
                      <p>
                        Predict what will happen, then open the explanation and compare it with the
                        workbench.
                      </p>
                      {questions.map((question) => (
                        <details key={question.question}>
                          <summary>{question.question}</summary>
                          <p>{linked(question.answer)}</p>
                          <div>
                            <strong>Look for this</strong>
                            <p>{linked(question.investigate)}</p>
                          </div>
                        </details>
                      ))}
                    </section>
                  )}
                </div>
                {article.tensors && article.tensors.length > 0 && (
                  <section className="wiki-section" id={`${prefix}-tensors`}>
                    <h4>Read the tensor axes</h4>
                    <div className="wiki-tensors">
                      {article.tensors.map((tensor, i) => (
                        <div key={i}>
                          <span>{tensor.role}</span>
                          <strong>{tensor.name}</strong>
                          <code>{shapeLabel(tensor.shape, {}, true)}</code>
                          {context.symbols && (
                            <code className="wiki-resolved">
                              Current study: {shapeLabel(tensor.shape, context.symbols)}
                            </code>
                          )}
                          {tensor.description && <p>{linked(tensor.description)}</p>}
                          <div className="wiki-tensor-symbols">
                            {[
                              ...new Set(
                                tensor.shape.flatMap((axis) => axis.match(/[A-Za-z]+/g) ?? []),
                              ),
                            ].map((symbol) => {
                              const term = glossary.find((term) => term.symbol === symbol)
                              return term ? (
                                <button
                                  key={symbol}
                                  aria-label={`Explain tensor axis ${symbol}`}
                                  onClick={() => read(`glossary:${term.id}`)}
                                >
                                  {symbol}
                                  <small>{context.symbols?.[symbol] ?? 'definition'}</small>
                                </button>
                              ) : null
                            })}
                          </div>
                        </div>
                      ))}
                    </div>
                    <p>
                      Axes are ordered. A shape describes a logical tensor; it does not by itself
                      prove an equation correct or specify physical storage.{' '}
                      {link('tensors', 'Understand shape notation')}
                    </p>
                  </section>
                )}
                {article.experiment && (
                  <section className="wiki-experiment" id={`${prefix}-experiment`}>
                    <span className="eyebrow">TRY IT IN THE WORKBENCH</span>
                    <h4>{article.experiment.title}</h4>
                    <ol>
                      {article.experiment.steps.map((step) => (
                        <li key={step}>{linked(step)}</li>
                      ))}
                    </ol>
                    <p>
                      <Check size={15} />
                      <span>{linked(article.experiment.observation)}</span>
                    </p>
                  </section>
                )}
                {article.related.length > 0 && (
                  <section className="wiki-section wiki-related" id={`${prefix}-related`}>
                    <h4>
                      {article.origin === 'document'
                        ? 'Understand the underlying concepts'
                        : 'Keep following the idea'}
                    </h4>
                    <div>{article.related.map((target) => link(target))}</div>
                  </section>
                )}
                {article.references.length > 0 && (
                  <section className="wiki-section wiki-references" id={`${prefix}-sources`}>
                    <h4>
                      {article.origin === 'document'
                        ? 'Authored primary references'
                        : 'Primary sources & scope'}
                    </h4>
                    <p>
                      These sources establish the cited mechanisms. Worked numbers are synthetic
                      teaching examples unless the current workbench explicitly identifies a
                      recorded numerical run.
                    </p>
                    {article.references.map((source) => (
                      <a key={source.url} href={source.url} target="_blank" rel="noreferrer">
                        <BookOpen size={16} />
                        <span>
                          <strong>
                            {source.title}
                            <ArrowUpRight size={13} />
                          </strong>
                          <small>{source.note}</small>
                        </span>
                      </a>
                    ))}
                  </section>
                )}
                {learningPath && (
                  <nav className="wiki-path-navigation" aria-label="Learning path navigation">
                    <span>
                      {learningPath.title} · guide {pathIndex + 1} of {learningPath.articles.length}
                    </span>
                    <div>
                      {pathIndex > 0 &&
                        link(learningPath.articles[pathIndex - 1], 'Previous guide')}
                      {pathIndex < learningPath.articles.length - 1 &&
                        link(
                          learningPath.articles[pathIndex + 1],
                          `Next: ${entryMap.get(learningPath.articles[pathIndex + 1])?.title}`,
                        )}
                    </div>
                  </nav>
                )}
              </article>
              <aside className="wiki-outline" aria-label="On this page">
                <span className="eyebrow">ON THIS PAGE</span>
                {sections.map((section) => (
                  <button key={section.id} onClick={() => jump(section.id)}>
                    {section.title}
                  </button>
                ))}
                {article.tensors?.length ? (
                  <button onClick={() => jump('tensors')}>Tensor axes</button>
                ) : null}
                {explorationFor[id] && (
                  <button
                    onClick={() => {
                      changeMode('explore')
                      requestAnimationFrame(() => jump('exploration'))
                    }}
                  >
                    Interactive example
                  </button>
                )}
                {questions.length > 0 && activeMode !== 'introduction' && (
                  <button onClick={() => jump('questions')}>Questions & explanations</button>
                )}
                {article.experiment && <button onClick={() => jump('experiment')}>Try it</button>}
                {article.related.length > 0 && (
                  <button onClick={() => jump('related')}>Related concepts</button>
                )}
                {article.references.length > 0 && (
                  <button onClick={() => jump('sources')}>Primary sources</button>
                )}
              </aside>
            </div>
          ) : (
            <div className="wiki-home">
              <span className="eyebrow">AN EXPLANATION BEHIND EVERY OPERATION</span>
              <h3 tabIndex={-1}>
                See the model.
                <br />
                Understand the mechanism.
              </h3>
              <p className="wiki-lead">
                Follow an idea from its purpose to its equation, inspect a concrete example, then
                return to the model and test what you learned.
              </p>
              <div className="wiki-library-count">
                <BookOpen size={16} />
                {wikiArticles.length} detailed guides<span>·</span>
                {entries.filter((entry) => entry.origin === 'document').length} model entries
                <span>·</span>
                {glossary.length} definitions
              </div>
              <section className="wiki-current-study">
                <span className="eyebrow">CONNECTED TO WHAT YOU’RE VIEWING</span>
                <h4>{contextEntry?.title ?? context.graph.title}</h4>
                <p>{contextEntry?.summary}</p>
                <div>
                  {link(contextId, 'Explain this selection')}
                  {suggested.slice(0, 4).map((target) => link(target))}
                </div>
              </section>
              <InvestigationQuestions onChoose={chooseInvestigation} />
              <WikiReadingCollection
                compact
                library={reading.library}
                entries={entryMap}
                modelId={context.doc.id}
                status={reading.status}
                onRead={read}
                onRemove={reading.remove}
              />
              <h4 className="wiki-home-heading">Choose a learning path</h4>
              <div className="wiki-paths">
                {learningPaths.map((path, i) => (
                  <section key={path.title}>
                    <span className="wiki-path-number">0{i + 1}</span>
                    <h4>{path.title}</h4>
                    <p>{path.description}</p>
                    <ol>
                      {path.articles.map((target) => (
                        <li key={target}>{link(target)}</li>
                      ))}
                    </ol>
                  </section>
                ))}
              </div>
              <h4 className="wiki-home-heading">Explore the library</h4>
              <div className="wiki-category-grid">
                {wikiCategories.map((name) => (
                  <button
                    key={name}
                    onClick={() => {
                      setCategory(name)
                      contentRef.current?.scrollTo({ top: 0 })
                    }}
                  >
                    <span>{name}</span>
                    <small>
                      {wikiArticles.filter((entry) => entry.category === name).length} guides
                      <ArrowRight size={13} />
                    </small>
                  </button>
                ))}
              </div>
              <div className="wiki-home-footer">
                The wiki combines editorial concept guides with the current model’s own authored
                explanations. Reading help does not change your notes, layouts, configuration or
                model document.
              </div>
            </div>
          )}
        </div>
      </div>
    </Dialog>
  )
}
