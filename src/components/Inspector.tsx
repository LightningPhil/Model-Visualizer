import {
  ArrowRight,
  ArrowUpRight,
  BookOpen,
  ChevronDown,
  CircleHelp,
  Copy,
  FileText,
  FlaskConical,
  Shapes,
} from 'lucide-react'
import { useState } from 'react'
import { ComponentIcon } from './Icons'
import { MathBlock } from './Math'
import {
  shapeLabel,
  type Component,
  type Connection,
  type ModelDocument,
  type ModelGraph,
  type Tensor,
} from '../model/schema'
import type { Selection } from './GraphCanvas'
import { numericalExampleFor } from '../model/numerics/examples'

type Props = {
  doc: ModelDocument
  graph: ModelGraph
  selection: Selection
  symbols: Record<string, number>
  symbolic: boolean
  layer: number
  onOpen: (id: string) => void
  onNotes: (notes: string) => void
  notify: (message: string) => void
  onNumerical: (entry: string) => void
}

function TensorCard({
  tensor,
  symbols,
  symbolic,
}: {
  tensor: Tensor
  symbols: Record<string, number>
  symbolic: boolean
}) {
  const [showAxes, setShowAxes] = useState(false)
  return (
    <div className="tensor-card">
      <button
        className="tensor-card-header"
        onClick={() => setShowAxes(!showAxes)}
        aria-expanded={showAxes}
      >
        <span>
          <small>{tensor.role}</small>
          <strong>{tensor.name}</strong>
        </span>
        <ChevronDown size={13} className={showAxes ? 'rotate' : ''} />
      </button>
      <code>{shapeLabel(tensor.shape, symbols, symbolic)}</code>
      {showAxes && (
        <div className="tensor-axes">
          {tensor.shape.map((axis, i) => (
            <span key={i}>
              {axis} <b>{shapeLabel([axis], symbols).slice(1, -1)}</b>
            </span>
          ))}
          {tensor.description && <p>{tensor.description}</p>}
        </div>
      )}
    </div>
  )
}

export function Inspector({
  doc,
  graph,
  selection,
  symbols,
  symbolic,
  layer,
  onOpen,
  onNotes,
  notify,
  onNumerical,
}: Props) {
  const [tab, setTab] = useState<'inspect' | 'notes'>('inspect')
  const component: Component | undefined =
    selection?.type === 'node' ? graph.nodes.find((n) => n.id === selection.id) : undefined
  const connection: Connection | undefined =
    selection?.type === 'edge' ? graph.edges.find((e) => e.id === selection.id) : undefined
  const title = component?.title ?? connection?.label ?? 'Explore the architecture'
  const equation = component?.equation ?? connection?.equation
  const example =
    numericalExampleFor(
      graph.id,
      component ?? graph.nodes.find((n) => n.id === connection?.target),
      layer,
    ) ??
    (connection
      ? numericalExampleFor(
          graph.id,
          graph.nodes.find((n) => n.id === connection.source),
          layer,
        )
      : undefined)

  async function copyEquation() {
    if (!equation) return
    try {
      await navigator.clipboard.writeText(equation)
      notify('LaTeX copied.')
    } catch {
      notify('Clipboard unavailable. Select and copy the equation from exported JSON.')
    }
  }

  return (
    <aside className="inspector" aria-label="Component inspector">
      <div className="inspector-tabs">
        <button className={tab === 'inspect' ? 'active' : ''} onClick={() => setTab('inspect')}>
          <Shapes size={14} />
          Inspector
        </button>
        <button className={tab === 'notes' ? 'active' : ''} onClick={() => setTab('notes')}>
          <FileText size={14} />
          Notebook
        </button>
      </div>
      {tab === 'notes' ? (
        <div className="notebook">
          <span className="eyebrow">YOUR WORKSPACE NOTES</span>
          <h2>Keep the connections.</h2>
          <p>Notes travel with your exported model file.</p>
          <textarea
            aria-label="Workspace notes"
            value={doc.ui.notes}
            onChange={(e) => onNotes(e.target.value)}
            maxLength={24000}
          />
          <span className="muted">Plain text · saved locally</span>
        </div>
      ) : (
        <div className="inspector-scroll">
          <div className={`inspector-symbol category-${component?.category ?? 'compute'}`}>
            {component ? (
              <ComponentIcon component={component} size={24} />
            ) : connection ? (
              <ArrowRight size={24} />
            ) : (
              <CircleHelp size={24} />
            )}
          </div>
          <div className="inspector-eyebrow">
            <span className="eyebrow">
              {component
                ? `${component.category} component`
                : connection
                  ? 'CONNECTION / OPERATION'
                  : 'INTERACTIVE MODEL ATLAS'}
            </span>
            {component?.repeat && <span className="quiet-badge">{symbols.L} layers</span>}
          </div>
          <h2>{title}</h2>
          {['block', 'attention', 'attention-heads', 'feedforward', 'cache'].includes(graph.id) && (
            <div className="layer-context">
              Layer ℓ = {layer} · independent parameters and state
            </div>
          )}
          <p className="inspector-description">
            {component?.description ??
              connection?.description ??
              'Select a component to see its mathematics and tensors. Select a connection to inspect the operation between components.'}
          </p>
          {component?.childGraph && (
            <button className="explore-button" onClick={() => onOpen(component.childGraph!)}>
              Explore inside
              <ArrowUpRight size={16} />
            </button>
          )}
          {example && (
            <button className="numerical-example-button" onClick={() => onNumerical(example)}>
              <FlaskConical size={15} />
              <span>
                Try a numerical example<small>Tiny reference model · untrained weights</small>
              </span>
              <ArrowUpRight size={14} />
            </button>
          )}
          {equation && (
            <section className="inspector-section">
              <div className="section-heading">
                <h3>THE MATHEMATICS</h3>
                <button
                  className="icon-button"
                  aria-label="Copy LaTeX"
                  data-tooltip-detail="Copy LaTeX"
                  onClick={() => void copyEquation()}
                >
                  <Copy size={13} />
                </button>
              </div>
              <MathBlock tex={equation} />
            </section>
          )}
          {component && component.tensors.length > 0 && (
            <section className="inspector-section">
              <div className="section-heading">
                <h3>TENSOR DIMENSIONS</h3>
                <span>{symbolic ? 'symbolic' : 'resolved'}</span>
              </div>
              <div className="tensor-list">
                {component.tensors.map((tensor, i) => (
                  <TensorCard
                    key={`${component.id}-${i}`}
                    tensor={tensor}
                    symbols={symbols}
                    symbolic={symbolic}
                  />
                ))}
              </div>
            </section>
          )}
          {connection && (
            <>
              <section className="inspector-section">
                <div className="section-heading">
                  <h3>OPERATION</h3>
                </div>
                <p>{connection.operation}</p>
                <div className="connection-path">
                  <span>{graph.nodes.find((n) => n.id === connection.source)?.title}</span>
                  <ArrowRight size={13} style={{ transform: 'rotate(90deg)' }} />
                  <span>{graph.nodes.find((n) => n.id === connection.target)?.title}</span>
                </div>
              </section>
              {connection.shape.length > 0 && (
                <section className="inspector-section">
                  <h3>RESULT / CARRIED TENSOR</h3>
                  <div className="tensor-card">
                    <code>{shapeLabel(connection.shape, symbols, symbolic)}</code>
                    <small>{shapeLabel(connection.shape, symbols, !symbolic)}</small>
                  </div>
                </section>
              )}
            </>
          )}
          {component && component.notes.length > 0 && (
            <section className="inspector-section reading-notes">
              <h3>WORTH UNDERSTANDING</h3>
              {component.notes.map((note, i) => (
                <p key={i}>{note}</p>
              ))}
            </section>
          )}
          {component && component.sourceIds.length > 0 && (
            <section className="inspector-section">
              <h3>GO TO THE SOURCE</h3>
              {component.sourceIds.map((id) => {
                const source = doc.sources.find((s) => s.id === id)!
                return (
                  <a
                    key={id}
                    className="source-link"
                    href={source.url}
                    target="_blank"
                    rel="noreferrer"
                  >
                    <BookOpen size={13} />
                    <span>{source.title}</span>
                    <ArrowUpRight size={12} />
                  </a>
                )
              })}
            </section>
          )}
          {!component && !connection && (
            <section className="inspector-section">
              <h3>READ THE SHAPES</h3>
              <dl className="symbol-key">
                {Object.entries({
                  B: 'Batch size',
                  T: 'Current query positions',
                  S: 'Absolute sequence extent',
                  R: 'Resident cache positions',
                  P: 'Retained prefix before call',
                  U: 'Working keys: prefix + new',
                  D: 'Hidden features',
                  Hq: 'Query heads',
                  Hkv: 'Key / value heads',
                  Dh: 'Features per head',
                  F: 'Feed-forward width',
                  V: 'Vocabulary size',
                  L: 'Decoder layers',
                }).map(([axis, description]) => (
                  <div key={axis}>
                    <dt>{axis}</dt>
                    <dd>{description}</dd>
                  </div>
                ))}
              </dl>
            </section>
          )}
        </div>
      )}
      <div className="inspector-bottom">
        <span className="tiny-square" />
        SHAPES FOLLOW YOUR CONFIGURATION
      </div>
    </aside>
  )
}
