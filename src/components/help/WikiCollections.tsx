import { useMemo, useState } from 'react'
import { ArrowRight, Bookmark, Clock, Search, X } from 'lucide-react'
import { glossary } from '../../help/wiki/glossary'
import { normalizeWikiText, type WikiContext, type WikiEntry } from '../../help/wiki/catalog'
import type { ReadingLibrary } from '../../help/wiki/reading'

export function WikiGlossary({
  context,
  onRead,
}: {
  context: WikiContext
  onRead: (id: string) => void
}) {
  const [query, setQuery] = useState('')
  const terms = useMemo(
    () =>
      glossary.filter((term) =>
        normalizeWikiText([term.term, term.definition, ...(term.aliases ?? [])].join(' ')).includes(
          normalizeWikiText(query),
        ),
      ),
    [query],
  )
  return (
    <div className="wiki-glossary-page">
      <span className="eyebrow">A VOCABULARY FOR READING THE MODEL</span>
      <h3 tabIndex={-1}>Terms, symbols & axes</h3>
      <p className="wiki-lead">
        Definitions connect to detailed guides and primary sources. Symbol values below come from
        the study that opened help; scratch examples use their own labelled numbers.
      </p>
      <label className="wiki-glossary-search">
        <Search size={16} />
        <input
          aria-label="Search glossary"
          placeholder="Find a term or symbol…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
      </label>
      <p className="wiki-collection-count" role="status">
        {terms.length} definitions
      </p>
      <dl className="wiki-glossary-list">
        {terms.map((term) => (
          <div key={term.id}>
            <dt>
              <button onClick={() => onRead(`glossary:${term.id}`)}>
                {term.term}
                <ArrowRight size={13} />
              </button>
              {term.symbol && context.symbols?.[term.symbol] !== undefined && (
                <code>Current study: {context.symbols[term.symbol].toLocaleString()}</code>
              )}
            </dt>
            <dd>
              {term.definition}
              <button onClick={() => onRead(term.article)}>
                Read the mechanism
                <ArrowRight size={12} />
              </button>
            </dd>
          </div>
        ))}
      </dl>
      {!terms.length && (
        <p>No matching definition. Try “head”, “position”, “cache” or an axis such as Hkv.</p>
      )}
    </div>
  )
}

export function WikiReadingCollection({
  library,
  entries,
  modelId,
  status,
  onRead,
  onRemove,
  compact = false,
}: {
  library: ReadingLibrary
  entries: Map<string, WikiEntry>
  modelId: string
  status: string
  onRead: (id: string, resume?: boolean) => void
  onRemove: (key: string) => void
  compact?: boolean
}) {
  const available = (place: ReadingLibrary['recent'][number]) =>
    (!place.modelId || place.modelId === modelId) && entries.has(place.entryId)
  const recent = compact ? library.recent.filter(available).slice(0, 4) : library.recent
  if (compact && !recent.length) return null
  return (
    <div className={compact ? 'wiki-reading-collection compact' : 'wiki-reading-collection'}>
      {!compact && (
        <>
          <span className="eyebrow">YOUR READING LIBRARY</span>
          <h3 tabIndex={-1}>Keep the useful connections.</h3>
          <p className="wiki-lead">
            Save a guide or return to a recent reading position. Model-specific entries stay
            associated with their source document.
          </p>
          <h4>
            <Bookmark size={16} />
            Saved articles
          </h4>
          <div className="wiki-saved-list">
            {library.bookmarks.map((place) => (
              <div key={place.key}>
                <button disabled={!available(place)} onClick={() => onRead(place.entryId, true)}>
                  <strong>{entries.get(place.entryId)?.title ?? place.title}</strong>
                  <small>
                    {available(place)
                      ? 'Resume at your reading position'
                      : 'Open the original model to read this entry'}
                  </small>
                </button>
                <button
                  aria-label={`Remove saved article: ${place.title}`}
                  onClick={() => onRemove(place.key)}
                >
                  <X size={14} />
                </button>
              </div>
            ))}
          </div>
          {!library.bookmarks.length && (
            <p className="wiki-empty-library">
              Use Save article in a guide’s header to collect explanations worth revisiting.
            </p>
          )}
        </>
      )}
      <h4>
        <Clock size={16} />
        {compact ? 'Continue reading' : 'Recently read'}
      </h4>
      <div className="wiki-recent-list">
        {recent.map((place) => (
          <button
            key={place.key}
            disabled={!available(place)}
            onClick={() => onRead(place.entryId, true)}
          >
            <span>
              <strong>{available(place) ? entries.get(place.entryId)?.title : place.title}</strong>
              <small>
                {available(place)
                  ? place.scroll > 20
                    ? 'Resume where you stopped'
                    : 'Read again'
                  : 'Belongs to another model document'}
              </small>
            </span>
            <ArrowRight size={14} />
          </button>
        ))}
      </div>
      {!recent.length && <p className="wiki-empty-library">Articles appear here as you read.</p>}
      {!compact && (
        <p className="wiki-reading-status" role="status">
          {status}
        </p>
      )}
    </div>
  )
}

export function WikiModelConnections({
  matches,
  context,
  onRead,
  onNavigate,
}: {
  matches: WikiEntry[]
  context: WikiContext
  onRead: (id: string) => void
  onNavigate: (entry: WikiEntry) => void
}) {
  const [expanded, setExpanded] = useState(false)
  if (!matches.length) return null
  return (
    <details className="wiki-model-connections">
      <summary>
        Find this mechanism in {context.currentGraphOnly ? 'the active run' : 'your model'}
        <span>
          {matches.length} match{matches.length === 1 ? '' : 'es'}
        </span>
      </summary>
      <p>
        These links use the existing graph IDs and descriptions. Concept guides describe foundations
        mathematics; imported architectures can differ.
      </p>
      {matches.slice(0, expanded ? 30 : 4).map((entry) => (
        <div key={entry.id}>
          <button onClick={() => onRead(entry.id)}>
            <strong>{entry.title}</strong>
            <small>
              {entry.target!.graphId} · {entry.target!.id}
            </small>
          </button>
          {context.onNavigate && (
            <button aria-label={`Show ${entry.title} on canvas`} onClick={() => onNavigate(entry)}>
              Show
              <ArrowRight size={12} />
            </button>
          )}
        </div>
      ))}
      {matches.length > 4 && (
        <button className="wiki-show-matches" onClick={() => setExpanded(!expanded)}>
          {expanded ? 'Show fewer matches' : `Show ${Math.min(30, matches.length)} matches`}
        </button>
      )}
      {expanded && matches.length > 30 && (
        <p>The first 30 matches are shown. Use model-document search for another exact instance.</p>
      )}
    </details>
  )
}
