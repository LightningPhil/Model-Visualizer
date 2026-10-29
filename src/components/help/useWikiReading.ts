import { useRef, useState } from 'react'
import {
  emptyLibrary,
  readLibrary,
  readingKey,
  serializeLibrary,
  WIKI_READING_KEY,
  type ReadingLibrary,
  type ReadingPlace,
} from '../../help/wiki/reading'
import type { WikiEntry } from '../../help/wiki/catalog'
import type { ReadingMode } from '../../help/wiki/perspective-types'

export function useWikiReading(modelId: string) {
  const [library, setLibrary] = useState(() => {
    try {
      return readLibrary(localStorage.getItem(WIKI_READING_KEY))
    } catch {
      return emptyLibrary()
    }
  })
  const current = useRef(library)
  const [status, setStatus] = useState('Saved on this browser, separately from model documents.')
  function update(next: ReadingLibrary) {
    current.current = next
    setLibrary(next)
    try {
      const encoded = serializeLibrary(localStorage.getItem(WIKI_READING_KEY), next)
      if (encoded === undefined)
        setStatus(
          'A newer reading-library version is preserved. Reading places apply to this session.',
        )
      else localStorage.setItem(WIKI_READING_KEY, encoded)
    } catch {
      setStatus('Browser storage is unavailable. Reading places apply to this session.')
    }
  }
  function place(entry: WikiEntry, scroll: number, mode?: ReadingMode): ReadingPlace {
    return {
      key: readingKey(entry.id, modelId),
      entryId: entry.id,
      title: entry.title,
      ...(entry.origin === 'document' ? { modelId } : {}),
      at: Date.now(),
      scroll,
      ...(mode ? { readingMode: mode } : {}),
    }
  }
  function remember(entry: WikiEntry, scroll: number, mode?: ReadingMode) {
    const next = place(entry, scroll, mode),
      old = current.current
    update({
      bookmarks: old.bookmarks.map((bookmark) =>
        bookmark.key === next.key ? { ...bookmark, ...next } : bookmark,
      ),
      recent: [next, ...old.recent.filter((p) => p.key !== next.key)].slice(0, 24),
    })
  }
  function toggle(entry: WikiEntry, scroll: number, mode?: ReadingMode) {
    const next = place(entry, scroll, mode),
      old = current.current
    update({
      ...old,
      bookmarks: old.bookmarks.some((p) => p.key === next.key)
        ? old.bookmarks.filter((p) => p.key !== next.key)
        : [next, ...old.bookmarks].slice(0, 100),
    })
  }
  function remove(key: string) {
    update({
      ...current.current,
      bookmarks: current.current.bookmarks.filter((p) => p.key !== key),
    })
  }
  return { library, status, remember, toggle, remove }
}
