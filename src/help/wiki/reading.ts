export const WIKI_READING_KEY = 'model-visualizer.wiki-reading.v1'
export type ReadingPlace = {
  key: string
  entryId: string
  title: string
  modelId?: string
  at: number
  scroll: number
  [key: string]: unknown
}
export type ReadingLibrary = { bookmarks: ReadingPlace[]; recent: ReadingPlace[] }
export const emptyLibrary = (): ReadingLibrary => ({ bookmarks: [], recent: [] })
export const readingKey = (entryId: string, modelId: string) =>
  entryId.startsWith('document:') ? `${modelId}|${entryId}` : entryId

function places(value: unknown): ReadingPlace[] {
  if (!Array.isArray(value)) return []
  const seen = new Set<string>()
  return value
    .flatMap((place) => {
      if (
        !place ||
        typeof place !== 'object' ||
        typeof place.key !== 'string' ||
        place.key.length > 1000 ||
        typeof place.entryId !== 'string' ||
        place.entryId.length > 800 ||
        typeof place.title !== 'string' ||
        place.title.length > 240 ||
        (place.modelId !== undefined && typeof place.modelId !== 'string') ||
        !Number.isFinite(place.at) ||
        !Number.isFinite(place.scroll) ||
        place.scroll < 0 ||
        seen.has(place.key)
      )
        return []
      seen.add(place.key)
      return [{ ...place, scroll: Math.min(place.scroll, 1e7) } as ReadingPlace]
    })
    .slice(0, 100)
}
export function readLibrary(raw: string | null): ReadingLibrary {
  try {
    const value = JSON.parse(raw ?? 'null')
    return value?.version === 1
      ? { bookmarks: places(value.bookmarks), recent: places(value.recent) }
      : emptyLibrary()
  } catch {
    return emptyLibrary()
  }
}
export function serializeLibrary(raw: string | null, library: ReadingLibrary): string | undefined {
  let existing: Record<string, unknown> = {}
  try {
    const value = JSON.parse(raw ?? 'null')
    if (value && typeof value === 'object' && !Array.isArray(value)) {
      existing = value
      if (value.version !== undefined && value.version !== 1) return undefined
    }
  } catch {
    /* Invalid JSON has no recoverable structure. */
  }
  return JSON.stringify({
    ...existing,
    version: 1,
    bookmarks: places(library.bookmarks),
    recent: places(library.recent),
  })
}
