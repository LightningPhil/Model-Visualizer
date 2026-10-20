import { parseDocument, type ModelDocument } from './schema'

const KEY = 'model-visualizer.workspace.v1'

export function loadWorkspace(): ModelDocument | null {
  try {
    const saved = localStorage.getItem(KEY)
    return saved ? parseDocument(JSON.parse(saved)) : null
  } catch {
    return null
  }
}

export function saveWorkspace(document: ModelDocument): boolean {
  try {
    localStorage.setItem(KEY, JSON.stringify(document))
    return true
  } catch {
    return false
  }
}

export function exportDocument(document: ModelDocument) {
  const blob = new Blob([`${JSON.stringify(document, null, 2)}\n`], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const link = window.document.createElement('a')
  link.href = url
  link.download = `${document.id}.model.json`
  link.click()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}

export async function importDocument(file: File) {
  if (file.size > 3 * 1024 * 1024) throw new Error('Model files must be smaller than 3 MiB.')
  return parseDocument(JSON.parse(await file.text()))
}
