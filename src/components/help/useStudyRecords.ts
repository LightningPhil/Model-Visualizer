import { useEffect, useState } from 'react'
import {
  STUDY_KEY,
  parseStudyLibrary,
  studyLibrarySchema,
  type StudyRecord,
} from '../../model/study/records'

function load() {
  try {
    const raw = localStorage.getItem(STUDY_KEY)
    return {
      library: raw ? parseStudyLibrary(raw) : studyLibrarySchema.parse({ version: 1, records: [] }),
      writable: true,
      message: '',
    }
  } catch {
    return {
      library: studyLibrarySchema.parse({ version: 1, records: [] }),
      writable: false,
      message:
        'Saved investigation data is unavailable or unsupported and has been preserved. Export any temporary records before closing.',
    }
  }
}
export function useStudyRecords() {
  const [state, setState] = useState(load)
  useEffect(() => {
    const refresh = (event: Event) => {
      if (event instanceof StorageEvent && event.key !== STUDY_KEY) return
      setState(load())
    }
    window.addEventListener('study-records-changed', refresh)
    window.addEventListener('storage', refresh)
    return () => {
      window.removeEventListener('study-records-changed', refresh)
      window.removeEventListener('storage', refresh)
    }
  }, [])
  function commit(records: StudyRecord[]) {
    try {
      // Re-read at mutation time so another open panel cannot lose a newer record.
      if (!state.writable) {
        setState((s) => ({ ...s, library: { ...s.library, records } }))
        return false
      }
      const fresh = load()
      if (!fresh.writable) {
        setState({ ...fresh, library: { ...fresh.library, records } })
        return false
      }
      const library = studyLibrarySchema.parse({ ...fresh.library, records })
      localStorage.setItem(STUDY_KEY, JSON.stringify(library))
      window.dispatchEvent(new Event('study-records-changed'))
      return true
    } catch {
      setState((s) => ({
        ...s,
        library: { ...s.library, records },
        writable: false,
        message:
          'Browser storage is unavailable. These records exist only in this session; export them to keep them.',
      }))
      return false
    }
  }
  const latest = () => (state.writable ? load().library.records : state.library.records)
  return {
    ...state,
    save: (record: StudyRecord) => {
      const records = latest()
      if (records.length >= 100)
        throw new Error(
          'The library holds 100 investigations. Export and remove records before saving another.',
        )
      return commit([record, ...records])
    },
    remove: (id: string) => commit(latest().filter((r) => r.id !== id)),
    importRecords: (records: StudyRecord[]) => {
      const old = latest()
      const added = records.filter((r) => !old.some((o) => o.id === r.id))
      if (old.length + added.length > 100)
        throw new Error('Import exceeds the 100-record library limit.')
      commit([...added, ...old])
      return added.length
    },
  }
}
