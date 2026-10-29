import {
  createContext,
  lazy,
  Suspense,
  useCallback,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from 'react'
import { createPortal } from 'react-dom'
import { Dialog } from '../Dialog'
import type { WikiContext } from '../../help/wiki/catalog'
import './wiki-launch.css'

const Wiki = lazy(() => import('./Wiki').then((module) => ({ default: module.Wiki })))

const Context = createContext<(context: WikiContext) => void>(() => {})
const FollowContext = createContext<(context: WikiContext) => void>(() => {})

/** One reader shared by the architecture and both numerical workbenches.
 * Help is session UI state: it never changes the portable model document. */
export function WikiProvider({ children }: { children: ReactNode }) {
  const [context, setContext] = useState<WikiContext>()
  const [opening, setOpening] = useState(0)
  const [host, setHost] = useState<Element>(document.body)
  const [docked, setDocked] = useState(false)
  const [wide, setWide] = useState(() => window.innerWidth >= 1100)
  const effectiveDocked = docked && wide
  useEffect(() => {
    const resize = () => setWide(window.innerWidth >= 1100)
    window.addEventListener('resize', resize)
    return () => window.removeEventListener('resize', resize)
  }, [])
  useEffect(() => {
    if (!context || !effectiveDocked) return
    const target = host === document.body ? document.querySelector('.app') : host
    target?.classList.add('wiki-docked-host')
    return () => target?.classList.remove('wiki-docked-host')
  }, [context !== undefined, effectiveDocked, host])
  const open = useCallback((next: WikiContext) => {
    setOpening((r) => r + 1)
    setHost(
      Array.from(document.querySelectorAll('dialog[open]'))
        .filter((el) => !el.classList.contains('wiki-modal'))
        .at(-1) ?? document.body,
    )
    setContext(next)
  }, [])
  useEffect(() => {
    if (!context || host === document.body) return
    const observer = new MutationObserver(() => {
      if (!host.isConnected) setContext(undefined)
    })
    observer.observe(document.body, { childList: true, subtree: true })
    return () => observer.disconnect()
  }, [context !== undefined, host])
  const follow = useCallback(
    (next: WikiContext) => {
      if (effectiveDocked)
        setContext((old) =>
          old && old.mode === next.mode && old.doc.id === next.doc.id
            ? { ...next, initialArticle: old.initialArticle }
            : old,
        )
    },
    [effectiveDocked],
  )
  return (
    <Context.Provider value={open}>
      <FollowContext.Provider value={follow}>
        {children}
        {context &&
          createPortal(
            <Suspense
              fallback={
                <Dialog title="Opening the model wiki…" onClose={() => setContext(undefined)}>
                  <p className="modal-description">Preparing the local reference library.</p>
                </Dialog>
              }
            >
              <Wiki
                key={opening}
                context={context}
                onClose={() => setContext(undefined)}
                docked={effectiveDocked}
                canDock={wide}
                onDock={() => setDocked(!effectiveDocked)}
              />
            </Suspense>,
            host,
          )}
      </FollowContext.Provider>
    </Context.Provider>
  )
}

export const useWiki = () => useContext(Context)
export function useWikiFollow(context: WikiContext) {
  const follow = useContext(FollowContext)
  // Callbacks are refreshed without subscribing to layout changes or the reader itself.
  const { doc, graph, selection, mode, symbols } = context
  useEffect(
    () => follow(context),
    [
      follow,
      doc,
      graph.id,
      selection?.id,
      selection?.type,
      mode,
      symbols?.T,
      symbols?.S,
      context.inspection?.name,
      context.inspection?.index,
      context.inspection?.value,
    ],
  )
}
