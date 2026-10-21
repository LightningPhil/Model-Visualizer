import {
  createContext,
  useContext,
  useEffect,
  useId,
  useLayoutEffect,
  useRef,
  useState,
  type ReactNode,
} from 'react'
import { createPortal } from 'react-dom'
import { HelpCircle, Timer } from 'lucide-react'
import { Dialog } from '../Dialog'
import { HELP_TARGETS, resolveHelp } from './resolveHelp'
import {
  defaultHelpPreferences,
  HELP_PREFERENCES_KEY,
  positionHelp,
  readHelpPreferences,
  serializeHelpPreferences,
  type HelpPreferences,
} from '../../help/preferences'
import type { HelpContent } from '../../help/catalog'

const HelpContext = createContext<{ openSettings: (trigger: HTMLElement) => void } | null>(null)
type Target = { element: HTMLElement; region: HTMLElement; mode: 'hover' | 'focus' }
type Visible = { target: Target; content: HelpContent }

function loadPreferences() {
  try {
    return readHelpPreferences(localStorage.getItem(HELP_PREFERENCES_KEY))
  } catch {
    return defaultHelpPreferences
  }
}

function targetAt(node: EventTarget | null): Omit<Target, 'mode'> | undefined {
  if (!(node instanceof Element) || node.closest('.help-tooltip')) return
  let element =
    node.closest<HTMLElement>(HELP_TARGETS) ??
    node.closest<HTMLElement>('[data-help],[data-tooltip-detail]')
  const label = node.closest('label')
  if ((!element || element instanceof HTMLLabelElement) && label?.control instanceof HTMLElement)
    element = label.control
  if (!element || element.closest('[hidden],[inert]')) return
  const containingLabel = element.closest('label')
  return { element, region: containingLabel ?? element }
}

function HelpCard({ visible, id, delayMs }: { visible: Visible; id: string; delayMs: number }) {
  const ref = useRef<HTMLDivElement>(null)
  useLayoutEffect(() => {
    const element = ref.current!
    const anchor = visible.target.element
    element.showPopover()
    const box = element.getBoundingClientRect()
    const point = positionHelp(anchor.getBoundingClientRect(), box, {
      width: innerWidth,
      height: innerHeight,
    })
    element.style.left = `${point.x}px`
    element.style.top = `${point.y}px`
    const describedBy = anchor.getAttribute('aria-describedby')?.split(/\s+/).filter(Boolean) ?? []
    anchor.setAttribute('aria-describedby', [...new Set([...describedBy, id])].join(' '))
    return () => {
      if (element.matches(':popover-open')) element.hidePopover()
      const remaining = (anchor.getAttribute('aria-describedby')?.split(/\s+/) ?? []).filter(
        (value) => value && value !== id,
      )
      if (remaining.length) anchor.setAttribute('aria-describedby', remaining.join(' '))
      else anchor.removeAttribute('aria-describedby')
    }
  }, [visible, id])
  const content = visible.content
  return createPortal(
    <div ref={ref} id={id} role="tooltip" popover="manual" className="help-tooltip">
      <div className="help-tooltip-heading">
        <HelpCircle size={16} />
        <strong>{content.title}</strong>
      </div>
      {content.paragraphs.map((p, i) => (
        <p key={i}>{p}</p>
      ))}
      {content.choices && (
        <dl>
          {content.choices.map((choice) => (
            <div key={choice.label}>
              <dt>{choice.label}</dt>
              <dd>{choice.description}</dd>
            </div>
          ))}
        </dl>
      )}
      {content.sources && (
        <p className="help-tooltip-sources">
          Primary sources: {content.sources.join('; ')}. Links and equations are in the field guide
          / inspector.
        </p>
      )}
      <div className="help-tooltip-footer">
        <span>
          <kbd>Esc</kbd> to dismiss
        </span>
        <span>
          <Timer size={11} />
          {(delayMs / 1000).toLocaleString()} s hover
        </span>
      </div>
    </div>,
    visible.target.element.closest('dialog[open]') ?? document.body,
  )
}

export function TooltipProvider({ children }: { children: ReactNode }) {
  const [preferences, setPreferences] = useState(loadPreferences)
  const [visible, setVisible] = useState<Visible>()
  const [settingsParent, setSettingsParent] = useState<HTMLElement>()
  const [saveStatus, setSaveStatus] = useState(
    'Changes save on this browser, separately from model workspaces.',
  )
  const [delayDraft, setDelayDraft] = useState(String(preferences.delayMs / 1000))
  const tooltipId = useId()
  const cancelRef = useRef<() => void>(() => {})
  useEffect(() => {
    let candidate: Target | undefined
    let pending: ReturnType<typeof setTimeout> | undefined
    let closing: ReturnType<typeof setTimeout> | undefined
    let shown = false
    let keyboard = false
    let pointer: { x: number; y: number } | undefined
    const clearTimers = () => {
      clearTimeout(pending)
      clearTimeout(closing)
      pending = undefined
      closing = undefined
    }
    const hide = () => {
      clearTimers()
      shown = false
      setVisible(undefined)
    }
    const reset = () => {
      hide()
      candidate = undefined
    }
    const stillPresent = (target: Target) =>
      target.element.isConnected &&
      (!target.element.closest('dialog') || target.element.closest('dialog')!.open) &&
      (target.mode === 'focus'
        ? document.activeElement === target.element
        : target.region.matches(':hover'))
    const begin = (target: Target) => {
      clearTimeout(closing)
      if (candidate?.element === target.element && candidate.mode === target.mode) return
      hide()
      candidate = target
      if (!preferences.enabled) return
      pending = setTimeout(() => {
        pending = undefined
        if (candidate !== target || !stillPresent(target)) return
        shown = true
        setVisible({ target, content: resolveHelp(target.element) })
      }, preferences.delayMs)
    }
    const over = (event: PointerEvent) => {
      if (event.pointerType === 'touch' || event.buttons !== 0) return
      pointer = { x: event.clientX, y: event.clientY }
      if (event.target instanceof Element && event.target.closest('.help-tooltip')) {
        clearTimeout(closing)
        return
      }
      const target = targetAt(event.target)
      if (target) begin({ ...target, mode: 'hover' })
    }
    const move = (event: PointerEvent) => {
      if (event.pointerType === 'touch' || event.buttons !== 0) return
      pointer = { x: event.clientX, y: event.clientY }
      if (!candidate) {
        const target = targetAt(event.target)
        if (target) begin({ ...target, mode: 'hover' })
      }
    }
    const out = (event: PointerEvent) => {
      if (candidate?.mode !== 'hover') return
      if (event.relatedTarget instanceof Node && candidate.region.contains(event.relatedTarget))
        return
      if (event.relatedTarget instanceof Element && event.relatedTarget.closest('.help-tooltip')) {
        clearTimeout(pending)
        return
      }
      clearTimeout(pending)
      pending = undefined
      if (!shown) {
        candidate = undefined
        return
      }
      // A small close grace lets the pointer cross the gutter to read a card.
      // Pending cards are cancelled immediately; no grace is used to open one.
      clearTimeout(closing)
      closing = setTimeout(reset, 120)
    }
    const focus = (event: FocusEvent) => {
      if (!keyboard) return
      const target = targetAt(event.target)
      if (target) begin({ ...target, mode: 'focus' })
    }
    const blur = () => {
      if (candidate?.mode === 'focus') reset()
    }
    const down = (event: PointerEvent) => {
      keyboard = false
      if (!(event.target instanceof Element && event.target.closest('.help-tooltip'))) hide()
    }
    const key = (event: KeyboardEvent) => {
      if (event.key === 'Tab') {
        keyboard = true
        reset()
      }
      if (event.key === 'Escape') {
        if (shown) {
          event.preventDefault()
          event.stopPropagation()
        }
        hide()
      }
      if (event.key === 'Enter' || event.key === ' ') hide()
    }
    const scroll = (event: Event) => {
      if (event.target instanceof Element && event.target.closest('.help-tooltip')) return
      const previous = candidate
      const active = shown || pending !== undefined
      reset()
      if (!active) return
      // Scrolling invalidates the old hover. If a control remains under the
      // pointer, it gets a complete new delay measured from this scroll event.
      if (previous?.mode === 'focus' && document.activeElement === previous.element) begin(previous)
      else if (pointer && !keyboard) {
        const target = targetAt(document.elementFromPoint(pointer.x, pointer.y))
        if (target) begin({ ...target, mode: 'hover' })
      }
    }
    const mutation = new MutationObserver(() => {
      if (
        candidate &&
        (!candidate.element.isConnected ||
          (candidate.element.closest('dialog') && !candidate.element.closest('dialog')!.open))
      )
        reset()
    })
    mutation.observe(document.body, { childList: true, subtree: true })
    cancelRef.current = reset
    document.addEventListener('pointerover', over, true)
    document.addEventListener('pointermove', move, true)
    document.addEventListener('pointerout', out, true)
    document.addEventListener('pointerdown', down, true)
    document.addEventListener('focusin', focus, true)
    document.addEventListener('focusout', blur, true)
    document.addEventListener('keydown', key, true)
    document.addEventListener('scroll', scroll, true)
    document.addEventListener('change', hide, true)
    document.addEventListener('input', hide, true)
    document.addEventListener('visibilitychange', reset)
    window.addEventListener('resize', reset)
    window.addEventListener('blur', reset)
    return () => {
      clearTimers()
      mutation.disconnect()
      document.removeEventListener('pointerover', over, true)
      document.removeEventListener('pointermove', move, true)
      document.removeEventListener('pointerout', out, true)
      document.removeEventListener('pointerdown', down, true)
      document.removeEventListener('focusin', focus, true)
      document.removeEventListener('focusout', blur, true)
      document.removeEventListener('keydown', key, true)
      document.removeEventListener('scroll', scroll, true)
      document.removeEventListener('change', hide, true)
      document.removeEventListener('input', hide, true)
      document.removeEventListener('visibilitychange', reset)
      window.removeEventListener('resize', reset)
      window.removeEventListener('blur', reset)
    }
  }, [preferences])

  function update(change: Partial<HelpPreferences>) {
    const next = { ...preferences, ...change }
    cancelRef.current()
    setPreferences(next)
    try {
      const json = serializeHelpPreferences(localStorage.getItem(HELP_PREFERENCES_KEY), next)
      if (json === undefined)
        setSaveStatus(
          'A newer saved preference version is preserved. These choices apply to this session.',
        )
      else {
        localStorage.setItem(HELP_PREFERENCES_KEY, json)
        setSaveStatus('Saved on this browser, separately from model workspaces.')
      }
    } catch {
      setSaveStatus('Browser storage is unavailable. These choices apply to this session.')
    }
  }
  function openSettings(trigger: HTMLElement) {
    cancelRef.current()
    setDelayDraft(String(preferences.delayMs / 1000))
    setSettingsParent(trigger.closest('dialog[open]') ?? document.body)
  }
  return (
    <HelpContext.Provider value={{ openSettings }}>
      {children}
      {visible && <HelpCard visible={visible} id={tooltipId} delayMs={preferences.delayMs} />}
      {settingsParent &&
        createPortal(
          <Dialog
            title="Tooltip timing & help."
            className="help-preferences"
            onClose={() => setSettingsParent(undefined)}
          >
            <p className="modal-description">Detailed explanations, at your pace.</p>
            <div className="help-preferences-body">
              <label className="help-enable">
                <input
                  type="checkbox"
                  aria-label="Show explanatory tooltips"
                  checked={preferences.enabled}
                  onChange={(e) => update({ enabled: e.target.checked })}
                />
                Show explanatory tooltips
              </label>
              <label className="help-delay-label" htmlFor="tooltip-delay">
                Hover delay <span>seconds</span>
              </label>
              <input
                id="tooltip-delay"
                aria-label="Tooltip delay"
                type="number"
                min={0}
                max={5}
                step={0.1}
                value={delayDraft}
                onChange={(e) => {
                  const text = e.target.value
                  setDelayDraft(text)
                  const seconds = Number(text)
                  if (text.trim() && Number.isFinite(seconds) && seconds >= 0 && seconds <= 5)
                    update({ delayMs: Math.round(seconds * 1000) })
                }}
                onBlur={() => setDelayDraft(String(preferences.delayMs / 1000))}
              />
              <p>
                Default: <strong>0.7 seconds</strong>. The pointer must stay over the same control
                for the whole delay. Moving within it is fine; leaving it starts the next hover from
                zero.
              </p>
              <p>
                Clicking, scrolling or pressing Escape cancels a pending tooltip. Keyboard users can
                focus a control to read the same explanation. Escape dismisses the explanation
                before closing its dialog.
              </p>
              <button
                className="button"
                onClick={() => {
                  update(defaultHelpPreferences)
                  setDelayDraft('0.7')
                }}
              >
                Restore tooltip defaults
              </button>
              <p className="help-save-status" role="status">
                {saveStatus}
              </p>
            </div>
            <div className="modal-footer">
              <span>0–5 seconds · device preference</span>
              <button className="button export-button" onClick={() => setSettingsParent(undefined)}>
                Done
              </button>
            </div>
          </Dialog>,
          settingsParent,
        )}
    </HelpContext.Provider>
  )
}

export function TooltipSettingsButton() {
  const context = useContext(HelpContext)
  return (
    <button
      className="icon-button tooltip-settings-trigger"
      aria-label="Tooltip settings"
      onClick={(event) => context?.openSettings(event.currentTarget)}
    >
      <HelpCircle size={17} />
    </button>
  )
}
