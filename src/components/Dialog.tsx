import { useEffect, useRef, type KeyboardEventHandler, type ReactNode } from 'react'
import { X } from 'lucide-react'

export function Dialog({
  title,
  onClose,
  children,
  className = '',
  onKeyDown,
  modal = true,
}: {
  title: string
  onClose: () => void
  children: ReactNode
  className?: string
  onKeyDown?: KeyboardEventHandler<HTMLDialogElement>
  modal?: boolean
}) {
  const ref = useRef<HTMLDialogElement>(null)
  useEffect(() => {
    const element = ref.current!
    if (modal) element.showModal()
    else element.show()
    return () => element.close()
  }, [modal])
  return (
    <dialog
      ref={ref}
      className={`modal ${className}`}
      onKeyDown={onKeyDown}
      onCancel={(event) => {
        event.stopPropagation()
        onClose()
      }}
      onClick={(event) => {
        if (event.target === event.currentTarget) onClose()
      }}
      aria-label={title}
      aria-modal={modal}
    >
      <div className="modal-heading">
        <div>
          <span className="eyebrow">MODEL VISUALIZER</span>
          <h2>{title}</h2>
        </div>
        <button className="icon-button" aria-label="Close dialog" onClick={onClose}>
          <X size={20} />
        </button>
      </div>
      {children}
    </dialog>
  )
}
