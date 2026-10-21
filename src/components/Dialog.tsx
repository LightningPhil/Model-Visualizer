import { useEffect, useRef, type ReactNode } from 'react'
import { X } from 'lucide-react'

export function Dialog({
  title,
  onClose,
  children,
  className = '',
}: {
  title: string
  onClose: () => void
  children: ReactNode
  className?: string
}) {
  const ref = useRef<HTMLDialogElement>(null)
  useEffect(() => {
    const element = ref.current!
    element.showModal()
    return () => element.close()
  }, [])
  return (
    <dialog
      ref={ref}
      className={`modal ${className}`}
      onCancel={(event) => {
        event.stopPropagation()
        onClose()
      }}
      onClick={(event) => {
        if (event.target === event.currentTarget) onClose()
      }}
      aria-label={title}
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
