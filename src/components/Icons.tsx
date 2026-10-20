import {
  Binary,
  Boxes,
  Braces,
  CircleDot,
  Database,
  GitBranch,
  Layers3,
  Network,
  Sigma,
  Sparkles,
  TextCursorInput,
  Waves,
  Workflow,
} from 'lucide-react'
import type { Component } from '../model/schema'

export const graphIcons: Record<string, typeof Network> = {
  overview: Workflow,
  tokenization: TextCursorInput,
  block: Layers3,
  attention: Network,
  cache: Database,
  feedforward: GitBranch,
  memory: Boxes,
  mtp: Sparkles,
}

export function ComponentIcon({
  component,
  size = 18,
}: {
  component: Pick<Component, 'kind' | 'category'>
  size?: number
}) {
  const kind = component.kind
  const Icon = /norm/.test(kind)
    ? Waves
    : /add|multiply|scores|scale|softmax|silu/.test(kind)
      ? Sigma
      : /decoder|block/.test(kind)
        ? Layers3
        : /token|text|utf8|bpe|ids/.test(kind)
          ? Braces
          : /kv|cache|keys|values|hash|retrieve|memory/.test(kind)
            ? Database
            : /attn|rope|weighted|concat/.test(kind)
              ? Network
              : /head|sample|decode|loss|mtp/.test(kind)
                ? Sparkles
                : component.category === 'data'
                  ? Binary
                  : component.category === 'memory'
                    ? Boxes
                    : CircleDot
  return <Icon size={size} strokeWidth={1.65} />
}

export function Logo() {
  return (
    <svg viewBox="0 0 36 38" fill="none" aria-hidden="true">
      <path d="m18 2 15 8.5v17L18 36 3 27.5v-17L18 2Z" stroke="currentColor" strokeWidth="1.5" />
      <path
        d="m3 10.5 15 9 15-9M18 19.5V36M10.5 6.25l15 8.5v17"
        stroke="currentColor"
        strokeWidth="1.5"
      />
      <path d="m10.5 23.5 7.5-4 7.5 4" stroke="currentColor" strokeWidth="1.5" />
    </svg>
  )
}
