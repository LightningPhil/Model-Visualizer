import { useMemo } from 'react'
import katex from 'katex'

export function MathBlock({ tex }: { tex: string }) {
  const html = useMemo(
    () =>
      katex.renderToString(tex, {
        displayMode: true,
        throwOnError: false,
        trust: false,
        strict: 'warn',
        maxExpand: 200,
        maxSize: 20,
        output: 'htmlAndMathml',
      }),
    [tex],
  )
  return (
    <div
      className="equation"
      aria-label={`Equation: ${tex}`}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  )
}
