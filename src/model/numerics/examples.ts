import type { Component, ModelConfig } from '../schema'
import { instanceOf } from '../whole-model'

/** Explicit links to foundations teaching examples, not executable semantics
 * inferred from an arbitrary imported component or equation. */
export function numericalExampleFor(
  graphId: string,
  component: Component | undefined,
  layer = 0,
): string | undefined {
  if (!component) return
  const origin = instanceOf(component),
    graph = origin?.graph ?? graphId,
    id = origin?.node ?? component.id,
    l = origin?.layer ?? layer
  const prefix = `layer-${l}`
  if (graph === 'overview')
    return (
      {
        embedding: 'tokens.lookup',
        decoder: `${prefix}.input`,
        kv: `${prefix}.attn.cachek`,
        output: 'output.lmhead',
      } as Record<string, string>
    )[id]
  if (graph === 'tokenization')
    return ({ ids: 'tokens.ids', lookup: 'tokens.lookup' } as Record<string, string>)[id]
  if (graph === 'block')
    return (
      {
        input: `${prefix}.input`,
        norm1: `${prefix}.norm1`,
        attn: `${prefix}.attn.scores-0`,
        add1: `${prefix}.add1`,
        norm2: `${prefix}.norm2`,
        swiglu: `${prefix}.swiglu.multiply`,
        add2: `${prefix}.add2`,
      } as Record<string, string>
    )[id]
  if (graph === 'attention' || graph === 'attention-heads') {
    if (origin?.head !== undefined && id === 'weighted') return `${prefix}.attn.head-${origin.head}`
    if (origin?.head !== undefined && ['cachek', 'cachev'].includes(id))
      return `${prefix}.attn.${id}-${origin.head}`
    const name = (
      {
        x: 'x',
        q: 'q',
        k: 'k',
        v: 'v',
        ropeq: 'ropeq',
        ropek: 'ropek',
        cachek: 'cachek',
        cachev: 'cachev',
        scores: 'scores-0',
        scale: 'scale-0',
        softmax: 'softmax-0',
        weighted: 'head-0',
        concat: 'concat',
        project: 'project',
      } as Record<string, string>
    )[id]
    if (name) return `${prefix}.attn.${name}`
    if (/^(query|scores|scale|softmax|head|cachek|cachev)-\d+$/.test(id))
      return `${prefix}.attn.${id}`
  }
  if (graph === 'feedforward' && ['ffx', 'gate', 'up', 'silu', 'multiply', 'down'].includes(id))
    return `${prefix}.swiglu.${id}`
  if (graph === 'cache')
    return (
      {
        newk: `${prefix}.attn.k`,
        newv: `${prefix}.attn.v`,
        keys: `${prefix}.attn.cachek`,
        values: `${prefix}.attn.cachev`,
        read: `${prefix}.attn.softmax-0`,
      } as Record<string, string>
    )[id]
  if (graph === 'output')
    return (
      {
        finalnorm: 'output.finalnorm',
        lmhead: 'output.lmhead',
        sample: 'output.sample',
        decode: 'generation.decode',
      } as Record<string, string>
    )[id]
  if (graph === 'generation' && ['next', 'commit', 'decode', 'feedback'].includes(id))
    return `generation.${id}`
}

/** Large-model layer/head numbers are mapped explicitly to a small example.
 * The UI labels that separate profile; these are not the loaded model's values. */
export function resolveNumericalExample(
  entry: string,
  config: Pick<ModelConfig, 'nLayers' | 'nQueryHeads' | 'nKvHeads'>,
) {
  return entry
    .replace(/^layer-(\d+)/, (_, l: string) => `layer-${Number(l) % config.nLayers}`)
    .replace(
      /(query|scores|scale|softmax|head|cachek|cachev)-(\d+)$/,
      (_, kind: string, h: string) =>
        `${kind}-${Number(h) % (kind.startsWith('cache') ? config.nKvHeads : config.nQueryHeads)}`,
    )
}
