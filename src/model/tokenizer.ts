export type EncodedToken = { id: number; bytes: number[]; label: string }
export type MergeStep = { pair: [number, number]; result: number; rank: number; pieces: number[] }
export type Tokenization = { tokens: EncodedToken[]; steps: MergeStep[]; byteLength: number }
export interface TokenizerAdapter {
  id: string
  name: string
  vocabularySize: number
  encode: (text: string) => Tokenization
  decode: (ids: number[]) => string
}

// A small, explicit byte-level BPE vocabulary. This is a teaching tokenizer,
// not a surrogate for any production model's tokenizer or special-token rules.
const words = [
  'The',
  ' future',
  ' is',
  ' beautifully',
  ' complex',
  '.',
  'Hello',
  ' world',
  ' attention',
  ' language',
  ' model',
  ' token',
  ' neural',
  ' network',
  ' memory',
  ' learns',
  ' patterns',
  ' in',
  ' a',
  ' the',
  ' of',
  ' and',
  ' to',
  ' be',
  ' or',
  ' not',
  ' that',
  ' how',
  ' works',
  ' visual',
  'ize',
  'ing',
  'er',
  's',
  'This',
  ' can',
  ' you',
  ' see',
  '?',
  '!',
  'I',
  ' love',
  ' math',
]
const vocab = new Map<number, number[]>(Array.from({ length: 256 }, (_, i) => [i, [i]]))
const byBytes = new Map<string, number>(Array.from({ length: 256 }, (_, i) => [String(i), i]))
const rules: { left: number; right: number; result: number }[] = []
for (const word of words) {
  const bytes = [...new TextEncoder().encode(word)]
  let left = bytes[0]
  for (const right of bytes.slice(1)) {
    const merged = [...vocab.get(left)!, ...vocab.get(right)!]
    const key = merged.join(',')
    const existing = byBytes.get(key)
    if (existing !== undefined) {
      left = existing
      continue
    }
    const result = vocab.size
    rules.push({ left, right, result })
    vocab.set(result, merged)
    byBytes.set(key, result)
    left = result
  }
}

function visible(bytes: number[]) {
  const decoded = new TextDecoder('utf-8', { fatal: true })
  try {
    return decoded
      .decode(new Uint8Array(bytes))
      .replaceAll(' ', '␠')
      .replaceAll('\n', '↵')
      .replaceAll('\t', '⇥')
  } catch {
    return bytes.map((b) => `0x${b.toString(16).padStart(2, '0')}`).join(' ')
  }
}

export function demoTokenLabel(id: number) {
  const bytes = vocab.get(id)
  if (!bytes) throw new Error(`Unknown demo token: ${id}`)
  return visible(bytes)
}

export const demoTokenizer: TokenizerAdapter = {
  id: 'byte-bpe-demo-v1',
  name: 'Byte BPE · teaching vocabulary',
  vocabularySize: vocab.size,
  encode(text) {
    const bytes = [...new TextEncoder().encode(text)]
    let pieces = [...bytes]
    const steps: MergeStep[] = []
    while (true) {
      let bestRank = Infinity
      let bestIndex = -1
      for (let i = 0; i < pieces.length - 1; i++) {
        const rank = rules.findIndex(
          (rule) => rule.left === pieces[i] && rule.right === pieces[i + 1],
        )
        if (rank >= 0 && rank < bestRank) {
          bestRank = rank
          bestIndex = i
        }
      }
      if (bestIndex < 0) break
      const rule = rules[bestRank]
      pieces.splice(bestIndex, 2, rule.result)
      steps.push({
        pair: [rule.left, rule.right],
        result: rule.result,
        rank: bestRank,
        pieces: [...pieces],
      })
    }
    return {
      tokens: pieces.map((id) => ({ id, bytes: vocab.get(id)!, label: visible(vocab.get(id)!) })),
      steps,
      byteLength: bytes.length,
    }
  },
  decode(ids) {
    return new TextDecoder().decode(
      new Uint8Array(
        ids.flatMap((id) => {
          const bytes = vocab.get(id)
          if (!bytes) throw new Error(`Unknown demo token: ${id}`)
          return bytes
        }),
      ),
    )
  },
}

export function attentionWeights(count: number, head: number) {
  // Deterministic teaching scores, not measured activations from a trained model.
  return Array.from({ length: count }, (_, row) => {
    const scores = Array.from(
      { length: row + 1 },
      (_, col) =>
        Math.sin((row + 1) * (col + 2) * (head + 1) * 0.63) * 1.5 - Math.abs(row - col) * 0.22,
    )
    const max = Math.max(...scores)
    const exps = scores.map((score) => Math.exp(score - max))
    const total = exps.reduce((a, b) => a + b, 0)
    return Array.from({ length: count }, (_, col) => (col <= row ? exps[col] / total : 0))
  })
}
