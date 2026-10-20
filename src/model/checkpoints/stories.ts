import manifest from '../../../examples/stories260k.checkpoint.json' with { type: 'json' }
import { parseModelConfig } from '../schema'
import { numericalAdapter, type CheckpointWeights, type NumericalTrace } from '../numerics/engine'

export const storiesManifest = manifest
export { CHECKPOINT_SESSION_KEY } from './identity'
export type TrainedRunSettings = {
  prompt: string
  generate: number
  seed: number
  sampling: 'greedy' | 'sample'
  temperature: number
}
export const defaultTrainedSettings: TrainedRunSettings = {
  prompt: 'Once upon a time',
  generate: 12,
  seed: 37,
  sampling: 'greedy',
  temperature: 0.8,
}

export async function verifyAsset(buffer: ArrayBuffer, asset: typeof manifest.weights) {
  if (buffer.byteLength !== asset.bytes) throw new Error(`Wrong size for ${asset.file}.`)
  const hash = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', buffer)), (x) =>
    x.toString(16).padStart(2, '0'),
  ).join('')
  if (hash !== asset.sha256)
    throw new Error(`SHA-256 verification failed for ${asset.file}. No inference was run.`)
}

/** Legacy llama2.c v0 Float32 layout. This loader deliberately accepts only the
 * pinned Stories260K architecture; it never guesses another .bin format. */
export function parseStoriesCheckpoint(buffer: ArrayBuffer): CheckpointWeights {
  if (buffer.byteLength !== manifest.weights.bytes)
    throw new Error('Invalid Stories260K file length.')
  const header = new DataView(buffer),
    expected = [64, 172, 5, 8, 4, 512, 512]
  if (expected.some((v, i) => header.getInt32(i * 4, true) !== v))
    throw new Error('Unsupported checkpoint header.')
  const config = parseModelConfig(manifest.config),
    parameters: Record<string, Float32Array> = {}
  let at = 28
  const take = (count: number) => {
    const data = new Float32Array(count)
    for (let i = 0; i < count; i++) {
      const value = header.getFloat32(at, true)
      at += 4
      if (!Number.isFinite(value)) throw new Error('Checkpoint contains a non-finite weight.')
      data[i] = value
    }
    return data
  }
  const matrix = (name: string, input: number, output: number) => {
    const disk = take(input * output),
      rowVector = new Float32Array(disk.length)
    for (let i = 0; i < input; i++)
      for (let o = 0; o < output; o++) rowVector[i * output + o] = disk[o * input + i]
    parameters[name] = rowVector
  }
  const D = config.dModel,
    F = config.ffnDim,
    L = config.nLayers,
    K = config.nKvHeads * config.headDim
  parameters['token embeddings'] = take(config.vocabSize * D)
  for (let l = 0; l < L; l++) parameters[`L${l} attention RMS gain`] = take(D)
  for (const [name, input, output] of [
    ['Wq', D, D],
    ['Wk', D, K],
    ['Wv', D, K],
    ['Wo', D, D],
  ] as const)
    for (let l = 0; l < L; l++) matrix(`L${l} ${name}`, input, output)
  for (let l = 0; l < L; l++) parameters[`L${l} FFN RMS gain`] = take(D)
  for (const [name, input, output] of [
    ['Wgate', D, F],
    ['Wdown', F, D],
    ['Wup', D, F],
  ] as const)
    for (let l = 0; l < L; l++) matrix(`L${l} ${name}`, input, output)
  parameters['final RMS gain'] = take(D)
  // Old files contain precomputed rotary tables. The runtime uses the equivalent
  // position/frequency formula, like current upstream run.c, and skips these.
  take(config.contextLength * config.headDim)
  if (at !== buffer.byteLength) throw new Error('Unexpected checkpoint trailing data.')
  const vocab = new Float32Array(config.vocabSize * D),
    embedding = parameters['token embeddings']
  for (let d = 0; d < D; d++)
    for (let v = 0; v < config.vocabSize; v++)
      vocab[d * config.vocabSize + v] = embedding[v * D + d]
  parameters['vocabulary projection'] = vocab
  return {
    config,
    parameters,
    provenance: {
      id: manifest.id,
      name: manifest.name,
      revision: manifest.revision,
      sha256: manifest.weights.sha256,
      tokenizerSha256: manifest.tokenizer.sha256,
      source: manifest.source,
    },
  }
}

/** Matching scored-BPE tokenizer in llama2.c's exported SentencePiece vocabulary.
 * No teaching-tokenizer fallback, whitespace normalization or chat template. */
export class StoriesTokenizer {
  readonly pieces: string[] = []
  readonly scores: number[] = []
  private readonly lookup = new Map<string, number>()
  constructor(buffer: ArrayBuffer) {
    if (buffer.byteLength !== manifest.tokenizer.bytes)
      throw new Error('Invalid tokenizer file length.')
    const view = new DataView(buffer),
      decoder = new TextDecoder('utf-8', { fatal: true }),
      max = view.getInt32(0, true)
    let at = 4
    for (let i = 0; i < 512; i++) {
      if (at + 8 > buffer.byteLength) throw new Error('Truncated tokenizer.')
      const score = view.getFloat32(at, true),
        length = view.getInt32(at + 4, true)
      at += 8
      if (!Number.isFinite(score) || length < 1 || length > max || at + length > buffer.byteLength)
        throw new Error('Invalid vocabulary entry.')
      const piece = decoder.decode(new Uint8Array(buffer, at, length))
      at += length
      this.pieces.push(piece)
      this.scores.push(score)
      this.lookup.set(piece, i)
    }
    if (
      at !== buffer.byteLength ||
      this.pieces[1].trim() !== '<s>' ||
      this.pieces[2].trim() !== '</s>' ||
      !this.lookup.has(' ')
    )
      throw new Error('Unsupported tokenizer vocabulary.')
    for (let byte = 0; byte < 256; byte++)
      if (this.pieces[byte + 3] !== `<0x${byte.toString(16).toUpperCase().padStart(2, '0')}>`)
        throw new Error('Invalid tokenizer byte fallback.')
  }
  encode(text: string) {
    if (text.length > 600 || text.includes('\0'))
      throw new Error('Use at most 600 characters without NUL.')
    const tokens = [1]
    if (text) tokens.push(this.lookup.get(' ')!)
    for (const character of text) {
      const id = this.lookup.get(character)
      if (id !== undefined) tokens.push(id)
      else tokens.push(...Array.from(new TextEncoder().encode(character), (b) => b + 3))
    }
    for (;;) {
      let best = -1e10,
        index = -1,
        merged = -1
      for (let i = 0; i < tokens.length - 1; i++) {
        const id = this.lookup.get(this.pieces[tokens[i]] + this.pieces[tokens[i + 1]])
        if (id !== undefined && this.scores[id] > best) {
          best = this.scores[id]
          index = i
          merged = id
        }
      }
      if (index < 0) break
      tokens.splice(index, 2, merged)
    }
    return tokens
  }
  decode(ids: number[]) {
    const bytes: number[] = []
    ids.forEach((id, i) => {
      if (!Number.isInteger(id) || id < 0 || id >= 512) throw new Error('Invalid vocabulary ID.')
      if (id === 1 || id === 2) return
      let piece = this.pieces[id]
      if (ids[i - 1] === 1 && piece.startsWith(' ')) piece = piece.slice(1)
      const raw = piece.match(/^<0x([0-9A-F]{2})>$/)
      bytes.push(...(raw ? [parseInt(raw[1], 16)] : new TextEncoder().encode(piece)))
    })
    return new TextDecoder().decode(new Uint8Array(bytes))
  }
  piece(id: number) {
    if (id === 1) return 'BOS'
    if (id === 2) return 'EOS'
    return this.pieces[id] ?? `#${id}`
  }
}

export function runStories(
  weights: CheckpointWeights,
  tokenizer: StoriesTokenizer,
  settings: TrainedRunSettings,
  onToken?: (ids: number[], text: string) => void,
): NumericalTrace {
  const inputIds = tokenizer.encode(settings.prompt)
  const trace = numericalAdapter.run(weights.config, {
    promptTokens: inputIds.length,
    newTokens: settings.generate,
    mtp: false,
    scenario: 'accept',
    splitHeads: true,
    settings: {
      ...settings,
      inputIds,
      attention: weights.config.attention,
      kvCache: { mode: 'full' },
    },
    checkpoint: weights,
    onToken: (ids) =>
      onToken?.(
        ids,
        tokenizer.decode([...inputIds, ...ids]).slice(tokenizer.decode(inputIds).length),
      ),
  })
  trace.tokenization = {
    prompt: settings.prompt,
    inputIds,
    pieces: tokenizer.pieces,
    tokenizerRevision: manifest.revision,
    output: tokenizer
      .decode([...inputIds, ...trace.outputIds])
      .slice(tokenizer.decode(inputIds).length),
  }
  return trace
}
