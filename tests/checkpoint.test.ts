import { readFile } from 'node:fs/promises'
import { describe, expect, it } from 'vitest'
import reference from './fixtures/stories260k-reference.json'
import starter from '../examples/transformer-foundations.model.json'
import { parseDocument } from '../src/model/schema'
import { numericalGraph, exportNumericalTrace } from '../src/model/numerics/session'
import { explainCell } from '../src/model/numerics/explain'
import { defaultNumericalSettings } from '../src/model/numerics/engine'
import { checkpointStudySession, loadCheckpointSession } from '../src/model/checkpoints/session'
import {
  CHECKPOINT_SESSION_KEY,
  StoriesTokenizer,
  storiesManifest,
  parseStoriesCheckpoint,
  verifyAsset,
  runStories,
} from '../src/model/checkpoints/stories'

const read = async (file: string) => {
  const data = await readFile(`public/models/stories260k/${file}`)
  return data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength) as ArrayBuffer
}
const weightsBuffer = await read('stories260K.bin'),
  tokenizerBuffer = await read('tok512.bin')
const weights = parseStoriesCheckpoint(weightsBuffer),
  tokenizer = new StoriesTokenizer(tokenizerBuffer)
const settings = {
  prompt: reference.prompt,
  generate: 16,
  seed: 37,
  sampling: 'greedy' as const,
  temperature: 0.8,
}
const trace = runStories(weights, tokenizer, settings)
describe('verified local trained checkpoint', () => {
  it('verifies pinned bundled assets and refuses corruption or another format', async () => {
    await verifyAsset(weightsBuffer, storiesManifest.weights)
    await verifyAsset(tokenizerBuffer, storiesManifest.tokenizer)
    const changed = weightsBuffer.slice(0)
    new Uint8Array(changed)[101] ^= 1
    await expect(verifyAsset(changed, storiesManifest.weights)).rejects.toThrow('SHA-256')
    expect(() => parseStoriesCheckpoint(changed.slice(1))).toThrow('length')
    const wrongHeader = weightsBuffer.slice(0)
    new DataView(wrongHeader).setInt32(0, 32, true)
    expect(() => parseStoriesCheckpoint(wrongHeader)).toThrow('header')
  })
  it('matches original tokenizer IDs, byte fallback, BOS and Unicode decoding', () => {
    expect(tokenizer.encode(reference.prompt)).toEqual(reference.inputIds)
    expect(tokenizer.encode('Hello, café 🧠!')).toEqual(reference.unicodeIds)
    for (const text of ['', reference.prompt, 'Hello, café 🧠!', '  spaces\nnew line', '<s>'])
      expect(tokenizer.decode(tokenizer.encode(text))).toBe(text)
    expect(tokenizer.encode('')).toEqual([1])
    expect(tokenizer.piece(1)).toBe('BOS')
    expect(tokenizer.piece(2)).toBe('EOS')
    expect(() => tokenizer.encode('x\0y')).toThrow('NUL')
  })
  it('matches independent FP32 full-prefix logits, attention, RoPE and 16-token greedy continuation', () => {
    expect(trace.outputIds).toEqual(reference.outputIds)
    expect(tokenizer.decode([...trace.settings.inputIds, ...trace.outputIds])).toBe(reference.text)
    const logits = trace.tensors['call0/logits'],
      row = logits.data.subarray((reference.inputIds.length - 1) * 512)
    expect(
      Math.max(...Array.from(row, (v, i) => Math.abs(v - reference.lastQueryLogits[i]))),
    ).toBeLessThan(0.0001)
    const attention = trace.tensors['call0/l0/attention-0'].data.subarray(
      (reference.inputIds.length - 1) * reference.inputIds.length,
    )
    expect(
      Math.max(
        ...Array.from(attention, (v, i) => Math.abs(v - reference.layer0Head0.attention[i])),
      ),
    ).toBeLessThan(0.00001)
    const q = trace.tensors['call0/l0/query-0'].data.subarray((reference.inputIds.length - 1) * 8)
    expect(
      Math.max(...Array.from(q, (v, i) => Math.abs(v - reference.layer0Head0.query[i]))),
    ).toBeLessThan(0.00001)
    expect(
      trace.frames
        .at(-1)
        ?.cachePositions.every(
          (p) => p.length === reference.inputIds.length + trace.outputIds.length - 1,
        ),
    ).toBe(true)
  })
  it('retains exact trained weights with the correct orientation and tied output provenance', () => {
    const vocab = trace.tensors['weights/vocabulary projection'],
      E = trace.tensors['weights/token embeddings']
    expect(vocab.data[17 * 512 + 403]).toBe(E.data[403 * 64 + 17])
    expect(vocab.recipe.op).toBe('checkpoint')
    expect(explainCell(trace, vocab, 17 * 512 + 403).title).toContain('tied')
    const unique = Object.values(trace.tensors)
      .filter((t) => t.role === 'parameter' && !(t.recipe.op === 'checkpoint' && t.recipe.tiedTo))
      .reduce((n, t) => n + t.data.length, 0)
    expect(unique).toBe(260032)
    expect(trace.checkpoint?.sha256).toBe(reference.checkpointSha256)
    expect(trace.provenance.description).not.toContain('untrained')
    expect(
      Object.values(trace.tensors)
        .filter((t) => t.role === 'parameter')
        .every((t) => t.recipe.op === 'checkpoint'),
    ).toBe(true)
    expect(
      JSON.parse(
        exportNumericalTrace({
          ...trace,
          frames: trace.frames.slice(0, 1),
          tensors: { [vocab.id]: vocab },
        }),
      ).checkpoint.sha256,
    ).toBe(reference.checkpointSha256)
  })
  it('reuses authored graph semantics at all five layers and every GQA head', () => {
    const graph = numericalGraph(
      parseDocument(starter),
      defaultNumericalSettings,
      { direction: 'RIGHT', positions: {} },
      weights.config,
    )
    expect(graph.nodes.some((n) => n.id === 'layer-4.attn.softmax-7')).toBe(true)
    expect(graph.nodes.find((n) => n.id === 'tokens.lookup')?.description).toContain('trained')
    expect(graph.nodes.find((n) => n.id === 'output.lmhead')?.description).toContain('same learned')
    for (const frame of trace.frames)
      for (const id of frame.nodeIds) expect(graph.nodes.some((n) => n.id === id)).toBe(true)
    expect(graph.nodes.some((n) => /untrained|seeded/i.test(n.description))).toBe(false)
  })
  it('bounds complete captures and preserves unknown saved sessions without overwriting the original lab', () => {
    expect(() =>
      runStories(weights, tokenizer, { ...settings, prompt: 'hello '.repeat(50) }),
    ).toThrow('1–48')
    expect(() => runStories(weights, tokenizer, { ...settings, generate: 17 })).toThrow('settings')
    const doc = parseDocument(starter),
      unknown = { version: 99, future: 'preserve' }
    doc.extensions = { ...doc.extensions, [CHECKPOINT_SESSION_KEY]: unknown }
    const loaded = loadCheckpointSession(doc)
    expect(loaded.writable).toBe(false)
    expect(doc.extensions[CHECKPOINT_SESSION_KEY]).toEqual(unknown)
    expect(checkpointStudySession(loaded.session).settings).toEqual(defaultNumericalSettings)
    const supported = loadCheckpointSession(parseDocument(starter)).session
    supported.study.bookmarks.push({
      id: '15/head',
      title: 'Last decoder call',
      node: 'layer-4.attn.softmax-7',
      call: 15,
      note: 'Keep',
    })
    doc.extensions[CHECKPOINT_SESSION_KEY] = supported
    expect(loadCheckpointSession(doc).writable).toBe(true)
  })
})
