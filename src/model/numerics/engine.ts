import { parseModelConfig, type ModelConfig } from '../schema'
import type {
  ExecutionAdapter,
  SimulationFrame,
  SimulationOptions,
  SimulationTrace,
} from '../simulation'
import {
  canAttend,
  kvHeadFor,
  residentPositions,
  cachePolicy,
  type CachePolicy,
} from '../attention'
import {
  concatenate,
  elementwise,
  gather,
  headSlice,
  makeTensor,
  matmul,
  rmsnorm,
  rope,
  silu,
  softmax,
  splitHeads,
  transpose,
  type NumericTensor,
} from './tensor'

export type NumericalSettings = {
  seed: number
  inputIds: number[]
  generate: number
  attention: ModelConfig['attention']
  kvCache: CachePolicy
  sampling: 'greedy' | 'sample'
  temperature: number
}
export const defaultNumericalSettings: NumericalSettings = {
  seed: 37,
  inputIds: [7, 19, 31, 42],
  generate: 3,
  attention: 'GQA',
  kvCache: { mode: 'full' },
  sampling: 'greedy',
  temperature: 1,
}
export function teachingConfig(settings: NumericalSettings): ModelConfig {
  return {
    dModel: 16,
    nLayers: 2,
    nQueryHeads: 4,
    nKvHeads: settings.attention === 'MHA' ? 4 : settings.attention === 'MQA' ? 1 : 2,
    headDim: 4,
    ffnDim: 32,
    vocabSize: 256,
    contextLength: 64,
    dtypeBytes: 4,
    attention: settings.attention,
    kvCache: settings.kvCache,
  }
}
export type NumericalFrame = SimulationFrame & {
  call: number
  queryPositions: number[]
  inputs: string[]
  outputs: string[]
  bindings: Record<string, string[]>
}
export type NumericalTrace = Omit<SimulationTrace, 'frames'> & {
  version: 1
  config: ModelConfig
  settings: NumericalSettings
  tensors: Record<string, NumericTensor>
  frames: NumericalFrame[]
  outputIds: number[]
  arithmetic: string
  checkpoint?: CheckpointWeights['provenance']
  tokenization?: {
    prompt: string
    inputIds: number[]
    pieces: string[]
    tokenizerRevision: string
    output: string
  }
}
export type CheckpointWeights = {
  config: ModelConfig
  parameters: Record<string, Float32Array>
  provenance: {
    id: string
    name: string
    revision: string
    sha256: string
    tokenizerSha256: string
    source: string
  }
}
type Options = SimulationOptions & {
  settings: NumericalSettings
  checkpoint?: CheckpointWeights
  onToken?: (ids: number[]) => void
}
function random(seed: number) {
  let x = seed >>> 0
  return () => {
    x = (x + 0x6d2b79f5) >>> 0
    let t = Math.imul(x ^ (x >>> 15), 1 | x)
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}
function namedSeed(seed: number, name: string) {
  let hash = seed >>> 0
  for (const c of name) hash = Math.imul(hash ^ c.charCodeAt(0), 16777619)
  return hash >>> 0
}

export const numericalAdapter: ExecutionAdapter<Options, NumericalTrace> = {
  id: 'numerical-reference-v1',
  run(inputConfig, options) {
    const config = parseModelConfig(inputConfig)
    const settings = structuredClone(options.settings)
    const checkpoint = options.checkpoint
    if (
      checkpoint &&
      JSON.stringify(config) !== JSON.stringify(parseModelConfig(checkpoint.config))
    )
      throw new Error('Checkpoint architecture and execution configuration differ.')
    const signature = (policy: CachePolicy) =>
      JSON.stringify([
        policy.mode,
        'windowSize' in policy ? policy.windowSize : 0,
        'stride' in policy ? policy.stride : 0,
      ])
    if (
      signature(settings.kvCache) !== signature(cachePolicy(config)) ||
      options.mtp ||
      options.promptTokens !== settings.inputIds.length ||
      options.newTokens !== settings.generate
    )
      throw new Error(
        'Numerical adapter options must match the autoregressive experiment and cache policy.',
      )
    if (
      settings.inputIds.length < 1 ||
      settings.inputIds.length > (checkpoint ? 48 : 8) ||
      settings.inputIds.some((t) => !Number.isInteger(t) || t < 0 || t >= config.vocabSize)
    )
      throw new Error(`Choose 1–${checkpoint ? 48 : 8} valid input token IDs.`)
    if (
      !Number.isInteger(settings.seed) ||
      settings.seed < 0 ||
      settings.seed > 2147483647 ||
      !Number.isInteger(settings.generate) ||
      settings.generate < 1 ||
      settings.generate > (checkpoint ? 16 : 8) ||
      settings.temperature < 0.1 ||
      settings.temperature > 3
    )
      throw new Error('Invalid numerical laboratory settings.')
    if (
      config.dModel !== config.nQueryHeads * config.headDim ||
      config.nQueryHeads % config.nKvHeads ||
      config.headDim % 2 ||
      config.nLayers > (checkpoint ? 5 : 4) ||
      config.dModel > (checkpoint ? 64 : 32) ||
      config.vocabSize > 512 ||
      config.ffnDim > (checkpoint ? 172 : 64) ||
      config.nQueryHeads > 8 ||
      config.dtypeBytes !== 4 ||
      settings.attention !== config.attention ||
      !['greedy', 'sample'].includes(settings.sampling) ||
      !Number.isFinite(settings.temperature)
    )
      throw new Error('Numerical execution is limited to the small teaching profile.')
    if (settings.inputIds.length + settings.generate > config.contextLength)
      throw new Error('The requested run exceeds the checkpoint context capacity.')
    const frames: NumericalFrame[] = [],
      tensors: Record<string, NumericTensor> = {}
    const cache = Array.from({ length: config.nLayers }, (_, l) => ({
      k: makeTensor(
        `initial/l${l}/k`,
        'Empty K state',
        [1, config.nKvHeads, 0, config.headDim],
        ['B', 'Hkv', 'R', 'Dh'],
        [],
        { op: 'input' },
        'state',
      ),
      v: makeTensor(
        `initial/l${l}/v`,
        'Empty V state',
        [1, config.nKvHeads, 0, config.headDim],
        ['B', 'Hkv', 'R', 'Dh'],
        [],
        { op: 'input' },
        'state',
      ),
      positions: [] as number[],
      processed: 0,
    }))
    cache.forEach((c) => {
      tensors[c.k.id] = c.k
      tensors[c.v.id] = c.v
    })
    let confirmed = [...settings.inputIds],
      call = 0,
      phase: SimulationFrame['phase'] = 'input',
      queryPositions = [...settings.inputIds.keys()]
    let evicted: number[][] = Array.from({ length: config.nLayers }, () => [])
    const put = (t: NumericTensor, createdAt = frames.length) => {
      if (tensors[t.id]) throw new Error(`Duplicate trace tensor ${t.id}`)
      t.createdAt = createdAt
      tensors[t.id] = t
      return t
    }
    const parameter = (name: string, shape: number[], axes: string[], gain = false) => {
      if (checkpoint) {
        const values = checkpoint.parameters[name]
        if (!values || values.length !== shape.reduce((a, b) => a * b, 1))
          throw new Error(`Missing or incompatible checkpoint parameter: ${name}`)
        return put(
          makeTensor(
            `weights/${name}`,
            name,
            shape,
            axes,
            values,
            {
              op: 'checkpoint',
              ...checkpoint.provenance,
              ...(name === 'vocabulary projection' ? { tiedTo: 'weights/token embeddings' } : {}),
            },
            'parameter',
          ),
          -1,
        )
      }
      const rng = random(namedSeed(settings.seed, name)),
        count = shape.reduce((a, b) => a * b, 1),
        scale = Math.sqrt(3 / shape[0]) * 0.7
      return put(
        makeTensor(
          `weights/${name}`,
          name,
          shape,
          axes,
          Array.from({ length: count }, () => (gain ? 0.9 + rng() * 0.2 : (rng() * 2 - 1) * scale)),
          { op: 'parameter', seed: settings.seed },
          'parameter',
        ),
        -1,
      )
    }
    const embedding = parameter('token embeddings', [config.vocabSize, config.dModel], ['V', 'D'])
    // Independent vocabulary weights are explicit: this profile does not tie E/Wvocab.
    const vocab = parameter('vocabulary projection', [config.dModel, config.vocabSize], ['D', 'V'])
    const finalGain = parameter('final RMS gain', [config.dModel], ['D'], true)
    const layers = Array.from({ length: config.nLayers }, (_, l) => ({
      norm1: parameter(`L${l} attention RMS gain`, [config.dModel], ['D'], true),
      q: parameter(
        `L${l} Wq`,
        [config.dModel, config.nQueryHeads * config.headDim],
        ['D', 'Hq*Dh'],
      ),
      k: parameter(`L${l} Wk`, [config.dModel, config.nKvHeads * config.headDim], ['D', 'Hkv*Dh']),
      v: parameter(`L${l} Wv`, [config.dModel, config.nKvHeads * config.headDim], ['D', 'Hkv*Dh']),
      o: parameter(
        `L${l} Wo`,
        [config.nQueryHeads * config.headDim, config.dModel],
        ['Hq*Dh', 'D'],
      ),
      norm2: parameter(`L${l} FFN RMS gain`, [config.dModel], ['D'], true),
      gate: parameter(`L${l} Wgate`, [config.dModel, config.ffnDim], ['D', 'F']),
      up: parameter(`L${l} Wup`, [config.dModel, config.ffnDim], ['D', 'F']),
      down: parameter(`L${l} Wdown`, [config.ffnDim, config.dModel], ['F', 'D']),
    }))
    function emit(
      nodes: string[],
      title: string,
      description: string,
      inputs: NumericTensor[],
      outputs: NumericTensor[],
      layer?: number,
      milestone = false,
      bindings?: Record<string, string[]>,
    ) {
      frames.push({
        id: frames.length,
        phase,
        title,
        description,
        nodeIds: nodes,
        milestone,
        layer,
        call,
        queryPositions: [...queryPositions],
        queryLength: queryPositions.length,
        contextLength: confirmed.length,
        cacheLengths: cache.map((c) => c.processed),
        cachePositions: cache.map((c) => [...c.positions]),
        evictedPositions: evicted.map((p) => [...p]),
        confirmed: confirmed.map((id, position) => ({
          id: `#${id}`,
          position,
          status: position < settings.inputIds.length ? 'prompt' : 'confirmed',
        })),
        proposals: [],
        generated: confirmed.length - settings.inputIds.length,
        round: call,
        accepted: 0,
        rejected: 0,
        tentative: false,
        inputs: inputs.map((t) => t.id),
        outputs: outputs.map((t) => t.id),
        bindings: bindings ?? Object.fromEntries(nodes.map((n) => [n, outputs.map((t) => t.id)])),
      })
    }
    function forward(ids: number[], start: number) {
      queryPositions = ids.map((_, i) => start + i)
      phase = start === 0 ? 'prefill' : 'decode'
      const key = `call${call}`,
        T = ids.length
      const input = put(
        makeTensor(
          `${key}/ids`,
          'Input token IDs',
          [1, T],
          ['B', 'T'],
          ids,
          { op: 'input' },
          'input',
        ),
      )
      input.positions = { T: [...queryPositions] }
      emit(
        ['tokens.ids'],
        start === 0 ? 'Start with token IDs' : 'Feed the selected ID back',
        checkpoint
          ? 'IDs from the checkpoint’s matching tokenizer select trained embedding rows. Inference runs locally; no prompt is sent to a service.'
          : 'These are opaque vocabulary indices for the small numerical model. No tokenizer or language service is invoked.',
        [],
        [input],
        undefined,
        true,
      )
      let x = put(
        makeTensor(
          `${key}/embedding`,
          'Token embeddings',
          [1, T, config.dModel],
          ['B', 'T', 'D'],
          ids.flatMap((id) =>
            Array.from(embedding.data.subarray(id * config.dModel, (id + 1) * config.dModel)),
          ),
          { op: 'embedding', table: embedding.id, tokens: [...ids] },
        ),
      )
      x.positions = { T: [...queryPositions] }
      emit(
        ['tokens.lookup'],
        'Select embedding rows',
        checkpoint
          ? 'Each token ID selects its exact trained Float32 row from the verified checkpoint embedding matrix.'
          : 'Each ID selects one row of E[V,D]. The vectors are actual seeded FP32 values, with no training.',
        [input, embedding],
        [x],
      )
      for (let l = 0; l < config.nLayers; l++) {
        const p = `layer-${l}`,
          k = `${key}/l${l}`,
          w = layers[l],
          at = (...names: string[]) => names.map((n) => `${p}.${n}`)
        emit(
          at('input'),
          'Residual stream enters the layer',
          `Layer ${l} has separate parameters and K/V state.`,
          [x],
          [x],
          l,
          true,
        )
        const residual = x
        const norm = put(rmsnorm(`${k}/norm1`, x, w.norm1))
        norm.positions = x.positions
        emit(
          at('norm1', 'attn.x'),
          'Normalize attention input',
          'Compute the RMS over D features at each query position, then apply the learned gain. Epsilon = 10⁻⁵.',
          [x, w.norm1],
          [norm],
          l,
        )
        const project = (name: 'q' | 'k' | 'v', heads: number) => {
          const flat = put(
            matmul(`${k}/${name}-flat`, `${name.toUpperCase()} projection`, norm, w[name]),
          )
          const split = put(splitHeads(`${k}/${name}`, flat, heads))
          split.axes[1] = name === 'q' ? 'Hq' : 'Hkv'
          split.positions = { T: [...queryPositions] }
          emit(
            at(`attn.${name}`),
            `Project ${name.toUpperCase()} and expose heads`,
            'One batched linear map followed by a transpose/reshape. Splitting a head is a view of these features, not another projection.',
            [norm, w[name]],
            [split, flat],
            l,
          )
          return split
        }
        const q = project('q', config.nQueryHeads),
          kn = project('k', config.nKvHeads),
          vn = project('v', config.nKvHeads)
        const qr = put(rope(`${k}/rope-q`, q, queryPositions))
        qr.name = 'RoPE query features'
        emit(
          at('attn.ropeq'),
          'Rotate query feature pairs',
          'Adjacent pairs rotate by their absolute position and frequency. The base is 10,000; V is not rotated.',
          [q],
          [qr],
          l,
        )
        const kr = put(rope(`${k}/rope-k`, kn, queryPositions))
        kr.name = 'RoPE key features'
        emit(
          at('attn.ropek'),
          'Rotate key feature pairs',
          'Keys use the same absolute positions as their queries. Compact cache slots never replace RoPE positions.',
          [kn],
          [kr],
          l,
        )
        const old = cache[l],
          workingPositions = [...old.positions, ...queryPositions]
        const kw = put(
          concatenate(`${k}/working-k`, 'Working K: retained prefix + new keys', [old.k, kr], 2, [
            'B',
            'Hkv',
            'U',
            'Dh',
          ]),
        )
        const vw = put(
          concatenate(`${k}/working-v`, 'Working V: retained prefix + new values', [old.v, vn], 2, [
            'B',
            'Hkv',
            'U',
            'Dh',
          ]),
        )
        kw.positions = { U: [...workingPositions] }
        vw.positions = kw.positions
        const positions = residentPositions(config, start + T),
          indices = positions.map((p) => workingPositions.indexOf(p))
        if (indices.some((i) => i < 0))
          throw new Error('Retention requested a key absent from the working set.')
        const nextK = put(gather(`${k}/state-k`, 'Retained K state', kw, indices, 2, 'R')),
          nextV = put(gather(`${k}/state-v`, 'Retained V state', vw, indices, 2, 'R'))
        nextK.role = nextV.role = 'state'
        nextK.positions = nextV.positions = { R: [...positions] }
        evicted[l] = old.positions.filter((p) => !positions.includes(p))
        cache[l] = { k: nextK, v: nextV, positions, processed: start + T }
        emit(
          at('attn.cachek', 'attn.cachev'),
          'Retain K/V; keep working keys available',
          `${positions.length} persistent positions; ${workingPositions.length} working keys. Earlier queries in this batch still use transient K/V that final retention can evict.`,
          [old.k, old.v, kr, vn],
          [nextK, nextV, kw, vw],
          l,
          true,
          { [`${p}.attn.cachek`]: [nextK.id], [`${p}.attn.cachev`]: [nextV.id] },
        )
        const keys: NumericTensor[] = [],
          values: NumericTensor[] = []
        for (let h = 0; h < config.nKvHeads; h++) {
          keys.push(put(headSlice(`${k}/key-${h}`, kw, h)))
          values.push(put(headSlice(`${k}/value-${h}`, vw, h)))
          keys[h].name = `KV group ${h} · working K`
          values[h].name = `KV group ${h} · working V`
          emit(
            at(`attn.cachek-${h}`, `attn.cachev-${h}`),
            `Read KV group ${h}`,
            'These are the actual working K/V slices shared by their assigned query heads. Shared K/V does not imply shared attention probabilities.',
            [kw, vw],
            [keys[h], values[h]],
            l,
            false,
            { [`${p}.attn.cachek-${h}`]: [keys[h].id], [`${p}.attn.cachev-${h}`]: [values[h].id] },
          )
        }
        const headOutputs: NumericTensor[] = []
        for (let h = 0; h < config.nQueryHeads; h++) {
          const kv = kvHeadFor(config, h),
            qh = put(headSlice(`${k}/query-${h}`, qr, h)),
            kt = put(transpose(`${k}/key-transpose-${h}`, keys[kv]))
          qh.name = `Query head ${h} features`
          emit(
            at(`attn.query-${h}`),
            `Query head ${h} → KV group ${kv}`,
            'Each query head has independent scores and softmax. This teaching timeline visits independent heads in sequence for inspection; a kernel can batch them.',
            [qr],
            [qh],
            l,
          )
          const scores = put(matmul(`${k}/scores-${h}`, `Head ${h} dot products`, qh, kt))
          scores.positions = { T: [...queryPositions], U: [...workingPositions] }
          emit(
            at(`attn.scores-${h}`),
            `Q${h} · K${kv}ᵀ`,
            'Contract the Dh feature axis. Click one score cell to inspect every multiply and the final sum.',
            [qh, kt],
            [scores],
            l,
            true,
          )
          const allowed = queryPositions.flatMap((q) =>
              workingPositions.map((s) => canAttend(config, q, s)),
            ),
            data = Array.from(scores.data, (v, i) =>
              allowed[i] ? v / Math.sqrt(config.headDim) : -Infinity,
            )
          const masked = put(
            makeTensor(
              `${k}/masked-${h}`,
              'Scaled and masked scores',
              scores.shape,
              scores.axes,
              data,
              {
                op: 'mask',
                input: scores.id,
                queries: [...queryPositions],
                keys: [...workingPositions],
                headDim: config.headDim,
                allowed,
              },
            ),
          )
          masked.positions = scores.positions
          emit(
            at(`attn.scale-${h}`),
            'Scale scores and apply visibility',
            'Use absolute query/key positions for the causal full, sparse or sliding mask. Excluded entries are −∞, not zero scores.',
            [scores],
            [masked],
            l,
          )
          const weights = put(softmax(`${k}/attention-${h}`, masked))
          weights.name = `Head ${h} attention probabilities`
          weights.positions = scores.positions
          emit(
            at(`attn.softmax-${h}`),
            `Head ${h}: a distribution over visible keys`,
            'Subtract the row maximum, exponentiate, and divide by the visible sum. Masked entries have probability zero.',
            [masked],
            [weights],
            l,
            true,
          )
          const result = put(
            matmul(`${k}/head-${h}`, `Head ${h} value-weighted result`, weights, values[kv]),
          )
          result.positions = { T: [...queryPositions] }
          headOutputs.push(result)
          emit(
            at(`attn.head-${h}`),
            `Attention weights · V${kv}`,
            'Contract the key-slot axis. Each output feature is a weighted sum of the corresponding V feature, not an average across heads.',
            [weights, values[kv]],
            [result],
            l,
          )
        }
        const joined = put(
          concatenate(`${k}/joined`, 'Concatenated head features', headOutputs, 2, [
            'B',
            'T',
            'Hq*Dh',
          ]),
        )
        emit(
          at('attn.concat'),
          'Concatenate heads in query-head order',
          'Rearrange features to Hq × Dh. Concatenation does not sum head outputs.',
          headOutputs,
          [joined],
          l,
        )
        const projected = put(
          matmul(`${k}/output-projection`, 'Attention output projection', joined, w.o),
        )
        emit(
          at('attn.project'),
          'Mix head features with Wo',
          'Project the concatenated head features back to the residual width D.',
          [joined, w.o],
          [projected],
          l,
        )
        const added = put(
          elementwise(`${k}/residual1`, 'Attention residual', residual, projected, 'add'),
        )
        emit(
          at('add1'),
          'Add the attention residual',
          'Element-wise addition preserves the unmodified incoming residual alongside the attention update.',
          [residual, projected],
          [added],
          l,
          true,
        )
        const ff = put(rmsnorm(`${k}/norm2`, added, w.norm2))
        emit(
          at('norm2', 'swiglu.ffx'),
          'Normalize for the feed-forward path',
          'A separate RMSNorm and gain prepare each token for SwiGLU.',
          [added, w.norm2],
          [ff],
          l,
        )
        const gate = put(matmul(`${k}/gate`, 'Gate projection', ff, w.gate))
        emit(
          at('swiglu.gate'),
          'Project the gate branch',
          'D → F uses its own matrix Wgate.',
          [ff, w.gate],
          [gate],
          l,
        )
        const up = put(matmul(`${k}/up`, 'Up projection', ff, w.up))
        emit(
          at('swiglu.up'),
          'Project the up branch',
          'D → F uses a separate matrix Wup; it is not the gate projection reused.',
          [ff, w.up],
          [up],
          l,
        )
        const activated = put(silu(`${k}/silu`, gate))
        emit(
          at('swiglu.silu'),
          'Apply SiLU to the gate',
          'For each feature, compute x · sigmoid(x). A negative gate can remain negative.',
          [gate],
          [activated],
          l,
        )
        const gated = put(elementwise(`${k}/gated`, 'Gated features', activated, up, 'multiply'))
        emit(
          at('swiglu.multiply'),
          'Gate the up features',
          'Multiply matching features element by element. This is a Hadamard product, not matrix multiplication.',
          [activated, up],
          [gated],
          l,
          true,
        )
        const down = put(matmul(`${k}/down`, 'Contracted feed-forward features', gated, w.down))
        emit(
          at('swiglu.down'),
          'Project F → D',
          'The down matrix contracts the intermediate features back to residual width.',
          [gated, w.down],
          [down],
          l,
        )
        x = put(elementwise(`${k}/residual2`, 'Layer output residual', added, down, 'add'))
        emit(
          at('add2'),
          `Layer ${l} complete`,
          'Add the feed-forward update to the residual stream; the next layer receives these exact values.',
          [added, down],
          [x],
          l,
          true,
        )
      }
      const normalized = put(rmsnorm(`${key}/final-norm`, x, finalGain))
      emit(
        ['output.finalnorm'],
        'Final RMS normalization',
        'Normalize each final decoder representation before vocabulary projection.',
        [x, finalGain],
        [normalized],
      )
      const logits = put(matmul(`${key}/logits`, 'Vocabulary logits', normalized, vocab))
      emit(
        ['output.lmhead'],
        'Project into the vocabulary',
        `Each row contains ${config.vocabSize} computed logits. Only the final query row chooses the next token.${checkpoint ? ' The vocabulary projection is tied to the transposed embedding matrix.' : ' Parameters are untrained.'}`,
        [normalized, vocab],
        [logits],
        undefined,
        true,
      )
      return { logits, key }
    }
    let nextIds = [...settings.inputIds],
      start = 0
    const rng = random(namedSeed(settings.seed, 'sampling'))
    for (let generated = 0; generated < settings.generate; generated++) {
      call = generated
      const { logits, key } = forward(nextIds, start)
      phase = 'predict'
      const probabilities = put(softmax(`${key}/probabilities`, logits, settings.temperature)),
        row = nextIds.length - 1,
        distribution = Array.from(
          probabilities.data.subarray(row * config.vocabSize, (row + 1) * config.vocabSize),
        )
      let selected = distribution.indexOf(Math.max(...distribution)),
        draw: number | undefined
      if (settings.sampling === 'sample') {
        draw = rng()
        let remaining = draw * distribution.reduce((sum, value) => sum + value, 0)
        selected = distribution.length - 1
        for (let t = 0; t < distribution.length; t++) {
          remaining -= distribution[t]
          if (remaining <= 0) {
            selected = t
            break
          }
        }
      }
      const choice = put(
        makeTensor(`${key}/choice`, 'Selected token ID', [1, 1], ['B', 'T'], [selected], {
          op: 'choice',
          input: probabilities.id,
          token: selected,
          row,
          method: settings.sampling,
          draw,
        }),
      )
      choice.positions = { T: [start + nextIds.length] }
      emit(
        ['output.sample', 'generation.next'],
        'Select the next token',
        checkpoint
          ? settings.sampling === 'greedy'
            ? 'Choose the highest-probability token from the last query row, using the trained checkpoint.'
            : 'Sample from the trained model’s last-row probabilities using a reproducible random stream.'
          : settings.sampling === 'greedy'
            ? 'Choose the highest-probability token in the last query row. This untrained model does not generate meaningful language.'
            : 'Draw from the last row using a reproducible seeded random stream. These are real probabilities from an untrained model.',
        [logits],
        [probabilities, choice],
        undefined,
        true,
        { 'output.sample': [probabilities.id, choice.id], 'generation.next': [choice.id] },
      )
      confirmed.push(selected)
      phase = 'commit'
      emit(
        ['generation.commit', 'generation.decode'],
        'Commit an output ID',
        `Token #${selected} joins the confirmed sequence. It has not entered the cache until its next forward pass.`,
        [choice],
        [choice],
        undefined,
        true,
      )
      options.onToken?.(confirmed.slice(settings.inputIds.length))
      if (checkpoint && (selected === 1 || selected === 2)) break
      if (generated + 1 < settings.generate) {
        phase = 'decode'
        emit(
          ['generation.feedback'],
          'Loop the selected token back',
          'Reuse retained per-layer state. The generated ID is consumed directly; no text is re-tokenized.',
          [choice],
          [choice],
          undefined,
          true,
        )
      }
      nextIds = [selected]
      start = confirmed.length - 1
    }
    phase = 'done'
    emit(
      ['generation.decode'],
      'Numerical run complete',
      checkpoint
        ? 'Every displayed value was computed locally from the verified trained checkpoint. Generation stops at BOS/EOS or the requested budget. The last selected ID may remain uncached.'
        : 'Every displayed value came from the local reference operations and seeded, untrained weights. The last selected token may remain uncached.',
      [],
      [tensors[`call${call}/choice`]],
      undefined,
      true,
    )
    return {
      version: 1,
      config,
      settings,
      options: {
        promptTokens: options.promptTokens,
        newTokens: options.newTokens,
        mtp: options.mtp,
        scenario: options.scenario,
        splitHeads: options.splitHeads,
      },
      provenance: {
        kind: 'numerical',
        modelRevision: checkpoint?.provenance.revision ?? 'untrained-reference-v1',
        description: checkpoint
          ? `Trained ${checkpoint.provenance.name}; checkpoint SHA-256 ${checkpoint.provenance.sha256}. Local TypeScript reference arithmetic.`
          : 'Deterministic local reference transformer with seeded, untrained parameters. Not a trained checkpoint.',
      },
      arithmetic:
        'Float32 tensor storage; JavaScript Number accumulation; each operation output rounds to Float32. No GPU-kernel or BF16 equivalence is claimed.',
      frames,
      tensors,
      outputIds: confirmed.slice(settings.inputIds.length),
      ...(checkpoint ? { checkpoint: checkpoint.provenance } : {}),
    }
  },
}
export function executeTeachingModel(settings: NumericalSettings) {
  return numericalAdapter.run(teachingConfig(settings), {
    promptTokens: settings.inputIds.length,
    newTokens: settings.generate,
    mtp: false,
    scenario: 'accept',
    splitHeads: true,
    settings,
  })
}
export function latestBinding(trace: NumericalTrace, node: string, cursor: number) {
  for (let i = Math.min(cursor, trace.frames.length - 1); i >= 0; i--)
    if (trace.frames[i].bindings[node]) return trace.frames[i]
  return undefined
}
