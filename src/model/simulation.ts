import type { ModelConfig } from './schema'

/** A renderer-neutral trace. A future numerical/local-model adapter can emit
 * the same frames with measured tensors and actual token IDs. The graphical
 * adapter below makes no predictions and supplies no invented activations. */
export type Packet = {
  id: string
  position: number
  status:
    | 'prompt'
    | 'confirmed'
    | 'proposed'
    | 'accepted'
    | 'rejected'
    | 'discarded'
    | 'correction'
    | 'bonus'
}
export type SimulationFrame = {
  id: number
  phase:
    'input' | 'prefill' | 'predict' | 'draft' | 'verify' | 'rollback' | 'commit' | 'decode' | 'done'
  title: string
  description: string
  nodeIds: string[]
  milestone: boolean
  layer?: number
  queryLength: number
  contextLength: number
  cacheLengths: number[]
  confirmed: Packet[]
  proposals: Packet[]
  generated: number
  round: number
  accepted: number
  rejected: number
  tentative: boolean
}
export type SimulationOptions = {
  promptTokens: number
  newTokens: number
  mtp: boolean
  scenario: 'mixed' | 'accept' | 'reject'
}
export type SimulationTrace = {
  provenance: {
    kind: 'illustrative' | 'numerical' | 'measured'
    description: string
    modelRevision?: string
  }
  options: SimulationOptions
  frames: SimulationFrame[]
}
export interface ExecutionAdapter {
  id: string
  run(config: ModelConfig, options: SimulationOptions): SimulationTrace
}

/** Fixed abstract target sequence. Numbers identify opaque visual packets;
 * they are not vocabulary IDs or outputs of trained weights. */
export const graphicalAdapter: ExecutionAdapter = {
  id: 'graphical-execution-v1',
  run(config, options) {
    const { promptTokens, newTokens, mtp, scenario } = options
    if (
      !Number.isInteger(promptTokens) ||
      promptTokens < 1 ||
      promptTokens > 32 ||
      !Number.isInteger(newTokens) ||
      newTokens < 1 ||
      newTokens > 16
    )
      throw new Error('Choose 1–32 input packets and 1–16 output packets.')
    if (promptTokens + newTokens > config.contextLength)
      throw new Error('The input and output budget exceed the model context capacity.')
    const frames: SimulationFrame[] = []
    let phase: SimulationFrame['phase'] = 'input',
      queryLength = promptTokens,
      round = 0,
      accepted = 0,
      rejected = 0,
      tentative = false
    let confirmed: Packet[] = Array.from({ length: promptTokens }, (_, i) => ({
      id: `P${i}`,
      position: i,
      status: 'prompt',
    }))
    let proposals: Packet[] = [],
      cacheLengths = Array(config.nLayers).fill(0) as number[]
    const generated = () => confirmed.length - promptTokens
    function emit(
      nodeIds: string[],
      title: string,
      description: string,
      milestone = false,
      layer?: number,
    ) {
      frames.push({
        id: frames.length,
        phase,
        title,
        description,
        nodeIds,
        milestone,
        layer,
        queryLength,
        contextLength: confirmed.length,
        cacheLengths: [...cacheLengths],
        confirmed: confirmed.map((p) => ({ ...p })),
        proposals: proposals.map((p) => ({ ...p })),
        generated: generated(),
        round,
        accepted,
        rejected,
        tentative,
      })
    }
    function decoderPass(count: number, prefixLength: number, verify = false) {
      queryLength = count
      phase = verify ? 'verify' : prefixLength === 0 ? 'prefill' : 'decode'
      for (let layer = 0; layer < config.nLayers; layer++) {
        const p = `layer-${layer}`
        const at = (...ids: string[]) => ids.map((id) => `${p}.${id}`)
        emit(
          at('input'),
          `Layer ${layer} · residual input`,
          `${count} query position${count === 1 ? '' : 's'} enter this decoder layer.`,
          false,
          layer,
        )
        emit(
          at('norm1', 'attn.x'),
          'Normalize the attention input',
          'RMSNorm acts over the hidden features of each query position.',
          false,
          layer,
        )
        emit(
          at('attn.q', 'attn.k', 'attn.v'),
          'Project Q, K and V',
          `${config.nQueryHeads} query heads and ${config.nKvHeads} KV heads use this layer’s own weights.`,
          false,
          layer,
        )
        emit(
          at('attn.ropeq', 'attn.ropek'),
          'Apply positional rotations',
          'RoPE rotates query/key feature pairs at their absolute sequence positions.',
          false,
          layer,
        )
        cacheLengths[layer] = prefixLength + count
        emit(
          at(
            'attn.cachek',
            'attn.cachev',
            ...Array.from({ length: config.nKvHeads }, (_, h) => `attn.cachek-${h}`),
            ...Array.from({ length: config.nKvHeads }, (_, h) => `attn.cachev-${h}`),
          ),
          verify ? 'Write tentative K/V' : 'Append layer K/V',
          `Layer ${layer} now holds ${cacheLengths[layer]} sequence positions per KV head. ${verify ? 'These positions can be rolled back after verification.' : 'Earlier prefix state is reused.'}`,
          false,
          layer,
        )
        emit(
          at('attn.scores'),
          'Query–key dot products',
          `Each query contracts with its mapped KV head over Dh = ${config.headDim} features.`,
          false,
          layer,
        )
        emit(
          at('attn.scale'),
          'Scale and apply the causal mask',
          'A query can read the existing prefix and preceding positions in the current batch; future proposals are masked.',
          false,
          layer,
        )
        emit(
          at('attn.softmax'),
          'Normalize attention rows',
          'Softmax normalizes each query over its visible key positions. No activation values are fabricated in this graphical run.',
          false,
          layer,
        )
        emit(
          at(
            'attn.weighted',
            ...Array.from({ length: config.nQueryHeads }, (_, h) => `attn.head-${h}`),
          ),
          'All query heads produce results',
          `The ${config.nQueryHeads} query heads run as a batched operation, sharing ${config.nKvHeads} KV heads according to ${config.attention}.`,
          false,
          layer,
        )
        emit(
          at('attn.concat', 'attn.project'),
          'Join heads and project',
          'Concatenate the per-head features, then mix them with the learned output projection.',
          false,
          layer,
        )
        emit(
          at('add1'),
          'First residual addition',
          'Add the attention output to the unchanged incoming residual stream.',
          false,
          layer,
        )
        emit(
          at('norm2', 'swiglu.ffx'),
          'Normalize the feed-forward input',
          'A separate RMSNorm prepares the token-wise feed-forward path.',
          false,
          layer,
        )
        emit(
          at('swiglu.gate', 'swiglu.up'),
          'Expand two feature branches',
          `Independent D → F projections expand ${config.dModel} features to ${config.ffnDim}.`,
          false,
          layer,
        )
        emit(
          at('swiglu.silu'),
          'Apply the SiLU gate',
          'The gate branch applies x · sigmoid(x).',
          false,
          layer,
        )
        emit(
          at('swiglu.multiply', 'swiglu.down'),
          'Gate and contract features',
          'Element-wise multiplication combines the branches, then F → D returns to the residual width.',
          false,
          layer,
        )
        emit(
          at('add2'),
          `Layer ${layer} complete`,
          'The second residual addition produces the next layer’s input. Every decoder layer is visited explicitly.',
          true,
          layer,
        )
      }
    }
    function prediction() {
      phase = 'predict'
      emit(
        ['output.finalnorm'],
        'Final normalization',
        'Normalize the final decoder state at each query position.',
      )
      emit(
        ['output.lmhead'],
        'Project into the vocabulary',
        `The architectural vocabulary head has ${config.vocabSize} logits per query position. This run illustrates control flow; it does not compute logits.`,
      )
      emit(
        ['output.sample', 'generation.next'],
        'Select the next packet',
        'A scripted reference packet stands in for a greedy next-token choice.',
        true,
      )
    }
    const packet = (index: number, status: Packet['status'] = 'proposed'): Packet => ({
      id: `G${index}`,
      position: promptTokens + index,
      status,
    })
    emit(
      ['tokens.text'],
      'Ingest the input stream',
      `${promptTokens} abstract input packets. No text is required for this graphical simulation.`,
      true,
    )
    emit(
      ['tokens.utf8', 'tokens.bpe'],
      'Tokenizer pathway',
      'The packets illustrate text → bytes → BPE pieces. The separate tokenizer lab performs actual byte-BPE encoding.',
    )
    emit(
      ['tokens.ids'],
      'Form the token sequence',
      'Ordered input packets now represent the model’s token positions.',
      true,
    )
    emit(
      ['tokens.lookup'],
      'Look up embeddings',
      'Token indices select rows of the embedding matrix; the diagram carries vectors onward.',
    )
    decoderPass(promptTokens, 0)
    prediction()
    while (generated() < newTokens) {
      round++
      const base = generated(),
        prefix = confirmed.length,
        remaining = newTokens - base
      if (!mtp) {
        proposals = [packet(base)]
        emit(
          ['generation.next'],
          'One next-token proposal',
          'Ordinary autoregressive decoding produces one new token per iteration.',
          true,
        )
        phase = 'commit'
        confirmed.push(packet(base, 'confirmed'))
        proposals = []
        emit(
          ['generation.commit', 'generation.decode', 'output.decode'],
          'Commit one output packet',
          'Append the chosen token ID. Rendering text is optional; the next iteration consumes the ID directly.',
          true,
        )
        if (generated() < newTokens) {
          emit(
            ['generation.feedback', 'tokens.ids'],
            'Feed the selected ID back',
            'The next iteration consumes the selected token ID directly, without re-tokenizing rendered text.',
            true,
          )
          emit(
            ['tokens.lookup'],
            'Embed the new query token',
            'Reuse the token embedding matrix to prepare the one uncached query position.',
          )
          decoderPass(1, prefix)
          prediction()
        }
        continue
      }
      phase = 'draft'
      queryLength = 1
      proposals = [packet(base)]
      emit(
        ['generation.next', 'generation.hidden'],
        'Begin a speculative round',
        'The base decoder supplies the first proposal and its final hidden state.',
        true,
      )
      for (let depth = 1; depth <= 2 && depth < remaining; depth++) {
        emit(
          [`generation.embed${depth}`, `generation.combine${depth}`],
          `MTP depth ${depth} · condition`,
          'Embed the preceding proposal, normalize both inputs, concatenate, and project 2D → D.',
        )
        emit(
          [`generation.mtp${depth}`],
          `MTP depth ${depth} · decoder`,
          'An additional causal decoder module produces the next-depth hidden representation.',
        )
        const shouldMatch =
          scenario === 'accept' ||
          (scenario === 'mixed' && (round % 3 === 2 || (round % 3 === 1 && depth === 1)))
        proposals.push(
          shouldMatch
            ? packet(base + depth)
            : { id: `D${round}.${depth}`, position: prefix + depth, status: 'proposed' },
        )
        emit(
          [`generation.head${depth}`],
          `MTP proposes packet ${depth + 1}`,
          'The shared vocabulary head proposes a later token. Proposal identities and decisions are scripted for this scenario.',
          true,
        )
      }
      phase = 'verify'
      tentative = true
      queryLength = proposals.length
      emit(
        ['generation.verify'],
        'Verify the entire proposal batch',
        'One causal main-decoder pass checks all proposals. Existing prefix logits check the first proposal; each batch row predicts the following position.',
        true,
      )
      emit(
        ['tokens.lookup'],
        'Embed the proposal batch',
        'Look up all proposed token IDs as one batch. The main decoder then applies its causal mask to these positions.',
      )
      decoderPass(proposals.length, prefix, true)
      emit(
        ['output.finalnorm'],
        'Normalize verification states',
        'Apply the main decoder’s final normalization to each proposed query position.',
      )
      emit(
        ['output.lmhead'],
        'Produce shifted verification logits',
        'The main vocabulary head predicts the token following each proposal position. Prefix logits check proposal 1; row j checks proposal j + 1.',
      )
      let matching = 0
      for (let j = 0; j < proposals.length; j++) {
        const matches = proposals[j].id === packet(base + j).id
        proposals[j].status = matches ? 'accepted' : 'rejected'
        if (matches) {
          matching++
          accepted++
        } else {
          rejected += proposals.length - j
          proposals.slice(j + 1).forEach((p) => {
            p.status = 'discarded'
          })
        }
        phase = 'verify'
        emit(
          ['generation.compare'],
          matches ? `Proposal ${j + 1} accepted` : `Proposal ${j + 1} rejected`,
          matches
            ? 'This proposal matches the scripted reference at the same causal position.'
            : 'Stop at the first mismatch. Every subsequent proposal is discarded, even if it could match in isolation.',
          true,
        )
        if (!matches) break
      }
      const rejectedTail = matching < proposals.length
      if (rejectedTail) {
        phase = 'rollback'
        cacheLengths = cacheLengths.map(() => prefix + matching)
        emit(
          ['generation.reject'],
          'Roll back tentative K/V',
          `Keep ${matching} accepted proposal${matching === 1 ? '' : 's'}; remove the rejected suffix from every layer’s sequence axis.`,
          true,
        )
      }
      tentative = false
      confirmed.push(...Array.from({ length: matching }, (_, j) => packet(base + j, 'confirmed')))
      const extra = generated() < newTokens
      if (extra) confirmed.push(packet(generated(), rejectedTail ? 'correction' : 'bonus'))
      phase = 'commit'
      emit(
        ['generation.commit', 'generation.decode'],
        `Commit ${matching + Number(extra)} output packets`,
        rejectedTail
          ? 'Append the accepted prefix and the main-model correction. Rejected draft packets never enter the confirmed sequence.'
          : extra
            ? 'All proposals matched. Append them plus a main-model bonus token when the output budget permits.'
            : 'All remaining proposals matched. The requested output-token budget is now complete.',
        true,
      )
      proposals = []
      if (generated() < newTokens) {
        // The correction/bonus token has been selected but is not cached yet.
        // Its next forward pass materializes its K/V at each decoder layer.
        emit(
          ['generation.feedback', 'tokens.ids'],
          'Feed the confirmed ID back',
          'Reuse the accepted prefix state and process the one uncached correction or bonus token.',
          true,
        )
        if (extra) {
          emit(
            ['tokens.lookup'],
            'Embed the correction or bonus',
            'Only the newly selected token needs a new embedding and main-decoder query.',
          )
          decoderPass(1, confirmed.length - 1)
        }
        prediction()
      }
    }
    phase = 'done'
    emit(
      ['generation.decode', 'generation.feedback'],
      'Run complete',
      `${newTokens} output packets committed. Stop at the requested token budget. The final selected token may remain uncached because no further forward pass is needed.`,
      true,
    )
    return {
      provenance: {
        kind: 'illustrative',
        description:
          'Scripted graphical execution. Abstract packets, reference choices and acceptance outcomes; no trained weights, logits or measured activations.',
      },
      options,
      frames,
    }
  },
}

export function nextFrame(
  trace: SimulationTrace,
  current: number,
  detail: 'operation' | 'stage',
  direction = 1,
) {
  let next = Math.max(0, Math.min(trace.frames.length - 1, current + direction))
  while (
    detail === 'stage' &&
    next > 0 &&
    next < trace.frames.length - 1 &&
    !trace.frames[next].milestone
  )
    next += direction
  return next
}
