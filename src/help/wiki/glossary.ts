export type GlossaryTerm = {
  id: string
  term: string
  definition: string
  article: string
  aliases?: string[]
  symbol?: string
}

/** Definitions use this workbench's conventions. The linked guides provide
 * full equations, scope notes and primary references. */
export const glossary: GlossaryTerm[] = [
  {
    id: 'adam',
    term: 'Adam and moment buffers',
    definition:
      'An adaptive optimizer that tracks first and second raw gradient moments, corrects zero-initialization bias and scales coordinate-wise updates. Its buffers are training state, not inference K/V.',
    article: 'adam',
    aliases: ['beta1', 'beta2', 'bias correction'],
  },
  {
    id: 'adamw',
    term: 'AdamW and decoupled decay',
    definition:
      'Adam using loss-gradient moments plus a separate parameter shrinkage. Adding an L2 penalty to Adam gradients changes the moments and is generally a different update.',
    article: 'adamw',
    aliases: ['weight decay', 'L2 regularization'],
  },
  {
    id: 'muon',
    term: 'Muon and matrix update geometry',
    definition:
      'A matrix-aware optimizer design that transforms momentum candidates using finite approximate Newton–Schulz iterations. Eligible parameter groups and auxiliary optimizers remain explicit.',
    article: 'muon',
    aliases: ['polar factor', 'Newton Schulz'],
  },
  {
    id: 'backpropagation',
    term: 'Backpropagation and gradients',
    definition:
      'Computing loss derivatives backward through an executed graph using the chain rule. The optimizer uses these derivatives to choose parameter updates; inference normally keeps parameters fixed.',
    article: 'learning-loop',
    aliases: ['SGD', 'learning rate'],
  },
  {
    id: 'alibi',
    term: 'ALiBi and position methods',
    definition:
      'ALiBi adds head-dependent distance biases to attention scores. Learned/sinusoidal additions and RoPE rotations insert position at different locations in the graph.',
    article: 'position-methods',
  },
  {
    id: 'superposition',
    term: 'Superposition and polysemanticity',
    definition:
      'Overlapping representations can encode more feature directions than coordinate dimensions, with interference. A coordinate responding to several properties does not have a unique semantic label.',
    article: 'feature-representations',
  },
  {
    id: 'activation-patching',
    term: 'Activation patching',
    definition:
      'Replacing a specified activation with a compatible value from another run, then recomputing downstream effects. The source, background and outcome metric determine the claim.',
    article: 'causal-interventions',
    aliases: ['ablation', 'causal tracing'],
  },
  {
    id: 'post-training',
    term: 'SFT, RLHF and DPO',
    definition:
      'Different post-training data/objective choices: demonstrations, reward-model policy optimization, or direct policy/reference preference loss. These are distinct from inference-time prompting.',
    article: 'post-training',
  },
  {
    id: 'precision',
    term: 'Floating-point precision',
    definition:
      'Finite numerical representation with rounding and bounded range. Storage format, accumulation precision and operation order can all affect a result.',
    article: 'numerical-precision',
    aliases: ['FP32', 'FP16', 'BF16'],
  },
  {
    id: 'layernorm',
    term: 'LayerNorm versus RMSNorm',
    definition:
      'LayerNorm subtracts the feature mean and divides by the square root of centered variance plus epsilon; RMSNorm divides the uncentered row by the square root of mean square plus epsilon. Gains and optional bias are separate parameters.',
    article: 'rmsnorm',
  },
  {
    id: 'routed-experts',
    term: 'Dense versus routed experts',
    definition:
      'Dense layers use their specified transformation for each token; MoE routing selects and combines expert paths. Stored and activated expert parameters are different counts.',
    article: 'mixture-of-experts',
  },
  {
    id: 'linear-attention',
    term: 'Linear / recurrent attention',
    definition:
      'An attention family that can use structured running state. The linked normalized kernel example keeps a feature-by-value accumulator and normalizer; it is not exact arbitrary softmax.',
    article: 'linear-attention',
    aliases: ['constant state', 'kernel feature map'],
  },
  {
    id: 'mla',
    term: 'MLA · multi-head latent attention',
    definition:
      'An architecture with joint compressed K/V content and separate positional state in the cited DeepSeek design. Its cache representation is distinct from GQA.',
    article: 'mla',
    aliases: ['latent attention'],
  },
  {
    id: 'kv-quantization',
    term: 'KV quantization',
    definition:
      'Encoding input-dependent cached keys/values with fewer bits. Count group metadata, padding and any higher-precision residual state as well as packed codes.',
    article: 'cache-quantization',
    aliases: ['cache precision', 'KIVI'],
  },
  {
    id: 'prefix-sharing',
    term: 'Prefix sharing',
    definition:
      'Reusing compatible causal prefix state across requests or branches, potentially with one physical copy and multiple logical views. Identity includes more than visible text.',
    article: 'prefix-sharing',
  },
  {
    id: 'cross-attention',
    term: 'Cross-attention',
    definition:
      'Queries read keys and values from a different representation stream, such as a decoder reading encoder output. Source-state memory is separate from decoder self-attention state.',
    article: 'attention-patterns',
  },
  {
    id: 'hybrid-attention',
    term: 'Hybrid attention model',
    definition:
      'A model combining layers with different mixing/state mechanisms. Count each layer’s state; some full-context layers can keep total state growing with sequence length.',
    article: 'memory-budget',
  },
  {
    id: 'batch',
    term: 'B · batch size',
    symbol: 'B',
    definition:
      'Separate sequence instances processed together, without mixing their attention calculations across this axis. This does not assume their data are statistically independent. Numerical model execution fixes B to 1; hypothetical calculators may vary it.',
    article: 'tensors',
  },
  {
    id: 'queries',
    term: 'T · current query positions',
    symbol: 'T',
    definition:
      'Positions consumed by this forward call: multiple prompt positions during prefill, usually one during ordinary cached decoding. Verification can consume several.',
    article: 'prefill-decode',
  },
  {
    id: 'extent',
    term: 'S · processed sequence extent',
    symbol: 'S',
    definition:
      'The absolute sequence extent including the current call. Eviction can reduce retained state without reducing this extent or restarting position IDs.',
    article: 'cache-retention',
  },
  {
    id: 'resident',
    term: 'R · retained positions',
    symbol: 'R',
    definition:
      'Positions retained after this call under the cache policy. R can differ from S and from the number of working key slots used while evaluating the call.',
    article: 'cache-retention',
  },
  {
    id: 'prefix',
    term: 'P · retained prefix',
    symbol: 'P',
    definition:
      'Retained positions before the current call. An initial prefill has P = 0. This is a count of available prefix slots, not the largest absolute position ID.',
    article: 'cache-retention',
  },
  {
    id: 'working',
    term: 'U · working key slots',
    symbol: 'U',
    definition:
      'The P retained prefix entries plus T new positions: U = P + T. Working keys must remain available until every query that needs them has finished.',
    article: 'cache-retention',
  },
  {
    id: 'width',
    term: 'D · residual width',
    symbol: 'D',
    definition:
      'Feature coordinates carried by each token in the residual stream. In this supported profile, D = Hq × Dh. Width does not count tokens or layers.',
    article: 'tensors',
  },
  {
    id: 'query-heads',
    term: 'Hq · query heads',
    symbol: 'Hq',
    definition:
      'Separate query-head computations. Each has its own score and probability row, including when heads share K/V. Separate computation does not imply statistically independent activations.',
    article: 'head-sharing',
  },
  {
    id: 'kv-heads',
    term: 'Hkv · key/value heads',
    symbol: 'Hkv',
    definition:
      'Distinct K/V head pairs per layer. This axis indexes dense cached state; query heads sharing a pair do not allocate separate copies of that state.',
    article: 'head-sharing',
  },
  {
    id: 'head-width',
    term: 'Dh · features per head',
    symbol: 'Dh',
    definition:
      'Matching Q/K feature width contracted in a dot product and the value width in this profile. Dh is even for adjacent rotary feature pairs.',
    article: 'attention-scores',
  },
  {
    id: 'ff-width',
    term: 'F · intermediate width',
    symbol: 'F',
    definition:
      'Feature width inside the feed-forward network. Gate/up projections map D to F; the down projection maps F back to D. F is larger than D in the bundled profiles, but that is not the definition of an intermediate axis.',
    article: 'swiglu',
  },
  {
    id: 'vocab-size',
    term: 'V · vocabulary size',
    symbol: 'V',
    definition:
      'Number of candidate vocabulary IDs in the output projection. Elsewhere the letter V can name the attention value tensor; read its surrounding axes and context.',
    article: 'logits',
  },
  {
    id: 'layers',
    term: 'L · decoder layers',
    symbol: 'L',
    definition:
      'Successive independent blocks in this profile. Repeated structure does not imply shared parameters or shared cache contents.',
    article: 'residual',
  },
  {
    id: 'capacity',
    term: 'C · configured context capacity',
    symbol: 'C',
    definition:
      'The supported profile’s declared sequence capacity. A larger setting is not evidence that a trained checkpoint remains accurate at that length.',
    article: 'workspace',
  },
  {
    id: 'token',
    term: 'Token',
    aliases: ['word', 'piece'],
    definition:
      'A vocabulary ID associated with a piece of encoded input. It can represent a word, part of a word, punctuation or a byte fragment; token count need not equal character count.',
    article: 'tokenization',
  },
  {
    id: 'bpe',
    term: 'Byte-pair encoding',
    aliases: ['BPE', 'merges', 'tokenizer'],
    definition:
      'A segmentation algorithm that repeatedly combines eligible neighboring pieces according to merge ranks. The teaching tokenizer has an explicit small byte vocabulary and is not a production tokenizer.',
    article: 'tokenization',
  },
  {
    id: 'tensor',
    term: 'Tensor',
    aliases: ['array', 'matrix', 'vector', 'axes'],
    definition:
      'An array with ordered axes and semantic labels. A matching shape alone does not prove that two tensors have compatible meaning, positions or contraction rules.',
    article: 'tensors',
  },
  {
    id: 'contraction',
    term: 'Contraction',
    aliases: ['dot', 'matmul', 'sum'],
    definition:
      'Multiply matching coordinates and sum over an axis. QKᵀ sums head features; attention-weighted V sums key positions. They eliminate different axes.',
    article: 'linear-maps',
  },
  {
    id: 'transpose',
    term: 'Transpose',
    definition:
      'A permutation of tensor axes. It changes indexing or display orientation rather than learning a new projection. The microscope’s transpose control changes the view.',
    article: 'tensors',
  },
  {
    id: 'reshape',
    term: 'Reshape',
    definition:
      'A regrouping of coordinates with a specified element order, such as splitting projected features into heads. It must preserve the number of elements.',
    article: 'tensors',
  },
  {
    id: 'parameter',
    term: 'Parameter',
    aliases: ['weight', 'gain'],
    definition:
      'A coefficient or tensor defining the model’s transformation. Parameters remain fixed during a forward run. Seeded untrained parameters and learned checkpoint parameters have different provenance.',
    article: 'parameters-activations-state',
  },
  {
    id: 'activation',
    term: 'Activation',
    definition:
      'A value computed for the current input, such as a hidden state or attention probability. An activation is not automatically a learned weight or a human-readable concept.',
    article: 'parameters-activations-state',
  },
  {
    id: 'embedding',
    term: 'Embedding lookup',
    aliases: ['gather'],
    definition:
      'Gather the parameter row indexed by a vocabulary ID. Repeated IDs start with the same token embedding but can develop different contextual hidden states.',
    article: 'embeddings',
  },
  {
    id: 'residual',
    term: 'Residual addition',
    aliases: ['skip connection'],
    definition:
      'Add a sublayer update element by element to the ongoing stream. The identity path carries the original stream around the normalized computation branch.',
    article: 'residual',
  },
  {
    id: 'rms',
    term: 'RMSNorm',
    aliases: ['normalisation', 'normalization'],
    definition:
      'Divide a token’s features by the root of their mean square plus epsilon, then apply feature gains. It does not subtract the feature mean.',
    article: 'rmsnorm',
  },
  {
    id: 'query',
    term: 'Query, key and value',
    aliases: ['QKV', 'Q', 'K'],
    definition:
      'Queries compare with keys to form scores. Normalized score rows weight value vectors. These are mathematical roles of distinct projections, not literal questions and dictionary addresses.',
    article: 'attention-overview',
  },
  {
    id: 'mask',
    term: 'Causal mask',
    aliases: ['future', 'visibility'],
    definition:
      'A dependency rule excluding future keys before softmax. Excluded scores are −∞ so their exponentials are zero; a zero score would still receive probability.',
    article: 'causal-mask',
  },
  {
    id: 'softmax',
    term: 'Softmax',
    definition:
      'Exponentiate relative scores and divide by their row sum. Subtracting the row maximum improves numerical stability without changing the exact distribution.',
    article: 'softmax',
  },
  {
    id: 'rope',
    term: 'RoPE',
    aliases: ['rotary', 'positional embedding'],
    definition:
      'Position-dependent rotations of query/key feature pairs. The reference profile uses absolute positions, adjacent pairs and base 10,000; values are not rotated.',
    article: 'rope',
  },
  {
    id: 'gqa',
    term: 'GQA / MHA / MQA',
    aliases: ['grouped query', 'multi query', 'multi head'],
    definition:
      'Head-sharing schemes. MHA has one K/V head pair per query head; MQA shares one K/V head pair across all query heads; GQA uses intermediate equal groups in this profile. Every pair still contains state for retained positions.',
    article: 'head-sharing',
  },
  {
    id: 'kv',
    term: 'KV cache',
    aliases: ['persistent state', 'cached keys', 'cached values'],
    definition:
      'Input-dependent K/V state indexed by layer, batch, KV head, retained position and feature, plus absolute position metadata. It is separate from learned model parameters.',
    article: 'kv-cache',
  },
  {
    id: 'silu',
    term: 'SiLU and SwiGLU',
    aliases: ['swish', 'Hadamard', 'gate'],
    definition:
      'SiLU(z) = z·sigmoid(z). SwiGLU multiplies the activated gate branch by a separate up branch coordinate by coordinate, then applies a down projection.',
    article: 'swiglu',
  },
  {
    id: 'logit',
    term: 'Logit',
    aliases: ['output score'],
    definition:
      'An unnormalized vocabulary score. Logits can be negative and need not sum to one. Vocabulary probabilities arise after the configured normalization/selection process.',
    article: 'logits',
  },
  {
    id: 'temperature',
    term: 'Temperature',
    definition:
      'A positive divisor of vocabulary logits before softmax. It changes relative probability concentration while preserving the ordering of finite logits.',
    article: 'sampling',
  },
  {
    id: 'prefill',
    term: 'Prefill and decode',
    definition:
      'Prefill consumes a prompt as a causal batch. Ordinary decode consumes a selected next ID and predicts the following one using retained prefix state.',
    article: 'prefill-decode',
  },
  {
    id: 'loss',
    term: 'Cross-entropy',
    aliases: ['loss', 'teacher forcing'],
    definition:
      'For one observed target ID, negative log probability assigned to that target. Training aggregates such objectives and updates parameters; inference in this app does not train.',
    article: 'training-objective',
  },
  {
    id: 'mtp',
    term: 'Multi-token prediction',
    aliases: ['MTP'],
    definition:
      'Auxiliary prediction of farther-future targets. Independent-head and sequential designs differ. A training objective does not by itself specify a verified inference algorithm.',
    article: 'mtp',
  },
  {
    id: 'rollback',
    term: 'Speculative verification and rollback',
    definition:
      'Verify draft proposals with the main decoder, accept a valid prefix, and discard invalid suffix state while retaining a recoverable cache checkpoint.',
    article: 'speculative-decoding',
  },
  {
    id: 'kernel',
    term: 'Kernel',
    aliases: ['FlashAttention'],
    definition:
      'An implementation strategy for executing a calculation on hardware. A kernel can tile or fuse logical operations without materializing every conceptual matrix.',
    article: 'storage-kernels',
  },
  {
    id: 'paging',
    term: 'Physical KV storage',
    aliases: ['PagedAttention', 'paging', 'ring buffer'],
    definition:
      'Allocation and addressing of logical K/V state. Pages or wrapping slots do not replace absolute position IDs or change the head-sharing definition.',
    article: 'storage-kernels',
  },
]
