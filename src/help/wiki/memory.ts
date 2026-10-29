import type { WikiArticle } from './types'
import { section } from './foundations'

export const memoryArticles: WikiArticle[] = [
  {
    id: 'attention-comparison',
    title: 'Compare attention mechanisms and memory',
    category: 'Attention & memory',
    summary:
      'Compare what is stored, what grows with context, which savings combine, and what changes the model.',
    aliases: [
      'attention types',
      'memory efficient attention',
      'less memory',
      'MHA vs GQA vs MQA vs MLA',
      'linear vs softmax',
      'attention memory comparison',
    ],
    kinds: ['cachek', 'cachev'],
    graphs: ['kv-cache'],
    prerequisites: ['attention-overview', 'kv-cache'],
    related: [
      'head-sharing',
      'mla',
      'linear-attention',
      'attention-patterns',
      'cache-quantization',
      'prefix-sharing',
      'memory-budget',
      'storage-kernels',
    ],
    sources: ['attention', 'gqa', 'mqa', 'mistral', 'deepseek', 'linear', 'kivi', 'flash', 'paged'],
    sections: [
      section('comparison-axis', 'First ask which memory is being reduced', [
        'Persistent state survives a decoder call. Working memory holds current inputs, temporary scores, outputs and kernel scratch. Parameters are a third category. A smaller attention workspace does not necessarily mean a smaller persistent cache, and a smaller cache does not imply smaller model weights.',
        'The comparison lab counts hypothetical persistent attention-state payload only. It does not execute any of these architectures, load weights, measure allocator usage or predict output quality. All rows use the declared layer count, batch, query-head count and value width; latent and recurrent widths are separate explicit assumptions. This makes storage mechanisms comparable without pretending the models are equivalent.',
      ]),
      section(
        'comparison-heads',
        'MHA, GQA and MQA keep the same kind of indexed state',
        [
          'For equal-width keys and values, MHA retains a K/V pair per query head, while GQA uses fewer distinct K/V heads and MQA uses one. They still retain a representation for each stored sequence position. The familiar factor of two counts separate keys and values, not two copies of one tensor.',
          'Holding all other dimensions fixed, GQA’s payload divided by MHA’s payload is Hkv/Hq. Sharing does not make queries identical and does not guarantee identical quality. At fixed hidden width, changing query-head count can change head width too; control those variables before quoting a ratio.',
        ],
        String.raw`M_{\rm KV}=2LBH_{kv}SD_hb,\qquad M_{\rm GQA}/M_{\rm MHA}=H_{kv}/H_q`,
      ),
      section('comparison-representations', 'MLA and recurrence change the state representation', [
        'In the cited DeepSeek MLA design, a joint key/value latent and a separate positional key are retained for each position. With common storage size b, their payload is LBS(dc+dR)b. There is no extra factor of two: the latent jointly supplies content for both keys and values. It remains sequence-growing, and savings depend on the chosen widths. Projection rearrangements are a separate execution question.',
        'The normalized kernel-linear example instead retains a feature-by-value accumulator and feature normalizer per head. With fixed feature width, its recurrent state does not grow with processed length. This is a different mixing rule from softmax attention; it does not retain an individually addressable exact K/V pair for every old position. Other recurrent or state-space designs require their own formulas.',
      ]),
      section('comparison-combine', 'Several improvements can coexist', [
        'Factorized accounting also shows when multiplying ratios is legitimate. For an otherwise identical full-prefix explicit cache, changing eight K/V heads to two changes the scalar count by one quarter. Encoding those scalars in eight rather than sixteen bits adds another ideal factor of one half, giving one eighth of the original code payload. Metadata, padding, residual high-precision regions and physical sharing must then be counted explicitly; this is not automatically one eighth of total memory.',
        'A GQA model can also use local attention in some layers, lower-precision KV storage, a tiled attention kernel and a paged allocator. Head sharing changes representation count; windows change retained positions; quantization changes encoding; FlashAttention reduces large temporary materializations and memory traffic; paging manages physical allocation and sharing.',
        'Do not multiply independently advertised savings blindly. A local cache has already discarded positions; shared physical blocks have reference counts; quantized groups need metadata. Kernel scratch and training activations are outside this persistent-state comparison. No row is an overall winner: capability, training compatibility, precision, kernels and workload all matter.',
      ]),
    ],
  },
  {
    id: 'attention-patterns',
    title: 'Full, local, sparse and cross-attention',
    category: 'Attention & memory',
    summary:
      'Visibility patterns decide which positions can interact; head sharing is a separate choice.',
    aliases: [
      'sliding window',
      'local attention',
      'global attention',
      'bidirectional attention',
      'cross attention',
      'encoder decoder',
      'sparse attention',
    ],
    kinds: [],
    graphs: [],
    prerequisites: ['causal-mask', 'cache-retention'],
    related: ['attention-comparison', 'head-sharing', 'prefill-decode', 'prefix-sharing'],
    sources: ['attention', 'mistral', 'sparse'],
    sections: [
      section('patterns-axes', 'Separate where attention looks from how heads share', [
        'Cross-attention has two different position spaces. A decoder query at target position 1 may read encoder source position 100 when that source is already available; the decoder’s target-causal condition is not an inequality between those two indexes. Source padding and the architecture’s source visibility still matter. An encoder-decoder model can therefore combine bidirectional source encoding, causal target self-attention and full source cross-attention without leaking future target IDs.',
        'Self-attention takes queries, keys and values from one sequence representation. Cross-attention takes queries from one stream and keys/values from another, such as decoder queries reading encoder outputs. Causal attention excludes future dependencies; bidirectional encoder attention can read both earlier and later input positions. These distinctions do not specify MHA, GQA or MQA.',
        'A full causal decoder can reuse an unchanged prefix’s state because later inputs cannot alter that prefix’s causal representations. Appending input to a bidirectional encoder can change earlier representations; decoder-style append-only reuse is therefore not generally valid. Fixed encoder outputs can supply reusable cross-attention K/V for each decoder layer, while decoder self-attention state still grows independently.',
      ]),
      section(
        'patterns-window',
        'Window size bounds direct access, not total learned knowledge',
        [
          'In this app’s inclusive window convention, a query at q reads keys k satisfying max(0,q-W+1) ≤ k ≤ q. The last W processed positions remain resident. Chunked prefill may temporarily need a larger working set so every query in the chunk sees its proper window; final retained payload is not peak working memory.',
          'Earlier positions may influence newer representations through intervening layers and steps. That indirect propagation does not mean the model can recover any evicted item exactly. A larger accepted input length, a direct attention window and measured long-range task performance are three different things.',
        ],
        String.raw`\mathcal V(q)=\{k:\max(0,q-W+1)\le k\le q\},\qquad R=\min(S,W)`,
      ),
      section('patterns-sparse', 'Sparse does not automatically mean fixed memory', [
        'Sparse patterns can include local bands, fixed strides, selected blocks or globally visible positions. State accounting follows what future queries may still request. Keeping a fixed local window plus an anchor every stride steps still accumulates anchors as context grows. Merely showing fewer nonzero cells does not prove constant persistent memory.',
        'Mixed local/global layers must be counted layer by layer. A full-context layer retains its own complete prefix even if neighboring layers are local. A block-sparse kernel, a sparse visibility rule and a cache eviction policy must agree; changing only the storage cannot preserve dependencies on discarded keys.',
      ]),
    ],
  },
  {
    id: 'linear-attention',
    title: 'Linear attention and recurrent state',
    category: 'Attention & memory',
    summary:
      'Some attention rules summarize the past into fixed-size accumulators rather than retaining every key and value.',
    aliases: [
      'recurrent attention',
      'kernel attention',
      'constant memory',
      'feature maps',
      'state space',
      'hybrid attention',
    ],
    kinds: [],
    graphs: [],
    prerequisites: ['attention-overview', 'linear-maps'],
    related: ['attention-comparison', 'memory-budget', 'mla', 'attention-patterns'],
    sources: ['linear', 'attention'],
    sections: [
      section(
        'linear-rule',
        'Change the similarity rule before changing the evaluation order',
        [
          'The cited linear-attention construction uses a kernel similarity written as a dot product between feature maps. For one causal head, let phi(k) have F features and v have Dv features. Keep a matrix C of shape [F,Dv] and a normalizer vector z of shape [F]. Each arriving key/value adds an outer product to C and its key features to z.',
          'A query reads the current accumulators. The numerator is its feature vector multiplied by C; the denominator is its dot product with z. Use a feature map and numerical policy that keep normalization valid; the formula assumes a positive denominator. This is one specific normalized construction, not a universal formula for every architecture called linear or recurrent.',
        ],
        String.raw`C_t=C_{t-1}+\phi(k_t)v_t^\top,\quad z_t=z_{t-1}+\phi(k_t),\quad o_t=\frac{\phi(q_t)^\top C_t}{\phi(q_t)^\top z_t}`,
      ),
      section(
        'linear-memory',
        'Fixed state exchanges individual history for accumulated features',
        [
          'The per-head persistent scalar count is F·Dv+F. Across L layers, B sequences and H independent heads, multiply by LBH and the accumulator’s storage size. The comparison lab assumes FP32 accumulation even when conventional KV is stored in FP16, and makes F explicit. Its state stays flat as S changes; it still grows with feature width, value width, heads, layers and batch.',
          'The constant-state recurrence holds for autoregressive inference under this fixed-dimensional rule. Training can retain sequence activations and use parallel algorithms; fixed inference state is not a claim of constant total training memory. Standard softmax of arbitrary query/key dot products is not replaced exactly by a small fixed feature map simply by reassociating multiplication.',
        ],
        String.raw`M_{\rm recurrent}=LBH(FD_v+F)b_{\rm accum}`,
      ),
      section('linear-worked-recurrence', 'Reconstruct a tiny recurrent read', [
        'Synthetic kernel example with two feature-map coordinates and scalar values: first key features [1,0] with value 2 create C=[2,0]ᵀ and z=[1,0]. A second key with features [0,1] and value 6 gives C=[2,6]ᵀ and z=[1,1]. For query features [1,3], the numerator is 20 and the denominator is 4, so the output is 5. These feature vectors are declared directly for the example; they are not learned checkpoint activations.',
        'The same output is the direct weighted average with kernel weights [1,3], normalized to [1/4,3/4]. C contains two scalars and z two more regardless of how many keys have contributed. F here is the feature-map width, locally defined; it is not the feed-forward intermediate width also called F elsewhere. Reading empty zero-initialized state would have a zero denominator, so a real implementation needs a valid first contribution and a numerical policy.',
      ]),
      section('linear-hybrids', 'Gates, decay and hybrid layers need explicit contracts', [
        'Other designs add decay, gates, delta-style updates or distinct state-space equations. Those alter what the state represents and how it changes. They may use additional convolution state or normalizers. Do not reuse this accumulator formula by name alone, or describe a state-space model as merely a smaller softmax cache.',
        'A hybrid can alternate recurrent layers with full-context attention. Its payload is a sum: the fixed-state layers contribute a length-independent term, while every full-context layer contributes its own sequence-growing cache. The lab’s hybrid is hypothetical, with a control for the number of full-attention layers; it is not a simulation of a named trained architecture.',
      ]),
    ],
  },
  {
    id: 'cache-quantization',
    title: 'KV cache quantization and precision',
    category: 'Attention & memory',
    summary:
      'Fewer bits per stored value save payload, but groups, metadata, residual buffers and approximation error matter.',
    aliases: [
      'KV quantization',
      '4 bit',
      '2 bit',
      '8 bit',
      'KIVI',
      'low precision cache',
      'weight quantization',
    ],
    kinds: [],
    graphs: [],
    prerequisites: ['kv-cache', 'cache-bytes'],
    related: ['attention-comparison', 'memory-budget', 'storage-kernels'],
    sources: ['kivi', 'attention'],
    sections: [
      section(
        'quant-encoding',
        'Encoding changes the numbers that survive a call',
        [
          'An illustrative affine quantizer maps a range to integer codes and reconstructs approximate values using a scale and offset. Packing lower-bit codes reduces scalar payload, but the runtime must also store the information needed to decode them. Small groups can improve local range adaptation while adding more metadata.',
          'In the displayed formula, a is the positive spacing between reconstruction levels, c the lowest represented value and q the code bit count. With q=2, a=0.5 and c=−1, the four levels are [−1,−0.5,0,0.5]. An input of 0.3 rounds to 0.5, an error of 0.2. Rounding error is bounded by half a step only when clipping is unnecessary; an input of 2 clips to 0.5 and has much larger error.',
          'For G groups of n padded scalar slots, q bits per code and m metadata bytes per group, the illustrative byte count is G(ceil(nq/8)+m). The lab uses n=64, m=4 and separate groups per K/V layer/batch/head plane. It includes padding in each final group but omits residual buffers. This is a fictional accounting example, not the KIVI implementation.',
        ],
        String.raw`\widehat x=a\cdot\operatorname{clip}\!\left(\operatorname{round}\frac{x-c}{a},0,2^q-1\right)+c,\quad a>0`,
      ),
      section('quant-error', 'Key errors and value errors enter at different places', [
        'Perturbing a key changes a query/key score, which can change the normalized probability row. Perturbing a value changes the weighted output directly even if that row is held fixed. Quantizer grouping, clipping, outliers and precision affect these errors differently. Storage-bit ratios alone cannot predict downstream quality.',
        'KIVI studies asymmetric grouping for keys and values and retains a recent higher-precision residual region. It is a concrete example of why one uniform bits-per-scalar calculation is incomplete. A faithful estimate must specify group axes, scales/offsets, residual region, packing and supported kernels for the chosen implementation.',
      ]),
      section('quant-distinct', 'Weight precision is not cache precision', [
        'Model weights persist across unrelated requests. KV activations depend on a request’s prefix. Quantizing weights does not automatically quantize its cache, and a model with small weights can still have a large full-context KV state. A precision label also need not match accumulation precision or an actual runtime’s allocation.',
        'The app’s dtype-bytes controls have always been accounting assumptions rather than an executed quantizer. This new comparison likewise estimates storage only. Validate errors against a high-precision reference and measure actual kernels before claiming speed, exact equivalence or a quality-preserving compression ratio.',
      ]),
    ],
  },
  {
    id: 'prefix-sharing',
    title: 'Shared prefixes, paging and cache reuse',
    category: 'Attention & memory',
    summary:
      'Identical compatible prefixes can share physical state; sharing is different from shrinking the state of one unique sequence.',
    aliases: [
      'prefix caching',
      'prompt caching',
      'paged attention',
      'copy on write',
      'beam search',
      'shared cache',
    ],
    kinds: [],
    graphs: [],
    prerequisites: ['kv-cache', 'storage-kernels'],
    related: ['attention-comparison', 'memory-budget', 'attention-patterns'],
    sources: ['paged', 'attention'],
    sections: [
      section('prefix-identity', 'The same text is not a sufficient cache key', [
        'Reuse requires matching token IDs, model parameters and relevant adapters, position conventions, attention rules and other state-producing settings. For a causal decoder, matching an earlier prefix can make its derived K/V reusable. A different preceding context can change a token’s representation even if the token’s own text matches.',
        'Serving systems decide cache identity and isolation explicitly. This app explains sharing but does not introduce a cross-request cache, remote inference, or implicit access to another user’s state. A numerical baseline is also not a reusable serving cache: it is an inspection capture with its own provenance.',
      ]),
      section(
        'prefix-count',
        'Count unique physical blocks as well as logical per-request views',
        [
          'Let C_p count common-prefix tokens and S_u count private suffix tokens per request, for N compatible requests. The unique full-cache payload is proportional to C_p+N S_u with physical prefix sharing, compared with N(C_p+S_u) without it. These symbols deliberately differ from the runtime’s retained-prefix P and working-key U. Multiply by the per-position layer/head/feature payload. The idealized count excludes partially filled pages, block tables and copy-on-write effects.',
          'Synthetic count: four requests share an eight-token prefix and each has a private two-token suffix. Separate storage holds 4×(8+2)=40 position-copies; sharing holds 8+4×2=16. The saving is 24 position-copies, or 60% of this payload. Each request still logically sees ten positions; the reduction is in duplicate physical storage.',
          'Each request still has its logical prefix; sharing reduces duplicate physical storage. Once a request writes a divergent continuation, immutable shared blocks can remain shared while writable parts need ownership rules. PagedAttention describes block mapping, sharing and copy-on-write machinery rather than a new learned similarity rule.',
        ],
        String.raw`M_{\rm separate}=cN(C_p+S_u),\qquad M_{\rm shared}=c(C_p+NS_u),\quad c=2LH_{kv}D_hb`,
      ),
      section('prefix-limits', 'Paging reduces waste without erasing context growth', [
        'Fixed-size blocks avoid reserving one large contiguous maximum-length region for every request. They improve allocation flexibility and can support sharing, but a unique full-context sequence still needs its retained scalar state. Fragmentation savings depend on workload, block sizes and allocator policy.',
        'Prefill reuse can avoid recomputing compatible prefix state. It does not generally remove later queries’ need to read that state for attention. Capacity, compute saved by reuse, decode bandwidth and elapsed latency are separate measurements. Short shared prefixes or highly divergent requests can provide much less benefit.',
      ]),
    ],
  },
  {
    id: 'memory-budget',
    title: 'Attention memory budgets and scaling',
    category: 'Attention & memory',
    summary:
      'Account separately for weights, persistent state, temporary tensors, precision and physical overhead.',
    aliases: [
      'VRAM',
      'GPU memory',
      'quadratic memory',
      'prefill memory',
      'decode bandwidth',
      'batch memory',
      'hybrid models',
      'out of memory',
    ],
    kinds: [],
    graphs: [],
    prerequisites: ['parameters-activations-state', 'cache-bytes'],
    related: [
      'attention-comparison',
      'linear-attention',
      'cache-quantization',
      'prefix-sharing',
      'storage-kernels',
    ],
    sources: ['attention', 'flash', 'paged', 'linear', 'deepseek'],
    sections: [
      section('budget-ledger', 'Use a ledger, not one unexplained memory number', [
        'A synthetic scale check matches the comparison lab: one sequence, 32 layers, 32 K/V heads, width 128, 4,096 retained positions and two bytes per scalar use 2 GiB of full MHA K/V payload. Eight K/V heads use 512 MiB with the other dimensions fixed. Separately, one head’s one-layer materialized 4,096-by-4,096 probability matrix at two bytes per scalar uses 32 MiB. Doubling sequence length doubles those full-cache payloads but quadruples that full-prefix matrix; the components have different scopes and lifetimes.',
        'A runtime budget includes parameter storage, persistent per-sequence attention state, current-call activations, attention workspace, allocator reservations and metadata. Training adds gradients, optimizer state and saved or recomputed activations. The visualizer deliberately retains historical tensors for learning, which adds a cost an efficient inference engine need not pay.',
        'For explicit equal-width K/V at uniform layers and retained extent R, persistent payload is 2LBHkvRDh times storage bytes. Doubling batch or retained positions doubles this payload at fixed architecture. In a serving workload, requests can have different lengths and shared blocks, so sum their actual unique state rather than treating every request as a maximum-sized rectangle.',
      ]),
      section(
        'budget-quadratic',
        'Quadratic temporary matrices and linear KV state are different costs',
        [
          'A materialized score or probability tensor has B·Hq·T·U entries for the current call. Full-prefix prefill has T=U=S, so that particular tensor grows quadratically with prompt length. Ordinary cached single-position decode has T=1, giving a row linear in working-key count. One tensor’s size is not total peak memory: scores, probabilities, inputs and outputs may overlap differently by implementation.',
          'FlashAttention tiles the operation and uses online normalization to avoid storing a full attention matrix in device memory. It preserves the intended dense attention rule, with floating-point differences possible, while full-context pairwise arithmetic remains quadratic in prefill. It does not by itself turn persistent KV into fixed-size recurrent state.',
        ],
        String.raw`N_{\rm scores}=BH_qTU,\qquad \text{prefill: }T=U=S,\quad \text{ordinary decode: }T=1`,
      ),
      section('budget-layerwise', 'Mixed architectures need a layerwise sum', [
        'For heterogeneous layers, sum the actual stored components: full or local K/V, joint latents, recurrent accumulators and additional positional or convolution state. In a full/recurrent hybrid, even a small number of full-context layers leaves a sequence-growing term. The memory curve can have a fixed intercept plus a positive slope rather than being entirely flat.',
        'A long-context configuration does not prove useful long-context reasoning, nor does smaller payload guarantee lower latency. Full-context decode can be limited by repeated reads of persistent state, while other stages depend on compute, scheduling or communication. Use measured workload-specific evidence for performance claims; the lab’s curves are arithmetic, not benchmarks.',
      ]),
    ],
  },
]
