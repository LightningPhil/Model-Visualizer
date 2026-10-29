import { detail, introduction, type TopicPerspectives } from './perspective-types'

export const memoryPerspectives: Record<string, TopicPerspectives> = {
  'attention-comparison': {
    introduction: introduction(
      'There are several ways to reduce attention memory, and they save different things. Some share representations among heads, some retain fewer positions, some store a compact representation, and some change the attention rule so the past can be accumulated into fixed-size state. Understanding what remains available is more useful than memorizing a list of model names.',
      'Imagine managing a collection of measurements. You could avoid keeping repeated copies, keep only recent measurements, encode each measurement more compactly, or replace individual measurements with running summaries. These choices have different consequences. Faster processing and more efficient allocation are additional choices, rather than interchangeable kinds of compression.',
      [
        'Identify the persistent information each method retains.',
        'Compare the same hypothetical layer, batch, head and feature dimensions.',
        'Increase context length and see which state grows or levels off.',
        'Read the tradeoff before interpreting a smaller bar as a better model.',
      ],
      'The lab estimates hypothetical state, not trained-model quality or actual GPU allocation. MLA and recurrent attention need different state contracts from explicit K/V. FlashAttention reduces temporary materialization and memory traffic; paging improves physical allocation and sharing. Neither automatically shrinks the logical cache for one unique full-context sequence.',
    ),
    technical: [
      detail(
        'technical-comparison-controls',
        'State the controlled variables before comparing ratios',
        [
          'The head-sharing rows keep query-head count and per-head feature width fixed. The latent row explicitly chooses its joint latent and separate positional-key widths; it does not inherit a fictitious K/V-head count. The recurrent row assumes independent heads and FP32 matrix/vector accumulators for the stated normalized kernel rule.',
          'As a hand-checkable synthetic example, one layer, one sequence, four heads, two features and eight positions at two bytes per scalar gives MHA 256 bytes, two-head GQA 128 bytes and MQA 64 bytes. A two-position GQA window gives 32 bytes. These are different architecture/state assumptions, not interchangeable activations or promised compression ratios for a real checkpoint.',
        ],
      ),
    ],
  },
  'attention-patterns': {
    introduction: introduction(
      'An attention pattern decides which positions can supply information to a query. A causal pattern reads the past, a local pattern restricts direct reach, a sparse pattern selects a structured subset, and cross-attention reads a different stream. The number of shared key/value heads is a separate choice.',
      'Think of choosing which participants are allowed to contribute to each discussion. One rule allows everyone who has spoken so far; another allows only recent speakers; another also allows selected earlier contributors. The rule determines access, while the head-sharing design determines how the contributions are represented.',
      [
        'Identify which stream supplies queries and which supplies keys and values.',
        'Check whether future positions are excluded.',
        'Follow the query’s permitted positions before normalizing scores.',
        'Retain the state that future queries may still need, and count mixed layers separately.',
      ],
      'Discarded positions can still have indirect influence through later representations, but that is not a guarantee of exact retrieval. A bidirectional encoder cannot generally reuse append-only prefix state under the same argument as a causal decoder. The local numerical profiles here remain causal self-attention implementations.',
    ),
    technical: [
      detail(
        'technical-cross-state',
        'Cross-attention adds a distinct source-length term',
        [
          'For hypothetical decoder cross-attention with fixed source length E, equal key/value width Dh and Hkv cross-attention heads per layer, explicit cached source K/V contributes 2LBHkvEDh times storage bytes. Decoder self-attention contributes a separate term using its own retained target length and dimensions.',
          'These terms may have different layer counts, widths and sharing. Do not substitute the target context length into both. Encoder outputs themselves, current queries and kernel workspace are additional allocations. Source-state reuse requires identical compatible encoder representations and projections.',
        ],
        String.raw`M_{\rm cross}=2L_{\rm cross}BH_{kv,\rm cross}ED_{h,\rm cross}b`,
      ),
    ],
  },
  'linear-attention': {
    introduction: introduction(
      'Some attention mechanisms keep running summaries of earlier features instead of retaining a separate key and value for every position. Each new token updates the summary. A later query reads that summary to form its output. With fixed summary dimensions, persistent state can stay the same size even as the sequence gets longer.',
      'A running total lets you answer certain questions about a long history without retaining every individual observation. It cannot answer every question that a complete record could answer. The analogy is about what a summary preserves; the model’s summary is a structured numerical matrix and vector rather than a human-written account.',
      [
        'Map the new key into the mechanism’s feature representation.',
        'Use it and the value to update the accumulated state.',
        'Read that state using the current query’s features and normalization.',
        'Keep the updated state for the next step instead of appending another complete K/V record.',
      ],
      'This article explains one normalized kernel-linear formulation. It does not reproduce arbitrary softmax attention exactly or describe every gated recurrent or state-space model. Fixed inference state does not imply fixed total training memory, unlimited exact recall, or equal quality to another architecture.',
    ),
    technical: [
      detail(
        'technical-associativity',
        'Why the accumulator can replace a sequence sum',
        [
          'With similarity phi(q)ᵀphi(k), distribute the query feature vector across the sum of key-feature/value outer products. That sum is exactly the recurrent matrix for this kernel rule; the sum of key features supplies its normalizer. The algebra is exact for the declared feature map and valid denominator, apart from numerical arithmetic.',
          'The crucial assumption is finite feature-map width F. The usual exponential softmax dot-product kernel is not made exactly finite-dimensional for arbitrary inputs simply by writing this factorization. Replacing the kernel changes the model; approximating a kernel needs its own accuracy argument.',
        ],
        String.raw`\sum_{j\le t}(\phi(q_t)^\top\phi(k_j))v_j^\top=\phi(q_t)^\top\!\left(\sum_{j\le t}\phi(k_j)v_j^\top\right)`,
      ),
    ],
  },
  'cache-quantization': {
    introduction: introduction(
      'KV quantization stores earlier attention keys and values using a compact numerical encoding. The runtime keeps enough extra information to decode approximate values later. This can reduce memory per position without discarding the position itself, but the reconstructed numbers can differ from their higher-precision originals.',
      'Imagine recording measurements using a limited collection of allowed levels. Nearby values can then receive the same code, and values beyond the available range may be clipped. If each group uses its own scale and offset, those must be stored too. This describes the affine example’s codebook tradeoff without confusing it with merely changing the printed decimal precision.',
      [
        'Choose how values are grouped and how each group’s range is represented.',
        'Encode and pack lower-precision codes with the required metadata.',
        'Reconstruct the needed values or use a compatible low-precision kernel.',
        'Compare numerical error and task behavior with a higher-precision reference.',
      ],
      'Weight quantization and KV quantization apply to different tensors. Fewer storage bits do not imply the same accumulation precision, a fixed speedup or unchanged output quality. The lab estimates fictional packed groups and metadata; it does not execute KIVI or quantize the model.',
    ),
    technical: [
      detail(
        'technical-quant-sensitivity',
        'Track the path from an encoding error to attention output',
        [
          'For fixed query q, a key error delta-k changes its scaled score by qᵀdelta-k divided by the square root of head width. That score error participates in the entire softmax denominator. With probabilities held fixed, value errors instead accumulate as their probability-weighted sum.',
          'This separates two local mechanisms without pretending either captures a full end-to-end quantized model. Both keys and values may change together, and subsequent nonlinear layers can amplify or damp perturbations. Quality claims require the actual encoding, kernels, checkpoint and evaluation.',
        ],
        String.raw`\Delta s_j=\frac{q^\top\Delta k_j}{\sqrt{D_h}},\qquad \Delta o\big|_{A\ \mathrm{fixed}}=\sum_j A_j\Delta v_j`,
      ),
    ],
  },
  'prefix-sharing': {
    introduction: introduction(
      'When several requests begin with the same compatible token prefix, a serving system can reuse earlier computed state. It may keep one physical copy of that prefix while giving each request its own logical view and separate continuation. This saves duplicated work or storage across requests.',
      'Several readers can refer to the same immutable edition of a document while keeping separate notes. Sharing the original avoids copying it for everyone; their different notes still need separate space. This is an ownership analogy, not a suggestion that cached keys and values are readable documents.',
      [
        'Verify that model, tokens, positions and state-producing settings match.',
        'Find or compute the reusable prefix state.',
        'Share compatible immutable blocks with explicit ownership.',
        'Allocate divergent continuation state safely and release blocks only when no user still needs them.',
      ],
      'Matching visible text alone does not establish cache compatibility. Paging can reduce allocation waste and enable sharing, but does not make one unique full-context sequence have constant state. This app documents these mechanisms without implementing cross-request cache sharing.',
    ),
    technical: [
      detail(
        'technical-sharing-limit',
        'Sharing has a workload-dependent ceiling',
        [
          'Subtract the idealized shared count from the duplicated count: saved payload is the per-position payload times (N−1)C_p, where C_p is the common-prefix length. With no common prefix there is no such saving. With zero private suffix, the ideal payload ratio is one over request count, before metadata and allocation effects.',
          'Real systems share compatible complete blocks and may need copy-on-write for partially shared writable state. Cache identity must include all inputs that affect the cached representations and respect isolation policy. Attention reads for each later query remain necessary even when their prefix storage is physically shared.',
        ],
        String.raw`M_{\rm separate}-M_{\rm shared}=c(N-1)C_p`,
      ),
    ],
  },
  'memory-budget': {
    introduction: introduction(
      'A model can run out of memory for several different reasons. Its weights occupy space, each active sequence needs persistent state, the current calculation needs working space, and the runtime has allocation overhead. The visualizer also retains extra history so you can inspect earlier calculations.',
      'Think of a workshop with permanent equipment, ongoing projects, a working surface and storage organization. Reducing what one project retains is different from clearing the working surface or sharing equipment. An honest memory explanation identifies which part gets smaller and which parts remain.',
      [
        'Separate parameters, live persistent state and temporary calculations.',
        'Identify whether you are studying prompt processing, one decode step or training.',
        'Count actual state components and storage precision for each layer.',
        'Measure the real runtime before translating an arithmetic estimate into a hardware claim.',
      ],
      'The comparison curves count state payload, not total VRAM, speed or model capability. Long accepted context does not guarantee successful long-range reasoning. A tiled exact-attention kernel can reduce temporary storage while full-context KV still grows with length.',
    ),
    technical: [
      detail('technical-scaling-check', 'Use doubling experiments to identify the growing term', [
        'At fixed dimensions, full KV doubles when retained length doubles. A fixed window saturates after its window is full. The stated recurrent accumulator stays constant with sequence length. A hybrid with any full-context layers has a fixed component plus a growing component. A materialized full-prefix score tensor quadruples when both query and key lengths double.',
        'These are component-wise scaling laws under declared assumptions. They do not include every activation, allocator reservation or batch-scheduling effect. The lab uses one-layer, one-head logical score-matrix accounting separately from its all-layer persistent state bars to prevent adding unrelated quantities as if they were a measured peak.',
      ]),
    ],
  },
}
