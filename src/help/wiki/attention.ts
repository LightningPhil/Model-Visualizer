import type { WikiArticle } from './types'
import { section } from './foundations'

export const attentionArticles: WikiArticle[] = [
  {
    id: 'attention-overview',
    title: 'What self-attention computes',
    category: 'Attention',
    summary:
      'Queries score visible keys, probabilities weight values, and head results are projected into a residual update.',
    aliases: [
      'self-attention',
      'QKV',
      'query',
      'key',
      'value',
      'why attention',
      'context',
      'heads',
    ],
    kinds: ['attn', 'q', 'k', 'v', 'read', 'attention-expansion'],
    graphs: ['attention', 'attention-heads'],
    prerequisites: ['linear-maps', 'residual'],
    related: [
      'attention-scores',
      'causal-mask',
      'softmax',
      'head-sharing',
      'attention-interpretation',
    ],
    sources: ['attention', 'gqa'],
    sections: [
      section('qkv', 'Three projections have different jobs', [
        'A query is the vector used by the current position to score candidate positions. A key is a vector compared with queries. A value is the vector whose coordinates are mixed into the result. These names describe mathematical roles; they are not literal questions, dictionary addresses or human-readable facts.',
        'The normalized hidden state projects to Q, K and V with different learned matrices. RoPE rotates Q and K. K/V from the retained prefix join the new K/V for this call. For each query head, dot products with its assigned K head produce scores; masking removes forbidden dependencies; softmax yields a distribution over allowed key positions.',
      ]),
      section(
        'weighted',
        'A position-dependent weighted sum',
        [
          'Each query uses its probability row to average the corresponding value vectors. The output still has a feature axis: attention does not return just “the most relevant token.” A row may spread weight over many positions. Concatenating head outputs and applying Wo gives a D-feature update for the residual stream.',
          'For a synthetic two-value example, probabilities [0.25,0.75] and value vectors [2,0] and [0,4] give [0.5,3]. Changing the value vectors changes the result even if the attention probabilities stay identical. The output projection and residual addition then change what reaches later layers.',
        ],
        String.raw`O_{b,h,t,d}=\sum_{u=0}^{U-1}A_{b,h,t,u}\,V_{b,g(h),u,d}`,
      ),
      section('boundaries', 'What mixes, and what does not?', [
        'The weighting contracts the key-position axis U. Head concatenation preserves head order and joins feature coordinates; it is not an average of heads. Wo then mixes those concatenated features. In GQA, sharing K/V reduces the distinct state, but independent Q heads can still produce different probability rows.',
        'The architecture graph shows a logical factorization. A fused attention kernel can compute the same result using tiled intermediates instead of storing every displayed tensor. Use the reference microscope to learn the calculation, and keep kernel/storage claims separate.',
      ]),
    ],
  },
  {
    id: 'attention-scores',
    title: 'Dot-product scores and √Dh scaling',
    category: 'Attention',
    summary:
      'A score sums query–key feature products; scaling controls its magnitude before softmax.',
    aliases: ['QK', 'Q · Kᵀ', 'similarity', 'sqrt', 'scale', 'contraction', 'score'],
    kinds: ['scores', 'scale'],
    graphs: [],
    prerequisites: ['linear-maps'],
    related: ['causal-mask', 'softmax', 'rope', 'attention-overview'],
    sources: ['attention'],
    sections: [
      section(
        'dot',
        'One query meets one key',
        [
          'For query head h, query position t and working key slot u, multiply corresponding Dh feature coordinates and sum. The query head reads a specific K/V head g(h). This gives one scalar per query/key pair, so all scores have logical shape [B,Hq,T,U]. A large positive raw dot product favors that key relative to other keys in the same row, after masking and scaling.',
          'A dot product depends on vector lengths as well as directions. Calling it “similarity” is useful shorthand, but it is not necessarily cosine similarity: this profile does not divide Q and K by their norms before the score.',
        ],
        String.raw`s_{b,h,t,u}=\frac{1}{\sqrt{D_h}}\sum_{d=0}^{D_h-1}Q_{b,h,t,d}K_{b,g(h),u,d}`,
      ),
      section('scaling', 'Why divide by the square root?', [
        'The Transformer paper motivates scaling with an idealized model: if query and key coordinates are independent, zero mean and unit variance, summing Dh coordinate products has variance Dh. Dividing by √Dh brings that idealized variance back to 1, helping avoid excessively sharp softmax inputs as head width grows.',
        'Those independence and variance assumptions are motivation, not a promise about every learned activation. Scaling does not make the vectors unit length or force all score rows to have identical statistics.',
      ]),
      section('example', 'A two-feature example · synthetic', [
        'For q = [1,2] and k = [3,4], the raw score is 1×3 + 2×4 = 11. With Dh = 2, the scaled score is 11/√2 ≈ 7.778. Another key [−1,1] has raw score 1 and scaled score approximately 0.707. The difference between allowed scores determines their softmax probability ratio.',
        'The key axis lists working positions, which can be compacted after eviction. Inspect the absolute position map before interpreting a column as an earlier or later token. A causal mask can override a large future-key score completely.',
      ]),
    ],
    experiment: {
      title: 'Reconstruct a score',
      steps: [
        'Open “How is one attention score made?” in the tensor laboratory.',
        'Select one score cell and read its query head and absolute positions.',
        'Follow the query and key operands and compare the sum with the scaled-score operation.',
      ],
      observation:
        'The scalar contracts features, not positions. Its query and key come from the explicit head mapping.',
    },
  },
  {
    id: 'causal-mask',
    title: 'Causal masking and visible positions',
    category: 'Attention',
    summary: 'A query can only use allowed positions; −∞ excludes a score before softmax.',
    aliases: [
      'future',
      'invisible',
      'mask',
      'autoregressive',
      'triangle',
      'negative infinity',
      'visibility',
      'padding',
    ],
    kinds: ['scale'],
    graphs: [],
    prerequisites: ['attention-scores'],
    related: ['softmax', 'prefill-decode', 'cache-retention', 'speculative-decoding'],
    sources: ['attention', 'mistral'],
    sections: [
      section('causality', 'Visibility is a dependency rule', [
        'A causal query at absolute position q must not depend on a key at a future position k > q. Full causal attention allows 0 ≤ k ≤ q, including the current input position. This is compatible with next-token prediction: the hidden state at position q predicts the following token, not the input ID already present at q.',
        'For a fresh prefill with consecutive positions, this produces a lower triangle including its diagonal. With a retained prefix, a sliding window or sparse anchors, a simple triangle drawn over compact slot numbers is insufficient. The mask compares absolute query/key positions and the selected retention rule.',
      ]),
      section('minus-infinity', 'Mask before normalization', [
        'Set disallowed scores to negative infinity before softmax. Their exponentials are zero; the probability mass normalizes over allowed positions. Multiplying scores by zero would be wrong: a zero score still contributes exp(0) = 1 and can receive substantial probability.',
        'A valid row in this supported profile includes its own current position, so it has at least one allowed key. In a generalized implementation, an all-masked row needs an explicit convention; subtracting the maximum of an all-negative-infinity row produces undefined arithmetic. Do not infer such a convention from a heatmap.',
      ]),
      section('example', 'First-query example · synthetic', [
        'The mask is not fundamentally a triangle in screen coordinates. For query absolute position 10 and compact key positions [0,9,10], full causal attention allows all three columns; a two-position local window allows only 9 and 10. A query at position 9 excludes the final column because that key is in its future. Read position labels before inferring visibility from the picture.',
        'Suppose prefill positions are [0,1,2] and query 0 has raw scores [2,9,7]. Its causal masked row is [2,−∞,−∞], so probabilities are [1,0,0], regardless of the large forbidden scores. Query 2 may attend to all three positions under full retention.',
        'Sliding attention adds a lower bound: this app’s window W includes the current position, giving max(0,q−W+1) ≤ k ≤ q. Sparse teaching retention permits causal stride anchors or recent positions. Neither policy permits future information.',
      ]),
    ],
  },
  {
    id: 'softmax',
    title: 'Softmax and stable probability calculations',
    category: 'Attention',
    summary:
      'Convert one row of relative scores into a distribution and understand why masked cells become zero.',
    aliases: [
      'probability',
      'exponential',
      'row normalization',
      'denominator',
      'exp',
      'stable',
      'overflow',
    ],
    kinds: ['softmax', 'sample'],
    graphs: [],
    prerequisites: ['attention-scores'],
    related: ['causal-mask', 'logits', 'sampling', 'attention-interpretation'],
    sources: ['attention'],
    sections: [
      section(
        'formula',
        'Normalize along the key axis',
        [
          'For each query row, exponentiate its allowed scores and divide by the sum of their exponentials. Every output is nonnegative and the row sums to one in exact arithmetic. Here softmax operates independently for each batch item, query head and query position; it does not normalize all heads or all query rows together.',
          'Subtracting a common row maximum leaves the distribution unchanged because the common exponential factor cancels. It keeps finite exponentials at most 1 and avoids overflow for large positive scores. The reference implementation uses this stable form and records the maximum and denominator for cell explanations.',
        ],
        String.raw`p_i=\frac{e^{s_i-m}}{\sum_j e^{s_j-m}},\qquad m=\max_j s_j`,
      ),
      section('example', 'Relative scores matter · synthetic', [
        'Scores [1,2] become approximately [0.2689,0.7311]. Scores [101,102] give the same probabilities. The ratio between the second and first probabilities is exp(2−1) ≈ 2.718. Adding a constant changes neither ratio nor distribution; multiplying scores changes the gaps and usually changes the distribution.',
        'A masked −∞ contributes zero when there is a finite allowed maximum. In Float32, finite probabilities may round or underflow and their stored sum can differ slightly from 1. A displayed 0.0000 may be a rounded tiny value rather than a masked entry; inspect the visibility marker and precise cell value.',
      ]),
      section('two-softmaxes', 'Attention probability is not next-token probability', [
        'Attention softmax distributes weight over visible key positions. Vocabulary softmax distributes weight over candidate output IDs. Their formulas are related, but their axes and interpretations differ. A 0.8 attention weight does not mean an 80% probability of generating that attended token.',
        'Attention probabilities weight value vectors before head concatenation, output projection and more layers. They describe an internal mixing step; they are not calibrated confidence scores for the whole model’s answer.',
      ]),
    ],
  },
  {
    id: 'rope',
    title: 'RoPE: position as a rotation',
    category: 'Attention',
    summary:
      'Rotate query/key feature pairs using absolute positions while introducing relative-position structure into dot products.',
    aliases: [
      'rotary',
      'position encoding',
      'RoFormer',
      'sine',
      'cosine',
      'rotation',
      'theta',
      'absolute position',
    ],
    kinds: ['ropeq', 'ropek', 'newk'],
    graphs: [],
    prerequisites: ['tensors', 'attention-scores'],
    related: ['kv-cache', 'cache-retention', 'linear-maps', 'position-methods'],
    sources: ['rope'],
    sections: [
      section(
        'pair',
        'Rotate two features together',
        [
          'For each adjacent pair (a,b), compute angle θ = p·10000^(−2j/Dh), where p is the absolute position and j is the zero-based pair index. The numerical profile applies the two-dimensional rotation below to Q and K, before attention. V is not rotated in this profile.',
          'The pair index controls the frequency, so different pairs rotate at different rates. Dh is even because every feature belongs to a pair. Other implementations may use different feature-pair permutations, frequency bases or scaling; compare the exact convention rather than relying on the name RoPE alone.',
        ],
        String.raw`\begin{pmatrix}a'\\b'\end{pmatrix}=\begin{pmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{pmatrix}\begin{pmatrix}a\\b\end{pmatrix}`,
      ),
      section('relative', 'Absolute rotations yield relative-position dependence', [
        'For one frequency and feature-pair vectors a and b, the dot product of R(p)a and R(r)b equals aᵀR(r−p)b, because R(p)ᵀR(r) = R(r−p). Thus the comparison carries a relative displacement even though each vector is rotated using its own absolute position. Rotation preserves pair length in exact arithmetic, with small finite-precision differences in a reference run.',
        'Synthetic example: rotating [1,0] by π/2 gives [0,1]. Rotating [0,1] by the same angle gives [−1,0]. This is a geometric illustration; actual angles use the position/frequency formula, not an arbitrary quarter-turn for every token.',
      ]),
      section('cache', 'A cache slot is not a position ID', [
        'The numerical cache stores the already rotated keys alongside unrotated values. A ring buffer may reuse a physical slot, but the new key’s rotation must use its absolute sequence position. Resetting position IDs when slots wrap would change the model’s dot products.',
        'A larger context-capacity setting does not prove reliable long-context extrapolation. RoPE is a positional mechanism; a trained model’s behavior outside its training regime also depends on its training and any explicit frequency-scaling design.',
      ]),
    ],
    experiment: {
      title: 'Change the position, keep the pair in view',
      steps: [
        'Open the laboratory’s RoPE question and select a rotated coordinate.',
        'Inspect its original pair, sine, cosine and absolute position.',
        'Choose another position and another pair index.',
      ],
      observation:
        'Position changes the angle; pair index changes the frequency. Follow both operands because each rotated output depends on two original features.',
    },
  },
  {
    id: 'head-sharing',
    title: 'MHA, GQA and MQA',
    category: 'Attention',
    summary:
      'Independent query heads can share K/V state without sharing their attention distributions.',
    aliases: [
      'multi-head',
      'grouped-query',
      'multi-query',
      'Hq',
      'Hkv',
      'KV groups',
      'query heads',
      'KV heads',
      'split heads',
    ],
    kinds: ['attn', 'attention-head', 'concat', 'kv-head', 'attention-expansion'],
    graphs: ['attention-heads', 'whole-model-heads'],
    prerequisites: ['attention-overview'],
    related: ['kv-cache', 'cache-bytes', 'tensors', 'attention-comparison'],
    sources: ['gqa', 'mqa', 'attention'],
    sections: [
      section('architecture', 'What is shared?', [
        'Multi-head attention (MHA) gives each query head a distinct K/V head pair: Hkv = Hq. Multi-query attention (MQA) uses one shared K/V pair: Hkv = 1. Grouped-query attention (GQA) uses an intermediate number of pairs, with groups of query heads sharing each pair.',
        'This teaching profile partitions contiguous query heads into equal groups. Query head h reads KV head floor(h/(Hq/Hkv)). Hq must divide the fixed residual width D, Dh = D/Hq must be even, and Hkv must divide Hq. The GQA control offers intermediate divisors; the MHA and MQA endpoints have their own controls.',
      ]),
      section('example', 'Eight queries, two K/V pairs · synthetic configuration', [
        'For Hq = 8 and Hkv = 2, Q0–Q3 read K0/V0, and Q4–Q7 read K1/V1. Q0 and Q1 have different query vectors, so comparing them with the same K0 can produce different score rows and attention probabilities. They then use those different rows to weight the same V0.',
        'Each query still produces Dh features. Concatenating all eight outputs gives Hq×Dh features before Wo. Sharing K/V does not mean averaging query heads, cloning the cache for each query, or forcing identical outputs.',
      ]),
      section('tradeoff', 'State savings are architectural, not free weight conversion', [
        'Holding layer count, sequence retention and head width fixed, reducing Hkv reduces logical K/V payload proportionally. It also changes the K/V projection widths. The architecture controls describe a different teaching configuration; they do not convert a trained checkpoint’s matrices or guarantee unchanged model quality.',
        'Use the split-head view to see each independent query path, and the head observatory to compare actual probability rows. A logical slice is a view of shared state, not evidence of a new physical allocation. Kernel efficiency and trained quality require separate measurement.',
      ]),
    ],
    experiment: {
      title: 'Separate sharing from behavior',
      steps: [
        'In Attention & heads, choose GQA and inspect the Q-to-K/V mapping.',
        'Open Split attention heads and follow two queries in the same group.',
        'In the tensor laboratory, compare their rows in the attention head observatory.',
      ],
      observation:
        'The K/V head identity can match while query vectors and attention weights differ. The architecture heatmap is synthetic; the laboratory rows come from its untrained calculation.',
    },
  },
  {
    id: 'attention-interpretation',
    title: 'What an attention heatmap can tell you',
    category: 'Attention',
    summary:
      'Read positions, masks and probabilities without treating a bright cell as a complete explanation of a prediction.',
    aliases: [
      'interpretability',
      'explanation',
      'heatmap',
      'importance',
      'confidence',
      'causal influence',
      'bright cell',
    ],
    kinds: ['softmax', 'weighted'],
    graphs: [],
    prerequisites: ['attention-overview', 'softmax'],
    related: [
      'evidence',
      'tensor-laboratory',
      'linear-maps',
      'feature-representations',
      'causal-interventions',
    ],
    sources: ['attention', 'interpretability'],
    sections: [
      section('read', 'First establish what the axes and values are', [
        'An attention map normally puts queries on one axis and keys on the other. In this app the axes carry explicit positions and head context. The architecture lab uses a deterministic synthetic pattern; the numerical observatory uses recorded reference probabilities. Check that provenance before drawing conclusions.',
        'A masked cell is excluded, not just low-scoring. A compact retained-slot number may refer to a much later absolute position. Compare probability rows only after checking layer, query head, decoder call and visibility. Rows with different permitted keys normalize over different sets.',
      ]),
      section('limits', 'Probability weight is one part of a computation', [
        'An attention probability multiplies a value vector. Large probability on a small or canceling value contribution can have a different downstream effect from large probability on another vector. Wo mixes head features, residual addition adds the existing stream, and later layers transform the result again.',
        'The paper “Attention is not Explanation” supplies empirical reasons to question attention weights as standalone explanations. Its experiments do not establish that attention is never informative. Here, treat a map as evidence about a particular internal mixing operation, then inspect values and downstream calculations for a fuller account.',
      ]),
      section('investigate', 'A useful sequence of questions', [
        'Which key positions were eligible? How sharp was the distribution? What value features did those keys contribute? How did the projection and residual path change the selected coordinate? The microscope can answer these arithmetic questions for a captured trace.',
        'Changing a prompt or seed is an experiment, but many intermediate values can change at once. A difference between two runs does not isolate one causal mechanism by itself. Matching shape and position metadata are prerequisites for a meaningful cell comparison.',
      ]),
    ],
  },
  {
    id: 'kv-cache',
    title: 'KV cache: persistent attention state',
    category: 'State & execution',
    summary:
      'Store prior keys and values per layer and KV head so a later query can use its prefix without re-projecting it.',
    aliases: ['cache', 'persistent', 'keys', 'values', 'state', 'history', 'prefix', 'memory'],
    kinds: ['kv', 'cachek', 'cachev', 'keys', 'values', 'read', 'kv-head'],
    graphs: ['cache'],
    prerequisites: ['attention-overview', 'tensors'],
    related: [
      'prefill-decode',
      'cache-retention',
      'cache-bytes',
      'storage-kernels',
      'attention-comparison',
    ],
    sources: ['mqa', 'gqa', 'mistral'],
    sections: [
      section('state', 'Two tensors with explicit axes', [
        'For this dense attention family, persistent K and V each have logical axes [L,B,Hkv,R,Dh], plus a map from retained slots to absolute positions. L selects a decoder layer, B a batch item, Hkv a K/V head, R a retained position and Dh a feature. Separate layers and KV heads hold different values even when their shapes match.',
        'During a new call, each layer projects new K/V from that layer’s current input. New queries use both retained prefix state and current-call K/V. Only K and V are kept for future attention in this profile; the app does not cache all prior query vectors as ordinary decoding state.',
      ]),
      section('validity', 'Why can the old values be reused?', [
        'Under causal attention, appending future tokens cannot change earlier positions’ already-computed hidden states: their allowed inputs remain the same. Their layer-specific K/V can therefore be reused, provided the weights, positional rules, prior context and attention policy are unchanged. Editing an earlier prefix token invalidates downstream cached state; it is not just an in-place update to one ID.',
        'Caching saves repeated prefix computation, but a new query still compares with its eligible keys and mixes their values. Full attention’s decode work and logical state therefore grow with prefix length. Caching does not give the model infinite capacity or turn attention into constant work per step.',
      ]),
      section('lifecycle', 'Selected output is not necessarily cached input', [
        'The forward pass consumes input IDs and writes their K/V. Its output distribution selects the next ID. That chosen ID only obtains its own K/V when a subsequent forward pass consumes it. If generation stops immediately after selection, the final confirmed token can remain uncached.',
        'K/V state is not the model’s learned parameter memory, the experiment notebook, or a nested-table metaphor. Other attention families can need different state representations: MLA compresses state differently, and a linear-attention recurrence would require its own explicit semantics.',
      ]),
    ],
  },
  {
    id: 'prefill-decode',
    title: 'Prefill, decode and the generation loop',
    category: 'State & execution',
    summary:
      'Process a prompt in a causal batch, then repeatedly consume a new token and select the following one.',
    aliases: [
      'inference',
      'autoregressive',
      'T=1',
      'prompt processing',
      'next token',
      'feedback',
      'call',
      'generation',
    ],
    kinds: ['next', 'feedback', 'commit', 'decode', 'output'],
    graphs: ['generation'],
    prerequisites: ['reading-the-model', 'kv-cache'],
    related: ['causal-mask', 'logits', 'sampling', 'speculative-decoding'],
    sources: ['attention', 'mqa', 'stories'],
    sections: [
      section('prefill', 'Prefill processes multiple current positions', [
        'A fresh prompt of T tokens is processed with a causal mask. All its positions can be calculated as a batch because the mask prevents future dependencies. The final prompt position supplies the distribution used to select the first new output ID. The layer caches contain processed prompt positions according to the retention policy.',
        '“Parallel positions” does not mean that future prompt tokens leak into earlier predictions. The causal mask preserves the same dependency structure while allowing matrix operations across a batch of positions.',
      ]),
      section('decode', 'Ordinary decode has one query position', [
        'A later call consumes a selected ID, usually with T = 1. Its query attends over available prefix plus current K/V, then selects another output. Repeat until a stopping condition or output budget. The sequence extent increases, but residual width D and the number of layers stay fixed.',
        'Synthetic sequence example: prompt positions 0–2 are prefilled; the last prompt hidden state chooses ID a at position 3. The next call consumes a, writes its K/V at position 3 and chooses b at position 4. If the budget ends there, b is confirmed output while the cache only includes through a.',
      ]),
      section('views', 'Shape mode and runtime are different controls', [
        'In the architecture explorer, Prefill/Decode changes the described shapes. Decode uses a hypothetical next input position; it does not sample an actual ID or execute the imported graph. The numerical and trained workbenches instead record actual calls and selections in their trace.',
        'MTP verification is a distinct case with several current query positions in a trial batch. Do not assume that every post-prompt call has T = 1. Working state, final retention and confirmed output must each be tracked explicitly.',
      ]),
    ],
  },
  {
    id: 'cache-retention',
    title: 'Full, sliding and sparse retention',
    category: 'State & execution',
    summary:
      'Separate the processed extent, keys visible to each query, working K/V, and state retained for future calls.',
    aliases: [
      'window',
      'eviction',
      'sparse',
      'anchors',
      'stride',
      'absolute positions',
      'ring buffer',
      'P',
      'R',
      'U',
    ],
    kinds: ['cachek', 'cachev', 'keys', 'values', 'scale'],
    graphs: ['cache'],
    prerequisites: ['kv-cache', 'causal-mask'],
    related: ['cache-bytes', 'rope', 'speculative-decoding', 'attention-patterns'],
    sources: ['mistral', 'sparse'],
    sections: [
      section('policies', 'The exact policies in this workbench', [
        'Synthetic union check: at processed extent S=10, window W=3 and stride 4, recent positions are [7,8,9] and causal anchors are [0,4,8]. The retained union is [0,4,7,8,9], so R=5. Position 8 appears in both descriptions but occupies only one retained entry. This calculation depends on the stated fixed-anchor policy, not merely on the word sparse.',
        'Full prefix retains every processed position and permits every causal key. Sliding window W permits max(0,q−W+1) ≤ k ≤ q and retains the latest min(S,W) positions. W includes the current position in this app. Different implementations can count their window boundary differently, so use the inequality.',
        'The sparse teaching policy permits causal positions that are stride anchors (k mod stride = 0) or within the recent window. Retention stores the union without duplicates. Anchors keep accumulating as the processed extent grows; this is not a bounded-memory policy. It is an illustrative mask, not a reproduction of every sparse transformer design.',
      ]),
      section('working', 'Do not evict keys too early during prefill', [
        'A call reads the P-position retained prefix and adds T new positions, giving U = P + T working key slots. Every query applies visibility using its own absolute position. Only after the relevant calculations can retention reduce the persistent cache to its R-position result.',
        'Synthetic example: start from no prefix and prefill positions 0–5 with W = 3. Query 0 needs position 0 even though final retained state only contains positions 3–5. Trimming the working keys to the final window before calculating all queries would remove a required key.',
      ]),
      section('slots', 'Slots and position IDs must remain distinct', [
        'A ring-buffer illustration maps absolute position p to slot p mod W. The slot wraps; the absolute position does not. RoPE uses the absolute position, masks compare absolute positions, and the inspector labels them explicitly. The reference numerical cache uses compact tensors with position maps, not a claimed production ring-buffer allocator.',
        'Eviction deliberately removes the ability to attend directly to old keys outside the policy. Later hidden states can carry information from earlier interactions, but that is not equivalent to retaining the original K/V. Switching policies can change numerical results, not just memory consumption.',
      ]),
    ],
    experiment: {
      title: 'Watch absolute positions survive compaction',
      steps: [
        'Open KV cache and choose a sliding window smaller than the studied sequence.',
        'Inspect retained and evicted absolute positions and the optional ring-buffer illustration.',
        'Choose sparse retention and compare anchors with the recent window.',
      ],
      observation:
        'R can be much smaller than S. Sparse anchors and the recent window overlap and must be counted once; compact slots do not restart position IDs.',
    },
  },
  {
    id: 'cache-bytes',
    title: 'Calculate logical KV memory',
    category: 'State & execution',
    summary:
      'Count two sets of scalars across layers, batches, KV heads, retained positions and head features.',
    aliases: [
      'bytes',
      'memory cost',
      'capacity',
      'FP16',
      'BF16',
      'FP32',
      'quantization',
      'MiB',
      'GiB',
      'storage size',
    ],
    kinds: ['kv', 'keys', 'values'],
    graphs: [],
    prerequisites: ['kv-cache', 'head-sharing'],
    related: [
      'cache-retention',
      'storage-kernels',
      'evidence',
      'attention-comparison',
      'memory-budget',
    ],
    sources: ['gqa', 'mqa', 'paged'],
    sections: [
      section(
        'count',
        'Count the logical scalar payload',
        [
          'One K tensor contains L×B×Hkv×R×Dh scalars. V has the same logical size in this profile. Multiply by two and by bytes per stored scalar. For full-prefix retention R = S. If layer retention lengths differ during playback, calculate each layer’s K/V count and sum rather than assuming every layer has already advanced.',
          'Holding the other axes fixed, doubling layers, batch items, KV heads, retained positions or bytes per scalar doubles payload. Query heads do not directly index cached state; Hq affects cache size only through other architecture choices such as Hkv or Dh.',
        ],
        String.raw`\mathrm{KV\ payload}=2\,L\,B\,H_{kv}\,R\,D_h\,s\quad\text{bytes}`,
      ),
      section('example', 'Worked estimate · synthetic configuration', [
        'Let L = 6, B = 1, Hkv = 2, R = 128, Dh = 64 and s = 2 bytes. Payload is 2×6×1×2×128×64×2 = 393,216 bytes = 384 KiB. With eight KV heads at the same head width and retained length, it would be four times as large: 1.5 MiB.',
        'KiB uses 1,024 bytes; MiB uses 1,048,576. Decimal MB uses 1,000,000. The workbench formats binary units. A single-layer/single-head slice is not another full-cache allocation and should not be added again to the total.',
      ]),
      section('excluded', 'Payload is not total process memory', [
        'The estimate excludes model parameters, ordinary activations, attention scratch space, retained trace snapshots, position-map metadata, allocator overhead, padding, page fragmentation, sharding and quantization scales. Production memory can therefore differ substantially.',
        'The architecture scalar-storage setting changes an estimate. Selecting one byte does not implement quantization or change the laboratory’s Float32 arrays. Low-bit formats need actual encoding, scales and possibly other metadata. For measured memory or speed, specify the runtime and physical storage scheme separately.',
      ]),
    ],
  },
  {
    id: 'storage-kernels',
    title: 'Architecture, FlashAttention and PagedAttention',
    category: 'State & execution',
    summary:
      'Separate the attention equation from its execution kernel and the physical allocation of KV blocks.',
    aliases: [
      'GPU',
      'kernel',
      'tiling',
      'physical storage',
      'allocator',
      'FlashAttention',
      'PagedAttention',
      'vLLM',
      'pages',
      'performance',
    ],
    kinds: [],
    graphs: [],
    prerequisites: ['attention-overview', 'kv-cache'],
    related: [
      'cache-bytes',
      'head-sharing',
      'evidence',
      'attention-comparison',
      'prefix-sharing',
      'cache-quantization',
    ],
    sources: ['flash', 'paged', 'gqa'],
    sections: [
      section('layers', 'Three independent questions', [
        'Architecture: which queries can see which keys, how many query/KV heads exist, and what mathematical transformation is defined? GQA changes head sharing. Sliding attention changes visibility. MLA changes state representation. These choices belong to the model semantics.',
        'Kernel: how are those calculations scheduled, tiled, fused and rounded on hardware? Storage: how are the logical values allocated, addressed and shared? Changing one level does not automatically change the others. A model can combine GQA, a compatible optimized kernel and paged KV storage.',
      ]),
      section('flash', 'FlashAttention is an execution strategy', [
        'The original FlashAttention computes exact attention with IO-aware tiling and online normalization to reduce transfers between GPU memory levels. “Exact” describes the attention computation rather than an approximation that discards keys; finite-precision execution can still differ from another kernel’s rounding.',
        'Consequently, the score and probability matrices in a conceptual graph are logical objects. They need not both exist as full persistent arrays in a production kernel. The local numerical microscope deliberately materializes intermediates to make them inspectable; it is not a FlashAttention benchmark.',
      ]),
      section('pages', 'PagedAttention concerns physical KV management', [
        'PagedAttention organizes KV storage in blocks with logical-to-physical addressing, reducing waste and enabling sharing in serving systems. Logical token positions do not have to occupy one contiguous physical region. A page boundary is not an attention boundary or an extra feature axis.',
        'This workbench’s position maps and ring-buffer illustration help separate addressing from semantics, but they do not implement a production paged allocator, batching scheduler or GPU kernel. Claims about throughput, latency or fragmentation require measurements of such a system.',
      ]),
    ],
  },
]
