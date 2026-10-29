import { detail, introduction, type TopicPerspectives } from './perspective-types'

export const attentionPerspectives: Record<string, TopicPerspectives> = {
  'attention-overview': {
    introduction: introduction(
      'Self-attention lets a token position incorporate information from other positions it is allowed to read. It first works out how to weight those positions, then combines their value features. This is a way to make a context-dependent update, not simply to select the most important word.',
      'Consider three earlier positions that supply different feature vectors. The current position calculates a different weight for each and mixes those vectors. Another query can read the same stored vectors with a different mixture. The comparison features determine the weights; the value features determine what is mixed. This separates the jobs of queries, keys and values without treating them as literal natural-language questions or passages.',
      [
        'Create distinct query, key and value representations from the current hidden state.',
        'Compare each query with eligible keys, then exclude forbidden dependencies.',
        'Turn the remaining comparisons into relative weights.',
        'Combine the value features using those weights, join head results and project the update back into the stream.',
      ],
      'A large attention weight is only one coefficient in an internal mixing step. The value vectors, output projection, residual path and later layers all affect the result. A synthetic map teaches this mechanism; it is not evidence of learned attention for your text.',
    ),
    technical: [
      detail(
        'technical-full-attention',
        'Expand the two different contractions',
        [
          'A score contracts head features while preserving query and key positions. The weighted-value operation then contracts key positions while preserving query positions and output features. Track the query-to-K/V mapping explicitly for grouped attention.',
          'Read the indexes as a small program: h fixes the query head, t the current query position and u a working key slot; d is the feature being summed. The batch index is omitted here only to shorten notation. g(h) selects the K/V head, A is the probability tensor, and M is zero on allowed pairs and negative infinity elsewhere. V in this equation is the value tensor, not the vocabulary-size symbol.',
          'For a fixed finite probability row, each output coordinate is a convex combination of the corresponding visible V coordinates. This coordinate bound applies before Wo and residual addition. Changing the values can change the output without changing the probabilities, which is a useful controlled experiment in the scratch lab.',
        ],
        String.raw`A_{h,t,u}=\operatorname{softmax}_u\left(\frac{\sum_dQ_{h,t,d}K_{g(h),u,d}}{\sqrt{D_h}}+M_{t,u}\right),\qquad O_{h,t,d}=\sum_u A_{h,t,u}V_{g(h),u,d}`,
      ),
    ],
  },
  'attention-scores': {
    introduction: introduction(
      'An attention score is a comparison between the current query and one candidate key. It combines matching feature coordinates into a single value. The model uses the differences among eligible scores to decide how to distribute attention.',
      'Think of comparing candidates against several criteria, where each criterion can support or oppose a match. The comparison is not just counting shared words. Query and key features are learned numerical representations, and both direction and magnitude can affect the result.',
      [
        'Choose one query and one key in the correct head group.',
        'Combine the corresponding feature contributions into a raw comparison.',
        'Adjust the score scale for the head’s feature width.',
        'Pass all candidate comparisons through visibility rules and normalization.',
      ],
      'A raw score is neither a probability nor necessarily a cosine similarity. A very large score for a future position still receives no attention after causal exclusion. To interpret a column, check its absolute key position rather than assuming its compact slot number is a token position.',
    ),
    technical: [
      detail(
        'technical-score-variance',
        'State the assumptions behind square-root scaling',
        [
          'Under the motivating idealization of independent, zero-mean, unit-variance query and key coordinates, each coordinate product has unit variance and the sum has variance proportional to head width. Dividing by its square root compensates for that growth. Learned activations need not satisfy those idealized assumptions.',
          'The variance argument has two steps: independence gives E[q_d k_d]=0 and E[q_d² k_d²]=1, so each product has variance one; uncorrelated products across d make their variances add. This is why the denominator is a square root rather than the full width. If learned coordinates are correlated or differently scaled, that motivating calculation need not describe their measured variance.',
          'For a controlled intervention, scaling the entire query by a positive constant scales all its finite scores. Changing only one query feature instead adds a key-dependent change, so different candidates can move in different directions. Neither intervention changes the causal mask.',
        ],
        String.raw`s_u=\frac{q^\top k_u}{\sqrt{D_h}},\qquad \frac{\partial s_u}{\partial q_d}=\frac{k_{u,d}}{\sqrt{D_h}}`,
      ),
    ],
  },
  'causal-mask': {
    introduction: introduction(
      'A causal mask prevents a token position from using information that would still be in its future. This makes the calculation suitable for predicting a next token from a known prefix. The rule concerns information access, not the visual order in which the app animates operations.',
      'Imagine reconstructing a message as it arrives, without peeking at later words. A decoder may process a known prompt in a batch, but every position still has its own permitted view. Batching the work does not grant earlier positions permission to inspect later inputs.',
      [
        'Identify the absolute position of the current query.',
        'Determine which key positions the attention policy permits it to read.',
        'Exclude disallowed scores before producing attention weights.',
        'Normalize over the remaining keys and combine only their value contributions.',
      ],
      'The current input position is allowed because its ID is already known; its hidden state predicts the following ID. A window or sparse rule can exclude additional old keys, but it must never permit future information. Compact storage slots do not replace this absolute-position test.',
    ),
    technical: [
      detail(
        'technical-mask-normalization',
        'Pre-normalization exclusion has a precise effect',
        [
          'Adding negative infinity to a forbidden score makes its exponential zero. Merely setting that score to zero would leave a positive exponential contribution. Masking therefore changes both the excluded entry and the denominator shared by every allowed entry.',
          'In exact arithmetic, removing forbidden probabilities afterwards and renormalizing can recover the same allowed distribution. In floating point this is not a reliable implementation substitute: an overwhelming forbidden score can make the allowed probabilities underflow before renormalization. The reference masks scores first.',
        ],
        String.raw`M_{q,k}=\begin{cases}0&k\le q\text{ and policy permits }k\\-\infty&\text{otherwise}\end{cases}`,
      ),
    ],
  },
  softmax: {
    introduction: introduction(
      'Softmax turns a collection of relative preferences into a distribution of weights. Candidates with larger scores receive a larger share, while all shares together account for the available weight. It matters which other candidates are present because they compete in the same normalization.',
      'Picture dividing a fixed amount of emphasis among several alternatives. Raising one alternative’s preference changes not just its own share but the shares of its competitors. This picture describes relative weighting; it does not imply calibrated confidence, understanding or truth.',
      [
        'Take the permitted scores from a single row.',
        'Re-express them relative to a common reference to keep the calculation stable.',
        'Convert relative preferences into positive contributions.',
        'Normalize those contributions into a distribution over the correct axis.',
      ],
      'Attention softmax weights key positions, while vocabulary softmax weights output IDs. A probability in an attention map is not a probability that the model will print that token. Masked zeros also need to be distinguished from tiny allowed values that only look like zero when rounded.',
    ),
    technical: [
      detail(
        'technical-softmax-sensitivity',
        'Derive how changing one score changes the row',
        [
          'Differentiating the normalized exponentials reveals coupling across the whole row. Increasing one score directly raises its own probability and lowers the other probabilities, subject to floating-point precision and any fixed mask. The derivatives sum to zero because total probability is conserved.',
          'The Kronecker delta delta_ij is one for i=j and zero otherwise. The diagonal derivative is p_i(1−p_i); for i≠j it is −p_i p_j. The odds-ratio identity applies to two allowed finite-score candidates, so both probabilities are positive in exact arithmetic. It is not a ratio to form using a masked zero or an underflowed stored value.',
          'Subtracting a common maximum uses shift invariance and does not change the exact distribution. Scaling is different: it changes pairwise gaps. The stable form is essential when inspecting large logits rather than directly exponentiating their absolute magnitudes.',
        ],
        String.raw`\frac{\partial p_i}{\partial s_j}=p_i(\delta_{ij}-p_j),\qquad \frac{p_i}{p_j}=e^{s_i-s_j}`,
      ),
    ],
  },
  rope: {
    introduction: introduction(
      'RoPE adds positional structure to attention by rotating pairs of query and key features. A position changes the amount of rotation. When the rotated queries and keys are compared, their relative positions affect the comparison.',
      'Picture two arrows whose orientations are adjusted according to where they occur in a sequence. Their resulting alignment depends partly on how far apart their positions are. This is a geometric way to follow the calculation; the model’s feature pairs are not literal arrows representing whole words or distances in a sentence.',
      [
        'Select an adjacent pair of query or key features.',
        'Choose the rotation from its absolute position and the pair’s frequency.',
        'Rotate both coordinates together while preserving the pair’s length in exact arithmetic.',
        'Compare rotated queries and keys during attention; this profile leaves values unrotated.',
      ],
      'A cache slot can move or wrap while the token’s absolute position stays the same. Resetting rotations when slots wrap would change the model. RoPE also does not guarantee that a trained model will remain accurate at arbitrary context lengths.',
    ),
    technical: [
      detail(
        'technical-relative-rotation',
        'Derive the relative-position identity',
        [
          'A two-dimensional rotation is orthogonal: its transpose is its inverse. Composing the inverse query rotation with the key rotation leaves a rotation by the positional difference, for the same feature-pair frequency. Summing pair contributions gives the complete rotated dot product.',
          'R(p) is shorthand for rotation by p times the frequency of this pair, not a claim that every pair rotates by p radians. For pair index j and head width Dh, that frequency is 10000^(−2j/Dh) radians per position in this profile. Positions p and r are integer sequence indexes; the angle and pair coefficients are numerical quantities, unrelated to playback time.',
          'This reference pairs adjacent coordinates and uses a fixed base. Other implementations may permute pair coordinates or scale frequencies. Matching the name RoPE alone is insufficient for a bitwise or architecture-faithful comparison; verify pair ordering, position IDs, frequency rules and numerical arithmetic.',
        ],
        String.raw`(R(p)a)^\top R(r)b=a^\top R(p)^\top R(r)b=a^\top R(r-p)b,\qquad R(p)^\top R(p)=I`,
      ),
    ],
  },
  'head-sharing': {
    introduction: introduction(
      'Attention can compare the same context in several independent query-head paths. Head-sharing schemes determine how many distinct key and value representations those paths read. Sharing keys and values reduces duplicated state without forcing the queries to ask the same numerical question.',
      'Imagine several readers consulting a shared set of reference material while each has a different task. Sharing the material does not force their choices of emphasis to match. The model’s queries are numerical projections rather than human readers, but the distinction between shared inputs and independent comparisons is useful.',
      [
        'Identify the current query head.',
        'Follow its assignment to the appropriate key/value head.',
        'Compute its own score and probability row using that shared state.',
        'Keep its result separate until head features are joined and projected.',
      ],
      'Changing the teaching configuration describes a different head-sharing architecture. It does not convert a trained checkpoint for free or guarantee the same quality. A logical K/V head slice is also not a separate allocation for every query that reads it.',
    ),
    technical: [
      detail(
        'technical-groups',
        'Projection widths and cache widths change together',
        [
          'With contiguous equal groups, dividing the query-head index by group size identifies the K/V head. Queries remain independently projected; K/V projection widths use the number of distinct K/V heads instead of the number of queries.',
          'A projected Q row has Hq·Dh coordinates. Splitting that last axis creates [Hq,Dh]; reordering batch/token/head axes is a view/indexing step rather than another learned projection. After each query head produces Dh output coordinates, concatenation reverses that split before Wo mixes the joined features. Head sharing changes the K/V projections, not this count of query outputs.',
          'Holding head width, retained length and layer count fixed, persistent payload scales with K/V heads. Holding residual width fixed while changing the number of query heads can also change head width, so identify which dimensions are controlled in a comparison. Changing actual learned architecture requires compatible weights or an explicitly specified conversion/training procedure.',
        ],
        String.raw`g(h)=\left\lfloor\frac{h}{H_q/H_{kv}}\right\rfloor,\quad W_Q:[D,H_qD_h],\quad W_K,W_V:[D,H_{kv}D_h]`,
      ),
    ],
  },
  'attention-interpretation': {
    introduction: introduction(
      'An attention heatmap shows how one internal mixing operation distributes weight across visible positions. It can help you identify visibility, concentration and differences among heads. It is not a complete explanation of why a model produced its final answer.',
      'A list of which sources someone consulted would not by itself tell you how each source changed their final conclusion. The contents, how they were combined and what happened afterwards also matter. Similarly, attention weights need to be read together with value vectors and the rest of the computation.',
      [
        'Check whether the map is synthetic, untrained numerical data or trained-checkpoint data.',
        'Identify the layer, query head, decoder call and absolute position axes.',
        'Separate excluded cells from low-probability allowed cells.',
        'Follow weighted values, the output projection and the residual path before interpreting downstream effects.',
      ],
      'Two maps can have identical weights and different value contributions. Conversely, a change in the generated continuation may arise from many changed intermediates. A visualization supports investigation; it does not automatically isolate a causal mechanism.',
    ),
    technical: [
      detail('technical-interventions', 'Separate coefficients from interventions', [
        'For a synthetic scalar value coordinate, weights [0.9,0.1] and values [1,100] give contributions [0.9,10] and output 10.9. The smaller attention coefficient produces the larger coordinate contribution here. Even that contribution is measured before Wo, residual addition and later nonlinearities; it is not a final-token attribution.',
        'The value-weighted result depends jointly on the probability row and the values. The final residual additionally depends on the output matrix and bypass operand. Inspecting only the probability coefficients omits those dependencies.',
        'A controlled intervention must identify which tensor was changed, which quantities were held fixed and where the new result was measured. Changing the prompt recomputes many paths at once, while changing a scratch value with a pinned probability row isolates a much smaller arithmetic question. The latter teaches a mechanism but is not a trained-model causal intervention.',
      ]),
    ],
  },
  'kv-cache': {
    introduction: introduction(
      'A KV cache keeps earlier attention keys and values so the next decoder call can use its prefix without computing all those representations again. Every layer has its own state, and later queries read the appropriate K/V head and retained positions.',
      'Think of keeping reusable intermediate work from a previous stage of a calculation. You preserve the particular results future steps need, not every earlier scratch result. This is a lifetime analogy: the cache itself is explicitly indexed K/V state, not a hierarchy of nested tables or permanent learned knowledge.',
      [
        'Process known input positions through each layer to produce their keys and values.',
        'Retain the positions permitted by the policy, together with their absolute-position identities.',
        'Combine the retained prefix with new keys and values for the next call.',
        'Use the new queries to read that working state, then commit the next retained state.',
      ],
      'Selected output does not become cached input until another forward call consumes it. Editing an earlier token can invalidate later cached representations. Other attention architectures may store a different form of state, so this dense K/V picture must not be imposed on every model.',
    ),
    technical: [
      detail('technical-causal-reuse', 'Why cached prefix values remain valid', [
        'A useful implementation check is cached versus full-prefix evaluation of the same causal model and position policy: the new position should agree within the declared numerical tolerance. Compare the same logical model, not a full-context recomputation against a windowed cached run. Appending and editing are different operations because only appending preserves every earlier dependency.',
        'At a fixed earlier position, causal dependencies exclude newly appended future inputs. By induction over decoder layers, appending a token leaves the earlier hidden states and their layer-specific projected K/V unchanged, provided the weights, position rules, visibility policy and original prefix are unchanged.',
        'A changed earlier token breaks this reuse argument for positions that depended on it. A different positional convention or policy can also break compatibility. The full logical persistent state has layer, batch, KV-head, retained-position and feature axes plus a position map; working attention may temporarily need more positions than the final resident cache.',
      ]),
    ],
  },
  'prefill-decode': {
    introduction: introduction(
      'Generation alternates between consuming known input and choosing what comes next. Prefill consumes the prompt with causal visibility. Ordinary decode then consumes a newly selected token and predicts the following one, using the retained prefix state.',
      'Imagine completing a message one piece at a time: you read what is already written, choose the next piece, then include that piece when deciding what follows. Selecting a piece and processing it are separate events. This distinction explains why confirmed output and cached positions can advance at different moments.',
      [
        'Process the prompt and keep reusable attention state.',
        'Use the last current position’s output scores to select a new ID.',
        'Consume that selected ID in the next forward call.',
        'Repeat selection and consumption until a stopping condition or budget ends the run.',
      ],
      'The architecture explorer’s prefill/decode switch changes reference shapes; it does not execute a model. The numerical workbenches record actual calls. Verification can process several trial positions at once and needs its own state-commit rules.',
    ),
    technical: [
      detail(
        'technical-call-boundaries',
        'Track consumed positions separately from predicted targets',
        [
          'An input at absolute position q produces a hidden representation whose vocabulary row predicts the next target position. Prefill can compute the prompt’s causal positions as a batch, but ordinary generation selects from the final current row. The selected next ID is initially an output, not a processed hidden state.',
          'For a fresh full-prefix prefill, the retained-prefix count is zero and all prompt positions are working keys. During ordinary decode, current-query count is one while working keys include the retained prefix and that input position. Retention, selection and commit are separate events in the trace; the last budgeted output can remain uncached.',
        ],
      ),
    ],
  },
  'cache-retention': {
    introduction: introduction(
      'A retention policy decides which earlier keys and values remain available for future calls. Full retention keeps the processed prefix. A sliding window keeps recent positions. The illustrative sparse policy keeps recent positions plus fixed anchors.',
      'Picture a moving spotlight over recent events, with selected older landmarks kept visible as well. The spotlight is an intuition for eligibility and retention, not physical memory layout. A retained entry still carries its original absolute position even when its compact slot changes.',
      [
        'Distinguish how much sequence has been processed from how much state remains.',
        'For each query, apply its own causal and policy-specific visibility rule.',
        'Keep all working keys needed by queries in the current batch.',
        'After those queries finish, commit the state that the next call should inherit.',
      ],
      'Removing old state changes what future queries can directly read. Sparse does not automatically mean bounded memory: this policy’s anchors continue accumulating. Final retained state can also be smaller than the temporary working set needed during a batched call.',
    ),
    technical: [
      detail(
        'technical-retention-sets',
        'Define positions as sets before assigning slots',
        [
          'For a sliding window, keep the final recent interval. For the sparse teaching rule, take its union with eligible fixed-stride anchors and remove duplicates. A slot index is then an addressing choice over that set, not a replacement position ID.',
          'Working state contains the previous retained prefix and all current positions until each query has read its eligible keys. In verification, the pre-round checkpoint must additionally remain recoverable until acceptance and rollback are decided. Trimming to the final resident set too early is incorrect for earlier queries.',
        ],
        String.raw`\mathcal R_{\rm sliding}=\{k:\max(0,S-W)\le k<S\},\quad \mathcal R_{\rm sparse}=\mathcal R_{\rm sliding}\cup\{k:0\le k<S,\ k\bmod s=0\}`,
      ),
    ],
  },
  'cache-bytes': {
    introduction: introduction(
      'Logical cache memory counts the key and value numbers the model retains across layers, sequences, heads, positions and features. It helps you understand what makes persistent attention state grow and why head sharing or retention changes the estimate.',
      'Before measuring how much room a set of stored measurements takes, first count what is actually being kept and how large each stored measurement is. That is the idea here. It counts logical values rather than assuming a particular arrangement of physical memory pages.',
      [
        'Identify the number of layers and distinct K/V heads.',
        'Count retained positions under the current policy.',
        'Include the feature width and both key and value state.',
        'Apply the declared scalar storage size, then keep excluded overhead and trace history separate.',
      ],
      'A logical payload estimate is not total process memory or a throughput measurement. Selecting a smaller scalar size in the teaching configuration does not execute quantization. Parameters, working activations, metadata, allocator overhead and historical trace snapshots are separate costs.',
    ),
    technical: [
      detail(
        'technical-layerwise-bytes',
        'Use the general layerwise sum when states are uneven',
        [
          'The compact formula assumes equal retained length, head width, K/V width and dtype across layers. In the layerwise sum below, R_l is layer l’s retained count, Hkv,l its K/V-head count, Dh,l the common key/value width in that layer and s_bytes,l its bytes per scalar. This handles layers at different playback stages without assuming they have all advanced together. If keys and values have different widths or dtypes, count their payloads separately instead of using the factor of two.',
          'The scalar-byte term is an accounting assumption. Quantized encodings may need per-group scales or other metadata, and paged allocators add block-management effects. Keep logical scalar payload separate from both physical allocation and the inspector’s full historical capture.',
        ],
        String.raw`\mathrm{bytes}=2B\sum_{\ell=0}^{L-1}H_{kv,\ell}R_\ell D_{h,\ell}s_{\mathrm{bytes},\ell}`,
      ),
    ],
  },
  'storage-kernels': {
    introduction: introduction(
      'The model’s attention rule, the algorithm that executes it efficiently and the way its state is allocated are separate choices. Understanding which level you are looking at prevents performance terminology from being mistaken for a different mathematical model.',
      'A recipe, a cooking procedure and the kitchen’s storage organization are related but different. An optimized procedure can follow the same recipe while avoiding unnecessary intermediate work. Similarly, a kernel can calculate attention without storing every conceptual matrix shown in the diagram.',
      [
        'Identify the architecture’s allowed dependencies and head sharing.',
        'Identify how the execution kernel tiles or fuses the calculation.',
        'Identify how logical K/V positions map into physical storage.',
        'Measure speed and memory in the actual runtime rather than inferring them from the diagram alone.',
      ],
      'FlashAttention describes an IO-aware attention execution strategy; PagedAttention concerns physical K/V management. Neither name alone specifies the model’s head sharing or context policy. This local reference microscope deliberately retains inspectable intermediates and is not a production-kernel benchmark.',
    ),
    technical: [
      detail(
        'technical-online-softmax',
        'How a kernel can avoid a full stored probability matrix',
        [
          'For one query row, online softmax can maintain a running maximum, normalizer and unnormalized value accumulator across key blocks. When a larger maximum arrives, rescale the previous accumulator and denominator into the new reference scale before adding the block’s contributions.',
          'Here m is the running maximum score, l a scalar sum of exponentials relative to m, and o a value-width vector containing the corresponding unnormalized weighted sum. Each displayed sum runs over eligible keys j in the new block. The final attention output is o/l; the symbol o in the recurrence is not already normalized.',
          'Synthetic scalar check: process score 0 with value 2, then score ln(3) with value 4. After the first block (m,l,o)=(0,1,2). The second block changes the reference scale, giving l=1/3+1 and o=2/3+4. Their ratio is 3.5, the same as weighting [2,4] by [1/4,3/4] without retaining a full probability row.',
          'Initialize from the first block containing an eligible key, skip fully masked blocks, and divide the final value accumulator by the final normalizer. A row with no eligible keys needs an explicit policy; subtracting negative infinity from itself is not a valid initialization.',
          'The recurrence is mathematically equivalent to normalizing the complete eligible row in exact arithmetic. Kernel tiling, accumulation order and precision can still change floating-point results. Paging separately changes how K/V blocks are addressed, not this normalization identity.',
        ],
        String.raw`m'=\max(m,\max s_{\rm block}),\quad l'=e^{m-m'}l+\sum_j e^{s_j-m'},\quad o'=e^{m-m'}o+\sum_j e^{s_j-m'}v_j`,
      ),
    ],
  },
}
