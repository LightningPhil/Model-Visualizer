export type ReasoningQuestion = { question: string; answer: string; investigate: string }

/** Companion explanations: ask a concrete question, then expose a mechanism
 * and an observable check. The article's sources establish the equations. */
export const reasoningQuestions: Record<string, ReasoningQuestion[]> = {
  adam: [
    {
      question: 'Does zero current gradient mean Adam stops moving the parameter?',
      answer:
        'Not necessarily. A previous nonzero first moment can persist after the current gradient becomes zero. Bias correction, second-moment history and epsilon also affect the update. A skipped missing gradient can have different framework semantics.',
      investigate:
        'Advance the sign-change-then-zero preset to its fourth step and inspect the raw first moment and adaptive subtraction.',
    },
  ],
  adamw: [
    {
      question: 'Why does adding an L2 penalty not reproduce AdamW?',
      answer:
        'The penalty gradient enters both Adam moment buffers and the adaptive denominator. AdamW keeps decay separate from those loss-gradient moments. The two operations therefore generally have different histories and update sizes.',
      investigate:
        'Start from empty buffers with zero loss gradients and compare coupled L2 with AdamW’s separate shrinkage.',
    },
  ],
  muon: [
    {
      question: 'Does the finite Muon transform make the weights orthogonal?',
      answer:
        'No. It transforms a momentum-based update candidate, not the weight matrix itself. Its finite polynomial output is approximate and need not have exact unit singular values. The ideal polar target is a separate mathematical reference.',
      investigate:
        'Compare the finite matrix output and its singular magnitudes with the separate ideal target while changing the iteration count.',
    },
  ],
  'learning-loop': [
    {
      question: 'Does following the negative gradient always lower loss?',
      answer:
        'The gradient is local. A sufficiently large finite step can overshoot even in a scalar quadratic. The toy updates both weight and bias simultaneously, so its residual multiplier depends on both paths. Generalization is a separate question from lowering this example’s loss.',
      investigate: 'Use x=2 and compare learning rates 0.1 and 1 before taking a step.',
    },
  ],
  'position-methods': [
    {
      question: 'Does a defined position formula guarantee useful unlimited context?',
      answer:
        'No. Defining numbers at a new position is different from validating the trained model at that length. Attention visibility, trained representations, numerical behavior and task performance also matter.',
      investigate:
        'Compare common shifts of the fixed content pair, then distinguish that arithmetic identity from running a model on longer input.',
    },
  ],
  'feature-representations': [
    {
      question: 'Can a coordinate change while a readout stays identical?',
      answer:
        'Yes. Rotating both a vector and its readout direction by the same orthogonal transformation preserves their dot product. This identity does not make every nonlinear network invariant to arbitrary basis changes.',
      investigate:
        'Rotate the feature view and compare hidden coordinates with the unchanged F0 readout.',
    },
  ],
  'causal-interventions': [
    {
      question: 'Can removing a path increase an output?',
      answer:
        'Yes. A path can contribute negatively, and nonlinear interactions can create further effects. An ablation’s sign and meaning depend on the metric, replacement and background.',
      investigate:
        'Select Ablate negative path and compare both clean-baseline and altered-baseline deltas.',
    },
  ],
  'post-training': [
    {
      question: 'Can the preference objective change while policy probabilities stay fixed?',
      answer:
        'Yes. The DPO loss depends on the policy/reference gap difference. Changing the reference while holding the policy fixed changes that objective, without changing what the policy currently assigns to either completion.',
      investigate:
        'Move only the reference completion gap and compare demonstration loss, policy probability and DPO loss.',
    },
  ],
  'numerical-precision': [
    {
      question: 'Does a numerically different result prove the mathematical rule changed?',
      answer:
        'No. Equivalent real-number expressions can round differently under different operation orders, intermediate precisions or fused kernels. Validate the implementation contract and appropriate tolerances, while handling nonfinite values explicitly.',
      investigate:
        'Compare the two binary32 addition orders, then the naive and max-shifted softmax calculations.',
    },
  ],
  'attention-comparison': [
    {
      question: 'Does FlashAttention make a full-context KV cache constant-size?',
      answer:
        'No. Its tiled execution avoids full attention-matrix materialization and reduces memory traffic. A full-context decoder still retains state for its earlier positions unless another architectural or retention mechanism changes that.',
      investigate:
        'Compare the persistent state rows with the separate temporary-matrix panel. Double the context while keeping head widths fixed.',
    },
    {
      question: 'Is the smallest state row automatically the best model?',
      answer:
        'No. These rows have different representation, visibility, precision and training requirements. A byte estimate cannot establish equivalent recall, quality, latency or compatibility with existing weights.',
      investigate:
        'Select the recurrent and windowed rows, then read their tradeoffs before comparing their growth curves.',
    },
  ],
  'linear-attention': [
    {
      question: 'Does fixed recurrent state mean exact unlimited recall?',
      answer:
        'No. This mechanism accumulates feature summaries under a different attention rule. Fixed state dimensions do not retain an individually addressable exact record for every past key/value. Capability requires architecture-specific evaluation.',
      investigate:
        'Increase context in the comparison and observe a flat recurrent-state curve; distinguish that storage result from a claim about retrieval accuracy.',
    },
  ],
  'cache-quantization': [
    {
      question: 'Does four-bit KV use exactly one quarter of sixteen-bit KV memory?',
      answer:
        'Only the ideal scalar-code payload has that ratio. Scale/offset metadata, padded groups, any high-precision residual buffers and allocation overhead change the real ratio.',
      investigate:
        'Compare packed GQA with ordinary GQA and inspect the declared group size and metadata rather than dividing only the bits.',
    },
  ],
  'attention-overview': [
    {
      question:
        'If an attention weight doubles, does that token’s influence on the final answer double?',
      answer:
        'Only its coefficient in that particular value-weighted sum changes directly. The row renormalizes, so other coefficients may change too. Value features can reinforce or cancel, Wo mixes head features, the residual adds another path, and later nonlinear transformations intervene. A probability coefficient is not a standalone measurement of final causal influence.',
      investigate:
        'Compare the weighted V coordinates before the output projection, then inspect the residual addition rather than stopping at the heatmap.',
    },
    {
      question:
        'Can an attention result contain a feature value larger than every value it averages?',
      answer:
        'For one head and coordinate, finite softmax weights are nonnegative and sum to one in exact arithmetic, so the weighted coordinate lies between the smallest and largest visible V coordinates. That bound applies before the output projection and residual addition. Those later operations can produce values outside the interval. Finite-precision rounding can introduce small deviations.',
      investigate:
        'Use the worked row and compare its output coordinates with the visible value vectors. Do not apply this bound to the projected residual output.',
    },
  ],
  'attention-scores': [
    {
      question: 'Does a larger query vector necessarily make attention sharper?',
      answer:
        'Multiplying the entire query by a positive scalar multiplies its dot products by that scalar when keys stay fixed. This increases finite score gaps and can sharpen the distribution. But changing one feature can increase some scores while decreasing others, and masked keys remain excluded regardless of score. “Bigger vector” alone does not describe a controlled intervention.',
      investigate:
        'Double both query coordinates in the worked example, then change just one coordinate. Compare the relative score gaps and probabilities.',
    },
    {
      question: 'Why does the score matrix have a key-position axis but no head-feature axis?',
      answer:
        'A score retains which query and key were compared but sums all corresponding feature products into one scalar. The head-feature axis Dh is contracted. The later multiplication by V performs another contraction, this time eliminating key slots U and recovering a head-feature output.',
      investigate:
        'Follow one score cell’s products in the microscope, then inspect the corresponding value-weighted result.',
    },
  ],
  'causal-mask': [
    {
      question: 'Why is the diagonal allowed if the model predicts the next token?',
      answer:
        'The input ID at position q is already known. Its hidden state predicts the following ID at q+1. Reading its own input features does not reveal the unknown target. A training target that was accidentally placed in that input position would be a different, invalid alignment.',
      investigate:
        'For query 0, inspect the allowed self-key and the excluded key at position 1. Then compare the next-token target alignment in the training guide.',
    },
    {
      question: 'Can I recover causality by setting future probabilities to zero afterwards?',
      answer:
        'Post-hoc zeroing without renormalization leaves a row whose weights sum below one. Renormalizing can recover the mathematical distribution over allowed finite scores in exact arithmetic, but a large forbidden score can make allowed probabilities underflow before that correction in floating point. Masking scores first expresses the dependency directly and avoids this failure. The reference implements pre-softmax exclusion.',
      investigate:
        'Toggle the scratch mask and watch both the excluded key and the probabilities of the remaining visible keys change.',
    },
  ],
  softmax: [
    {
      question: 'Why does adding the same constant to every logit change nothing?',
      answer:
        'Every exponential gains the same multiplicative factor, which cancels between numerator and denominator. Subtracting the maximum uses that invariance to keep exponentials bounded. Multiplying scores is different: it changes their pairwise gaps and generally changes the distribution.',
      investigate:
        'Inspect the row maximum, exponentials and denominator in the numerical cell explanation. Compare this with temperature scaling in the output guide.',
    },
    {
      question: 'Does a displayed zero always mean a position was masked?',
      answer:
        'No. A small finite value may round to zero at the displayed precision or underflow in stored arithmetic. A masked entry carries an explicit exclusion decision and contributes zero by construction. Read the visibility marker and the precise scalar, not just a rounded heatmap label.',
      investigate:
        'Inspect a known future cell’s mask provenance, then a small allowed probability.',
    },
  ],
  rope: [
    {
      question: 'If the cache moves a key to another slot, should its RoPE angle change?',
      answer:
        'No. The rotation belongs to the key’s absolute sequence position and feature pair, not its current allocation slot. This reference stores already rotated keys with a position map. Reinterpreting compact slots as new position IDs would change the dot products and therefore the computation.',
      investigate:
        'Compare retained absolute positions with slot numbers under sliding retention; keep the same absolute position selected in the rotation explanation.',
    },
    {
      question: 'Why does rotating both Q and K retain information about their distance?',
      answer:
        'For one frequency, the transpose of the query rotation composed with the key rotation is the rotation by their positional difference. Thus their dot product depends on the displacement as well as the unrotated features. This is a statement about the rotation algebra, not a guarantee of learned long-context accuracy.',
      investigate:
        'Compare pair index 0 with pair index 1. Their different frequencies give different angles for the same absolute position.',
    },
  ],
  rmsnorm: [
    {
      question: 'How can one output feature change when its own input stayed fixed?',
      answer:
        'All features in the token row contribute to the shared mean-square denominator. Changing another coordinate changes that denominator, so it can rescale the inspected output even though its input and gain are unchanged. This normalization mixes row statistics without mixing token positions.',
      investigate:
        'Keep feature 0 fixed and vary feature 1 in the worked example. Follow the denominator and both outputs.',
    },
    {
      question: 'Is RMSNorm exactly invariant to rescaling its input?',
      answer:
        'With epsilon ignored and a positive scaling factor, dividing by RMS cancels the scale. With nonzero epsilon, the cancellation is not exact, especially for very small inputs. Negative scaling also flips the normalized sign. The reference retains epsilon rather than claiming exact invariance in every numerical case.',
      investigate:
        'Inspect a zero row and a nonzero row. The epsilon term is visible in the denominator calculation.',
    },
  ],
  swiglu: [
    {
      question: 'Why can the “gate” be negative or greater than one?',
      answer:
        'SwiGLU uses SiLU(z) = z·sigmoid(z), not sigmoid(z) alone. The sigmoid factor lies between zero and one, but multiplication by z changes the output’s range and sign. The word gate describes a multiplicative role rather than a probability constraint.',
      investigate:
        'Set the scratch preactivation to a negative value, then above 2. Compare sigmoid, SiLU and the final element-wise product.',
    },
    {
      question: 'Where does mixing across intermediate coordinates happen?',
      answer:
        'The Hadamard gate multiplies corresponding F coordinates without summing. Wdown then forms each residual-width output coordinate by summing weighted contributions across F. Confusing these stages hides which coordinates can influence which outputs.',
      investigate:
        'Trace one product cell, then one down-projection cell and compare their operand counts.',
    },
  ],
  'kv-cache': [
    {
      question: 'Why can a confirmed token be absent from the cache?',
      answer:
        'Selection confirms an output ID from the logits of an already consumed position. The new ID has no hidden state or K/V until another forward call consumes it. Ending generation immediately after selection leaves the final output uncached, which is valid.',
      investigate:
        'Finish a short numerical run and compare the last confirmed output position with the last cached absolute position.',
    },
    {
      question: 'Can I reuse a cache after changing a token in the middle of the prefix?',
      answer:
        'Later hidden states depended on that token under the visibility rules, so their K/V is no longer a valid continuation of the edited prefix. Reuse requires a compatible model, tokenizer, positional scheme, policy and unchanged causal history. This app recalculates a capture when run inputs change.',
      investigate:
        'Change an input ID and compare the recomputed later activations rather than assuming one cached row can simply be patched.',
    },
  ],
  'cache-retention': [
    {
      question: 'Why is final retained state sometimes too small to compute the current call?',
      answer:
        'A multi-position call includes early queries that need early keys. Those keys may be outside the final window after the last query. Working K/V must support each query until it finishes; only then can persistent retention commit. R describes the end state, while U describes the working set.',
      investigate:
        'Increase T in the scratch calculator and compare P, U and R. For an initial prefill, P is zero even when U is large.',
    },
    {
      question: 'Does sparse retention always have bounded memory?',
      answer:
        'No. The illustrative fixed-stride policy keeps a recent window plus causal stride anchors. The anchor set grows with processed extent, while sliding-window retention alone is bounded by its window. Sparsity describes a selected pattern, not a universal memory guarantee.',
      investigate:
        'Increase S under the sparse policy and compare R with the same window under sliding retention.',
    },
  ],
  'head-sharing': [
    {
      question: 'Do queries that share K/V necessarily have the same attention weights?',
      answer:
        'Their query vectors still differ. Taking dot products of different queries with the same key vectors can give different score rows, hence different softmax weights and different weighted value outputs. K/V sharing does not average the query paths.',
      investigate:
        'Select two queries mapped to the same group, then compare their actual probability rows in the head observatory.',
    },
    {
      question: 'Why is query-head count absent from the direct KV payload formula?',
      answer:
        'Persistent K/V stores one pair per KV head, not per query head. Hq can still affect the architecture indirectly: if D stays fixed while Hq changes, Dh may change too. A fair payload comparison states which dimensions are being held constant.',
      investigate:
        'Read the active Hq, Hkv and Dh together in the head mapping and cache scratch calculations.',
    },
  ],
  sampling: [
    {
      question: 'Why can sampling choose an ID that is not the most probable?',
      answer:
        'A categorical draw allocates a region of the random number interval to every candidate with positive probability. A lower-probability ID has a smaller region, not an impossible one. Greedy is a separate maximum-selection rule.',
      investigate:
        'Use the same prefix and change only the trained sampling seed. Compare the unchanged first-call logits with potentially different selections.',
    },
    {
      question: 'Does lowering temperature make the model more knowledgeable?',
      answer:
        'Temperature rescales the same logits. It can concentrate probability around existing preferences but does not add information, change parameters or verify an answer. Later generated continuations can diverge because selected IDs feed into subsequent calls.',
      investigate:
        'Compare probabilities for the same fixed logits in the worked example before comparing whole generated completions.',
    },
  ],
}
