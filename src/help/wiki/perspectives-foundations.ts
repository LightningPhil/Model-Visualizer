import { detail, introduction, type TopicPerspectives } from './perspective-types'

export const foundationPerspectives: Record<string, TopicPerspectives> = {
  'reading-the-model': {
    introduction: introduction(
      'A language model is a chain of transformations that turns a known text prefix into preferences for what could come next. The diagram separates that chain into pieces you can inspect. Some pieces change how text is represented, some combine information from earlier positions, and some turn the final representation into a choice.',
      'Think of following a document through several editing passes. Each pass receives the current version and adds a different kind of revision. This is only a way to follow the sequence: there is no person reading the words inside the model, and the intermediate representation is a collection of numerical features rather than a hidden English explanation.',
      [
        'Start with how text becomes token IDs, then follow an ID into its embedding.',
        'Follow the ongoing hidden representation through attention and feed-forward updates in each layer.',
        'Find the output scores, the selection rule and the feedback path into the next call.',
        'At every step ask what enters, what changes, what leaves and what must remain available later.',
      ],
      'A graph explains a supported design; it does not prove that every visible intermediate is allocated by a production kernel. The teaching architecture, scripted simulation, untrained calculation and local trained checkpoint provide different kinds of evidence. Their banners tell you which one you are inspecting.',
    ),
    technical: [
      detail('technical-contract', 'Audit the computation at three levels', [
        'Architecture fixes the transformation and dependency structure. Execution fixes numerical order, rounding, fusion and scheduling. Storage fixes layouts, addressing and lifetimes. A graph node may be a mathematical operation that a fused kernel never materializes separately; a state view may expose a slice that has no independent allocation.',
        'For a proposed interpretation, specify the model/profile identity, the batch/call, semantic axes, absolute positions and parameter provenance. Then trace the contraction or element-wise rule and identify which operands are parameters, current activations or persistent state. This is the level at which a diagram can be checked against a numerical capture.',
      ]),
    ],
  },
  evidence: {
    introduction: introduction(
      'A convincing picture and a measured model result are different things. This tool lets you study structure, follow a scripted execution story, inspect actual arithmetic from an untrained model, or inspect a small trained checkpoint. Knowing which kind of evidence you have is essential to learning the right lesson.',
      'An architectural plan, a rehearsal and a recording of an actual performance can all be useful, but they answer different questions. Likewise, an illustrative heatmap can explain a mask without telling you how a trained model attended to your prompt. Actual untrained values explain a calculation without showing learned language behavior.',
      [
        'Read the workbench banner before interpreting any number.',
        'Use architecture views for structure and the numerical microscope for operand-by-operand calculations.',
        'Use the explicitly loaded trained checkpoint when the question concerns learned behavior.',
        'Keep the model identity and the kind of evidence in your notes when comparing runs.',
      ],
      'A value can be real arithmetic and still come from untrained parameters. A trained value can still be computed with a reference implementation whose rounding differs from another runtime. Neither distinction makes the view useless; it determines what conclusions it supports.',
    ),
    technical: [
      detail('technical-provenance', 'Reproducibility requires more than a seed', [
        'A reproducible numerical observation identifies parameter source, tokenizer, input IDs, absolute positions, attention policy, sampling rule and arithmetic implementation. A parameter seed affects untrained weights; a sampling seed affects categorical draws. They are different sources of variation.',
        'Reference arrays are stored in Float32, while the current implementation accumulates many operations using JavaScript Number before rounding an operation output. A GPU implementation may accumulate in another order or fuse operations. Treat equality within a declared tolerance separately from bitwise equality, and compare aligned semantic coordinates rather than display-cell locations.',
      ]),
    ],
  },
  'parameters-activations-state': {
    introduction: introduction(
      'Three different things can look like arrays in the inspector. Parameters define the model’s transformations. Activations are calculated for the input currently being processed. Persistent state is input-dependent information kept so a later call can continue efficiently. Their roles and lifetimes are different.',
      'A reusable recipe, the result of cooking a particular meal and prepared ingredients reserved for the next course have different lifetimes. That distinction helps here: fixed weights are reused, intermediate results depend on the current input, and cached keys and values carry selected earlier results forward. The recipe analogy describes reuse, not how knowledge is literally stored.',
      [
        'Identify whether a tensor is labelled as a parameter, an activation or state.',
        'Change an input and observe activations change while the model parameters remain fixed.',
        'Follow a decoder call to see which derived values are retained as K/V for the next call.',
        'Distinguish the live state from the viewer’s extra historical snapshots.',
      ],
      'Untrained weights are still parameters. Cached state is still derived from input; it does not become permanent learned knowledge. The teaching trace keeps more history than an efficient inference runtime would need, so its total memory use is not the same as its displayed logical cache payload.',
    ),
    technical: [
      detail(
        'technical-lifetimes',
        'Separate parameter sharing, activation reuse and trace retention',
        [
          'A projection parameter is reused across token positions within its layer. Independent decoder layers have distinct parameter tensors in this profile. Embedding/classifier tying is another explicit sharing relation; it does not make their activations identical.',
          'K/V caching reuses derived activations when an unchanged causal prefix makes them valid. A trace retains historical snapshots for inspection even after they cease to be live decoder state. Count unique parameter storage, live activations, resident K/V and trace snapshots separately; adding every visible tensor as though it were a separate persistent allocation double-counts views and history.',
        ],
      ),
    ],
  },
  tensors: {
    introduction: introduction(
      'A tensor is an organized collection of numbers. Its axes tell you what you are selecting: a sequence, a token position, a head or a feature. Reading those meanings is more useful than simply counting how many dimensions an array has.',
      'Imagine a collection of recordings indexed by session, microphone and time. Choosing a microphone is different from choosing a moment, even if both indexes happen to have the same range. Tensor axes work the same way: equal lengths do not make two axes interchangeable. This is an indexing comparison, not a physical storage model.',
      [
        'Read the name of each axis before selecting a slice.',
        'Ask whether an operation preserves, rearranges, expands or sums away an axis.',
        'Compare absolute token positions when the cache has discarded earlier entries.',
        'Use the transpose control to change the view, while keeping the stored axis meanings in mind.',
      ],
      'A shape is a description of coordinate organization. It does not establish that an imported equation is correct, and it does not specify memory strides or physical allocation. The same number of cells can be organized into different, incompatible meanings.',
    ),
    technical: [
      detail(
        'technical-indexing',
        'Index notation makes a contraction unambiguous',
        [
          'For a batched projection, fix the batch and token indexes, then sum the input-feature index against the matching weight axis. Unmentioned batch axes identify independent instances rather than additional terms in that sum. A transpose changes which coordinate a display row/column refers to; a reshape needs an explicit element ordering.',
          'For attention, query and working-key position maps are part of the tensor semantics. Two probability tensors with equal dimensions but different absolute key maps are not cellwise aligned. This is why baseline subtraction requires axes and position metadata in addition to matching shape.',
        ],
        String.raw`Y_{b,t,f}=\sum_d X_{b,t,d}W_{d,f},\qquad [B,T,D]\times[D,F]\longrightarrow[B,T,F]`,
      ),
    ],
  },
  'linear-maps': {
    introduction: introduction(
      'A linear projection makes a new set of features by combining the existing ones with fixed weights. Each output feature uses its own mixture of the input features. The model learns those mixtures during training and reuses them during inference.',
      'Think of a mixing desk that creates several output channels from the same input channels. Different output channels use different settings. The model’s feature coordinates are usually not recognizable instruments, but the idea of making several different mixtures explains why the same input can become queries, keys and values.',
      [
        'Choose an output feature and identify the input features it reads.',
        'Follow the separate weight attached to each input contribution.',
        'Combine those contributions to obtain the chosen output feature.',
        'Compare another output feature: it reads the same input through different weights.',
      ],
      'A projection mixes features; it is not a mere change of display orientation. An element-wise gate combines matching coordinates differently. Changing weights in a labelled toy example is an intervention on that toy, not a resize or retraining of the checkpoint.',
    ),
    technical: [
      detail(
        'technical-linearity',
        'Derive one cell, then distinguish linearity from expressiveness',
        [
          'With no bias, each output cell is a dot product between an input row and a weight column. The contraction removes the input-feature axis and leaves the output-feature axis. Applying the same matrix to each position shares parameters across positions without directly mixing those positions.',
          'Composition of linear maps alone is still linear: two consecutive feature matrices can be multiplied into one matrix, provided no intervening nonlinear or input-dependent operation changes the mapping. Transformers gain more expressive behavior through normalization, nonlinear gates and attention weights that depend on the input. A rectangular matrix need not be invertible, and a projection’s output does not automatically retain all input information.',
        ],
        String.raw`y_f=\sum_{d=0}^{D-1}x_dW_{d,f},\qquad (xW_1)W_2=x(W_1W_2)`,
      ),
    ],
  },
  tokenization: {
    introduction: introduction(
      'The model does not begin with words as you see them on screen. A tokenizer turns encoded text into vocabulary IDs. A token may be a whole word, a fragment, punctuation or part of a character’s byte representation. The model’s first learned lookup uses those IDs.',
      'Picture packing a message using a fixed collection of reusable pieces. Frequently useful byte sequences can be represented by larger pieces; other parts remain split. Ranked byte-pair merging chooses which neighboring pieces combine. It is a segmentation procedure, not a judgment about the meaning or likelihood of the sentence.',
      [
        'Encode the original text into bytes.',
        'Inspect the permitted neighboring merges and choose the best available rank.',
        'Repeat merging until no permitted pair remains.',
        'Use the resulting vocabulary IDs with the matching embedding matrix.',
      ],
      'The small teaching tokenizer is not a production tokenizer. Other tokenizers may normalize text or apply different segmentation rules. A checkpoint’s tokenizer is part of its identity, so swapping an unrelated vocabulary while retaining the same numeric IDs changes what the model receives.',
    ),
    technical: [
      detail('technical-bpe', 'Merge priority and byte identity are separate invariants', [
        'For ranked BPE, a valid merge is determined by the current neighboring pieces and the merge-rank table. After a merge, adjacency changes; choosing every pair that was originally available can therefore yield a different segmentation from repeatedly choosing the current best pair.',
        'The byte-level teaching adapter keeps base coverage for every byte. Reconstruct bytes from the complete token sequence before decoding UTF-8: a token can end inside a multibyte code point. An individual token’s display label is not its authoritative byte identity. Production pre-tokenization, special-token handling and templates require their own explicit rules.',
      ]),
    ],
  },
  embeddings: {
    introduction: introduction(
      'An embedding lookup replaces each token ID with a reusable feature vector. It gives the rest of the model a representation it can transform. The numeric size of an ID is just an address; it does not tell you how important or meaningful the token is.',
      'Think of choosing a preset from a catalogue: the preset number selects its stored settings, rather than being a quantity you average with another preset number. A token embedding similarly selects a parameter row. Later layers adapt the representation to context, so the initial lookup is not the final interpretation of a word.',
      [
        'Select a vocabulary ID.',
        'Locate the corresponding embedding row.',
        'Copy or view that row as the token’s starting hidden representation.',
        'Follow later attention and feature updates to see how context changes the representation.',
      ],
      'The same token ID starts from the same embedding row in this profile, but can develop different hidden states at different positions. Tied embedding/output weights share parameters, not activations. The untrained and trained workbenches explicitly distinguish their tying choices.',
    ),
    technical: [
      detail(
        'technical-gather',
        'A gather can be written as a one-hot product without allocating one',
        [
          'The embedding result at a token position selects a row of the vocabulary-by-feature parameter matrix. This is mathematically equivalent to multiplying a one-hot vocabulary vector by that matrix. Implementations generally gather the row directly instead of allocating the full one-hot vector.',
          'When classifier weights are tied, the vocabulary projection uses the transposed embedding parameters under this presentation convention. The forward uses the same parameter information in a different contraction. The untrained numerical profile keeps these matrices independent, whereas Stories260K identifies its tied classifier.',
        ],
        String.raw`X_{b,t,d}=E_{\mathrm{id}_{b,t},d},\qquad \text{tied classifier: }z_v=\sum_d h_dE_{v,d}`,
      ),
    ],
  },
  residual: {
    introduction: introduction(
      'The residual stream is the ongoing representation passed through the decoder. A sublayer proposes an update, and the block adds that update to what was already there. The model can build on the current representation instead of replacing it wholesale at every small operation.',
      'Imagine revising a working draft by adding edits while keeping the draft as the starting point. Attention provides one kind of update, and the feed-forward network provides another. This picture explains the bypass lines in the graph, but the actual stream is a numerical vector rather than a readable draft.',
      [
        'Carry the incoming stream along the bypass path.',
        'Calculate an attention update from the normalized view and combine it with the stream.',
        'Calculate a feed-forward update from the updated, normalized representation.',
        'Pass the result to the next layer, whose parameters and state are independent.',
      ],
      'Preserving an identity path does not guarantee exact recovery of the original input or a fixed human meaning for every feature. Pre-norm, post-norm and parallel branches have different dependencies. Follow the actual graph rather than assuming every decoder block has the same arrangement.',
    ),
    technical: [
      detail(
        'technical-residual',
        'Separate the identity term from the learned update',
        [
          'At matching coordinates, residual addition preserves the input shape and adds both operands. The attention branch in this sequential pre-norm profile reads normalized input; the identity branch bypasses normalization. The next feed-forward branch reads the already updated stream.',
          'For a scalar differentiable residual map, the local derivative includes an identity contribution plus the update’s derivative. This helps explain the architecture’s identity path; it is not a promise that arbitrary deep training is stable or that information can never be canceled. The forward explorer does not perform backpropagation.',
        ],
        String.raw`y=x+f(x),\qquad \frac{dy}{dx}=1+f'(x)`,
      ),
    ],
  },
  rmsnorm: {
    introduction: introduction(
      'RMSNorm controls the overall scale of a token’s feature vector before another transformation uses it. It measures the row’s typical magnitude, rescales the row and applies learned feature gains. This gives the following sublayer a more controlled input scale.',
      'Think of adjusting the overall level of a recording while preserving the relationships among its channels, then applying channel-specific settings. LayerNorm makes a different adjustment: it first removes the row’s common baseline before controlling its spread. Moving every input by the same amount therefore separates their behavior. This picture concerns numerical scale, not the semantic importance of a token.',
      [
        'Measure the feature row’s overall magnitude.',
        'Use a small stabilizing term so an all-zero row remains well-defined.',
        'Rescale each feature using the shared row statistic.',
        'Apply the gain for each feature and pass the row onward.',
      ],
      'RMSNorm does not subtract the feature mean and does not turn features into probabilities. It also does not compare different token positions. The exact stabilizing term and placement in the block are implementation choices shown in the technical treatment.',
    ),
    technical: [
      detail(
        'technical-rms-dependence',
        'Derive the shared-denominator dependence',
        [
          'Write the root mean square with epsilon as a shared denominator r. A normalized output depends on its own numerator and on every input coordinate through r. Differentiating shows why changing another input feature can change the selected output even if its numerator stays fixed.',
          'In the formula, i labels an output feature, j an input feature, D is the feature count and g_i is its fixed gain during differentiation. The Kronecker delta delta_ij is one when i=j and zero otherwise. First compute dr/dx_j = x_j/(D r); applying the quotient rule to g_i x_i/r then produces the direct numerator term and the shared-denominator term.',
          'With epsilon ignored, positive input rescaling cancels in the normalization. With a nonzero epsilon the cancellation is approximate, especially near zero. The reference uses explicit epsilon and per-feature gain, so examine those operands when reconstructing a result.',
        ],
        String.raw`r=\sqrt{D^{-1}\sum_jx_j^2+\epsilon},\quad y_i=g_i x_i/r,\quad \frac{\partial y_i}{\partial x_j}=g_i\left(\frac{\delta_{ij}}r-\frac{x_ix_j}{D r^3}\right)`,
      ),
    ],
  },
  swiglu: {
    introduction: introduction(
      'The feed-forward network develops and recombines features at each token position. In SwiGLU, two different projections form separate branches. One branch is passed through a smooth activation and then modulates the other before the result returns to the residual stream.',
      'Picture two sets of controls: one prepares feature content, while the other changes how strongly matching pieces of that content pass through. This is only a guide to the branch structure. The gate is a numerical feature and can be negative; it is not a switch that always opens between closed and fully open.',
      [
        'Project the input into separate gate and up branches.',
        'Apply the smooth activation to the gate branch.',
        'Multiply matching features from the two branches without summing them yet.',
        'Use the down projection to mix the intermediate features back into a residual update.',
      ],
      'This dense feed-forward network acts on each position using shared matrices. It does not directly choose other token positions or route tokens among experts. Context is already present in its input because previous attention transformed the residual stream.',
    ),
    technical: [
      detail(
        'technical-gate-derivative',
        'The gate has its own nonlinear response',
        [
          'SiLU multiplies an input by its sigmoid. Differentiating exposes both the direct sigmoid term and the sigmoid’s input-dependent change. The output can be negative, and its magnitude is not constrained to the unit interval.',
          'Here sigma(z) = 1/(1+exp(−z)), so sigma′(z) = sigma(z)(1−sigma(z)); the product rule gives the displayed SiLU derivative. In m_f, f indexes one intermediate feature, a_f is the gate preactivation and b_f the up-branch value. For fixed b_f, dm_f/da_f = b_f SiLU′(a_f): even the direction of a gate-coordinate effect can depend on the sign of the other branch.',
          'For a fixed intermediate coordinate, the subsequent product uses exactly one activated gate coordinate and one up coordinate. Only the down projection contracts the intermediate-feature axis. In a bias-free dense implementation, the gate/up/down matrices contain three residual-width-by-intermediate-width sets of parameters.',
        ],
        String.raw`\operatorname{SiLU}'(z)=\sigma(z)+z\sigma(z)(1-\sigma(z)),\qquad m_f=\operatorname{SiLU}(a_f)b_f`,
      ),
    ],
  },
}
