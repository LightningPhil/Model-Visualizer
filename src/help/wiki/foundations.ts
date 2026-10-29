import type { WikiArticle, WikiSection } from './types'

export const section = (
  id: string,
  title: string,
  paragraphs: string[],
  equation?: string,
): WikiSection => ({ id, title, paragraphs, equation })

export const foundations: WikiArticle[] = [
  {
    id: 'reading-the-model',
    title: 'How to read a model',
    category: 'Start here',
    summary:
      'Follow information through the diagram and distinguish a model’s design, its calculated values, and the software that executes it.',
    aliases: [
      'beginner',
      'start',
      'architecture',
      'decoder',
      'transformer',
      'overview',
      'what am I looking at',
    ],
    kinds: ['decoder', 'model-expansion', 'input', 'x', 'ffx', 'current'],
    graphs: ['overview', 'whole-model', 'whole-model-heads', 'block'],
    prerequisites: [],
    related: ['evidence', 'tensors', 'residual', 'prefill-decode'],
    sources: ['attention'],
    sections: [
      section('journey', 'One token’s journey', [
        'The basic prediction problem is conditional: given the IDs already available, estimate a distribution over the next ID. Repeating this operation produces a sequence. Within one forward call, successive layers refine each current position’s representation; across calls, the selected output becomes part of a new input. Layer depth and generation time are therefore different directions through the diagram.',
        'A decoder takes a sequence of token IDs and calculates a distribution for a next ID. Tokenization turns bytes into IDs. An embedding lookup gives each ID a vector. Successive decoder layers transform these vectors, attention mixes information from visible positions, and feed-forward networks transform each position’s features. A final normalization and vocabulary projection produce logits; a selection rule chooses an ID.',
        'Read a connection as a specific operation or carried tensor. Click its label: a residual addition, a state read and a matrix product have different meanings even if their lines look similar. Expanding a component opens a more detailed study of its internals. The whole-model view instantiates each main layer separately from the authored templates.',
      ]),
      section('three-questions', 'Ask three questions at every component', [
        'What enters, what changes, and what leaves? Track both the information and its axes. An attention score reduces matching feature coordinates but retains query and key positions. A feed-forward projection changes feature width while preserving token positions. An addition combines equally shaped values element by element.',
        'Then ask what survives. Parameters are reused across calls; ordinary activations are temporary; K/V state persists for future attention. Repeated layers have distinct parameters and distinct cache state in this profile. A box marked “six layers” is a compact description of six transformations, not six applications of shared weights.',
      ]),
      section('scope', 'Design, execution and storage are different views', [
        'Architecture describes the mathematical operations and allowable dependencies. An execution kernel decides how to calculate them efficiently. Physical storage decides where values live. A logical score matrix does not imply that a GPU kernel stores that entire matrix, and a cache slice does not imply a separate allocation.',
        'The foundations document combines a causal decoder, pre-RMSNorm, RoPE and SwiGLU for teaching. It is not the original Transformer’s exact architecture or a claim about every language model. Imported documents may describe something different; their own explanations and references appear under “Model document” in this wiki.',
      ]),
    ],
    experiment: {
      title: 'Trace a path before looking at numbers',
      steps: [
        'Open Transformer block and select Attention RMSNorm.',
        'Follow self-attention into its detailed graph, then select Q · Kᵀ and its outgoing connection.',
        'Return to the block and find both residual additions.',
      ],
      observation:
        'Identify which operations mix positions, which mix features, and which preserve the identity path. Then use the tensor laboratory to inspect actual arithmetic.',
    },
  },
  {
    id: 'evidence',
    title: 'What the numbers actually mean',
    category: 'Start here',
    summary:
      'Architecture, synthetic illustrations, untrained numerical runs and trained activations answer different questions.',
    aliases: [
      'provenance',
      'real',
      'synthetic',
      'simulation',
      'untrained',
      'trained',
      'weights',
      'truth',
      'limitations',
    ],
    kinds: [],
    graphs: [],
    prerequisites: [],
    related: [
      'reading-the-model',
      'parameters-activations-state',
      'tensor-laboratory',
      'trained-model',
      'attention-interpretation',
    ],
    sources: ['stories'],
    sections: [
      section('modes', 'Four distinct kinds of evidence', [
        'Architecture explorer: symbolic shapes resolve from the teaching configuration and prompt length. The byte-BPE tokenizer really executes its small teaching algorithm. Architecture-lab attention heatmaps are labelled synthetic: they illustrate visibility and head sharing, rather than measuring a trained network.',
        'Graphical simulation: packets and choices follow scripted teaching scenarios. Position counts, cache retention and rollback are calculated, but there are no learned activations or computed language-model logits. Animation speed is a playback choice, not a benchmark.',
        'Tensor laboratory: a separate two-layer reference model executes actual operations using seeded, untrained parameters. Its arrays and cell calculations are real results of that reference computation. Its opaque output IDs do not represent meaningful language.',
        'Local trained model: the bundled Stories260K weights and matching tokenizer produce actual trained-model activations and text. This is a small, pinned base completion model; its behavior is evidence about that checkpoint and input, not about an arbitrary imported architecture or a large chat model.',
      ]),
      section('arithmetic', 'A recorded number has a calculation history', [
        'Two checks must be separated. Reconstructing a scalar from its operands checks the implementation of an operation. Testing whether a trained model completes a task checks behavior on specified inputs. A perfectly reconstructed calculation can still belong to a weak or untrained model; a plausible completion can still conceal a wrong implementation.',
        'The numerical workbenches store parameters and operation outputs in Float32 arrays and use JavaScript Number arithmetic for accumulation before rounding outputs to Float32. This reference path can differ from a GPU’s fusion, accumulation order and lower-precision arithmetic. Matching an equation does not promise bitwise equality across execution systems.',
        'A trace is calculated ahead of playback. The timeline controls what recorded results you can inspect, rather than pausing an operating GPU kernel. Trace snapshots also consume memory beyond the logical live K/V payload shown in the cache strip.',
      ]),
      section('claims', 'Use the right experiment for the question', [
        'To understand shapes, use the architecture explorer. To understand an individual scalar, use the numerical microscope. To ask what a learned model actually did, explicitly load and generate with the trained profile. To understand causal influence, inspecting a heatmap alone is insufficient: the value vectors, output projection, residual path and subsequent layers also matter.',
        'All of these features run locally. Opening help or importing a workspace does not load or generate with the checkpoint. Model documents save settings, explanations and study state; a separate trace export carries the large calculated arrays and provenance.',
      ]),
    ],
  },
  {
    id: 'parameters-activations-state',
    title: 'Parameters, activations and persistent state',
    category: 'Foundations',
    summary:
      'Identify what is learned, what is calculated for this input, and what must survive into the next decoder call.',
    aliases: [
      'parameter',
      'activation',
      'weights',
      'state',
      'lifetime',
      'memory',
      'gain',
      'inference',
      'training',
    ],
    kinds: [],
    graphs: [],
    prerequisites: ['tensors'],
    related: ['embeddings', 'linear-maps', 'kv-cache', 'training-objective', 'evidence'],
    sources: ['attention', 'gqa', 'swiglu'],
    sections: [
      section('parameters', 'Parameters define the transformation', [
        'Embedding rows, projection matrices and normalization gains are parameters. Training adjusts them; an ordinary inference run reads them without updating them. They are shared across token positions within their layer. Distinct layers in this profile have independent parameter sets, even though they use the same sequence of operation types.',
        'A seeded untrained matrix is still a parameter tensor: “parameter” describes its role, while “trained” describes its provenance. Changing a numerical parameter seed constructs a different untrained model. A sampling seed instead controls output draws and leaves the parameter values alone.',
      ]),
      section('activations', 'Activations depend on the current input', [
        'Hidden states, projected queries, score rows, attention probabilities and vocabulary logits are activations calculated during a forward pass. Different positions and prompts generally give different values even though they use the same projection matrix. Changing the prompt length changes activation shapes without necessarily changing parameter shapes.',
        'For a row-vector projection X[T,D] × W[D,F], the input contains T×D scalars, the weight contains D×F, and the output contains T×F. Increasing T changes the input/output sizes, while the same W can apply to all positions. This is a shape count, not a guarantee about an optimized kernel’s peak allocation.',
      ]),
      section('persistent', 'Some activations become persistent inference state', [
        'There is also persistent training state: an optimizer can keep momentum or moment estimates across updates. Those buffers describe gradient history, while inference K/V describes a processed input prefix. A weight checkpoint can support inference without an optimizer checkpoint; resuming the same training trajectory generally needs both.',
        'Cached keys and values are derived from the input, so they are activations by origin. Their special role is to persist across decoder calls. They are not new learned weights and do not permanently modify the checkpoint. A different request or edited prefix needs its own compatible cache state.',
        'Persistence is about the runtime’s dependency requirements. The educational trace also retains historical copies so you can inspect earlier operations. Those snapshots are additional application storage, not evidence that a production decoder must keep every old intermediate tensor.',
      ]),
      section('counting', 'Count different objects separately', [
        'A bias-free dense SwiGLU with gate/up matrices [D,F] and down matrix [F,D] contains 3DF matrix parameters. Synthetic example: D = 4 and F = 6 gives 72 weights in these three matrices. This excludes normalization gains and every other sublayer. Its per-call intermediate activations also depend on T.',
        'KV payload instead depends on retained positions, layers and KV heads. Parameter bytes, live activations, cache payload and recorded trace history answer different memory questions. Selecting a cache scalar size in the architecture builder does not change the actual parameter dtype or quantize the numerical run.',
      ]),
    ],
    experiment: {
      title: 'Compare roles across two decoder calls',
      steps: [
        'Open the tensor laboratory and inspect a projection in All parameter tensors.',
        'Inspect its output during prefill, then move to a later decode call.',
        'Compare the parameter shape, activation shape and cache position count.',
      ],
      observation:
        'The parameter matrix is reused. Current activations reflect the new input, while cached K/V carries selected earlier activations forward.',
    },
  },
  {
    id: 'tensors',
    title: 'Tensors, axes and shape notation',
    category: 'Foundations',
    summary:
      'Read [B,T,D] as named axes, follow contractions, and separate reshaping from mathematical computation.',
    aliases: [
      'dimensions',
      'shape',
      'matrix',
      'vector',
      'scalar',
      'batch',
      'B',
      'T',
      'S',
      'R',
      'P',
      'U',
      'D',
      'Hq',
      'Hkv',
      'Dh',
      'F',
      'V',
      'L',
      'C',
      'transpose',
      'reshape',
      'broadcast',
    ],
    kinds: ['concat', 'attention-head'],
    graphs: [],
    prerequisites: [],
    related: ['linear-maps', 'attention-scores', 'kv-cache', 'head-sharing'],
    sources: ['attention', 'gqa'],
    sections: [
      section('axes', 'An axis tells you what an index means', [
        'A scalar is one number, a vector has one axis, and a matrix has two. A tensor generalizes this to more axes. X[B,T,D] means that selecting a batch item and token position leaves a D-feature vector. Axis order matters: [B,Hq,T,Dh] and [B,T,Hq,Dh] can contain rearrangements of the same values while requiring different indexing.',
        'B is batch size (1 here); T is this call’s query count; D is residual width; L is decoder layers; Hq and Hkv count query and K/V heads; Dh is features per head; F is feed-forward intermediate width; V is vocabulary size; C is configured capacity. This teaching profile uses D = Hq × Dh and even Dh for paired rotary features. Those are profile constraints, not universal transformer identities.',
      ]),
      section('positions', 'Sequence extent is not always cache length', [
        'S is the absolute processed extent. P counts retained prefix positions before the call; U = P + T counts working key slots during attention; R counts positions retained after the call. Full-prefix decoding can make these look interchangeable, but eviction and multi-query prefill separate them. P and R can be zero for empty state.',
        'For a synthetic sliding-window example, take a processed prefix of 10 positions and window W = 4. Before the next call P = 4; with T = 1 the working set has U = 5, the processed extent becomes S = 11, and final retention has R = 4. The mask excludes the oldest working position for the new query. Slot zero need not mean absolute position zero.',
      ]),
      section('operations', 'Which axes disappear?', [
        'Use an index as a binding contract: in Y[b,t,f] = sum_d X[b,t,d]W[d,f], b, t and f are fixed while d is the summation index. The same letter on two factors identifies matching coordinates, not two independent loops. Here “tensor” means the machine-learning array representation with named axes; a shape alone does not specify a physical tensor transformation law or a memory layout.',
        'A contraction multiplies matching coordinates and sums along an axis. Q[T,Dh] × Kᵀ[Dh,U] contracts Dh, leaving scores[T,U]. Attention probabilities[T,U] × V[U,Dh] contract U, leaving result[T,Dh]. Batch and head axes select independent versions of these operations.',
        'A transpose permutes axes. A reshape groups or splits coordinates while preserving a specified element order. Neither is a learned projection. Broadcasting reuses a smaller operand, such as a feature gain across tokens; it must be explicit. Equal shapes are necessary for a plain residual addition, but equal shapes alone do not establish compatible meanings or positions.',
      ]),
    ],
    experiment: {
      title: 'Predict the result shape',
      steps: [
        'Select Query projection and compare its input, weight and output axes.',
        'Select Q · Kᵀ, then Attention · V.',
        'Switch from prefill to decode and repeat.',
      ],
      observation:
        'T shrinks to one during ordinary decode; the working key axis can remain long. The contracted feature axis and key-position axis disappear at different operations.',
    },
  },
  {
    id: 'linear-maps',
    title: 'Linear projections and matrix multiplication',
    category: 'Foundations',
    summary:
      'Understand how a learned weight matrix mixes features, and calculate a single output coordinate.',
    aliases: [
      'matmul',
      'dot product',
      'weight',
      'projection',
      'Wq',
      'Wk',
      'Wv',
      'Wo',
      'sum',
      'multiply',
      'contraction',
    ],
    kinds: ['q', 'k', 'v', 'project', 'newk', 'newv', 'gate', 'up', 'down', 'lmhead'],
    graphs: [],
    prerequisites: ['tensors'],
    related: ['attention-scores', 'embeddings', 'swiglu', 'logits'],
    sources: ['attention', 'swiglu'],
    sections: [
      section(
        'equation',
        'One output feature is a weighted sum',
        [
          'This app uses row-vector notation: an input X[B,T,D] multiplies W[D,F] to produce Y[B,T,F]. For each batch item and token, every output coordinate reads all D input coordinates with a different column of weights. The same W is reused at every position in that layer; changing token positions does not select another projection matrix.',
          'The numerical foundations profile has no projection biases. A model with a bias would add a feature vector after the product. Other implementations may store a transposed weight matrix: compare the operation and labelled axes, not just the array’s apparent orientation.',
        ],
        String.raw`Y_{b,t,f}=\sum_{d=0}^{D-1}X_{b,t,d}W_{d,f}`,
      ),
      section('example', 'Worked scalar example · synthetic', [
        'Take x = [2, −1] and W with rows [3, 4] and [5, −2]. The first output is 2×3 + (−1)×5 = 1. The second is 2×4 + (−1)×(−2) = 10. The output [1,10] mixes the two input features differently in each coordinate.',
        'A Hadamard product would instead multiply matching coordinates without summing. A reshape would only rearrange coordinates. The graph keeps these operations distinct because they have different information flow, even when a tiny example happens to have equal input and output widths.',
      ]),
      section('parameters', 'Parameters are fixed during this forward pass', [
        'The word projection is conventional neural-network terminology for a learned linear map. W need not be square, idempotent, orthogonal, or invertible. In particular, xW is not automatically a perpendicular projection onto a subspace. An input direction n with nW = 0 is lost by this map; later nonlinearities do not recover that missing information from this output alone.',
        'Training learns the coefficients; inference reads them. Q, K and V are three different projections of the same normalized hidden state, with independent weights and different downstream uses. In GQA, Q projects to Hq×Dh while K and V each project to Hkv×Dh.',
        'In the microscope, select one result cell to see its summands and follow an operand back to an input or parameter. Large dot products are presented as a navigable calculation; a particular coordinate is a numerical feature, not automatically a human-readable concept.',
      ]),
    ],
  },
  {
    id: 'tokenization',
    title: 'Text, UTF-8 and byte-pair encoding',
    category: 'Input',
    summary:
      'See why a token is not a word, how ranked merges work, and why the matching tokenizer belongs to the model.',
    aliases: [
      'BPE',
      'byte',
      'unicode',
      'UTF-8',
      'vocabulary ID',
      'tokenizer',
      'merge rank',
      'special tokens',
      'prompt',
    ],
    kinds: ['tokenizer', 'text', 'utf8', 'bpe', 'ids'],
    graphs: ['tokenization'],
    prerequisites: [],
    related: ['embeddings', 'trained-model', 'logits'],
    sources: ['bpe', 'stories'],
    sections: [
      section('bytes', 'Start with bytes, not words', [
        'The teaching tokenizer first encodes text as UTF-8 bytes. ASCII characters usually use one byte; many other characters use multiple bytes. A token can contain several characters or only part of a multibyte character. A displayed piece is a convenience for inspection, while its byte span and ID are the exact representation.',
        'This tokenizer keeps a base representation for every byte, allowing lossless byte reconstruction. Decoding a whole sequence joins token bytes before interpreting UTF-8. Interpreting each token separately can show replacement characters for fragments that become valid only when joined.',
      ]),
      section('merges', 'Ranked BPE chooses allowed neighboring merges', [
        'Separate the training of a vocabulary from its use. A production BPE training procedure chooses merge rules from a corpus; encoding a new prompt applies the already fixed ranks. The small authored vocabulary here supplies those rules directly. Encoding your prompt does not learn new vocabulary entries or update model weights.',
        'Begin with individual byte pieces. Find adjacent pairs whose concatenation is an allowed merge; choose the available pair with best (lowest) rank; merge and repeat until no allowed pair remains. Rank is learned vocabulary metadata in a production tokenizer; it is not a semantic similarity score or a token probability from the language model.',
        'Synthetic merge example: with pieces a, b, c and rules ab at rank 5 and bc at rank 2, merge b+c first. The sequence becomes a, bc. If a+bc has no rule, it stops there, even though ab was initially available. Actual rules and tie handling are inspectable in the teaching implementation.',
      ]),
      section('identity', 'A tokenizer is part of a checkpoint’s identity', [
        'An ID has meaning only within a specific vocabulary. Swapping tokenizers while keeping the same numeric IDs generally changes which embedding row is used. The local trained path therefore uses its pinned matching tokenizer, not the architecture explorer’s demonstration BPE.',
        'This small teaching vocabulary is not a production tokenizer. Production pipelines may add pre-tokenization, normalization, control tokens and chat templates. The generic explorer does not silently implement those features, and an unsupported imported adapter disables token labs rather than pretending to tokenize it correctly. A chat template belongs before the model’s ID sequence; it is not an attention mechanism.',
      ]),
    ],
    experiment: {
      title: 'Inspect a multibyte character',
      steps: [
        'Open Tokenization and enter “café 🧠” in the prompt.',
        'Move the BPE merge step back to zero and inspect the byte pieces.',
        'Advance through the ranked merges, comparing byte spans and token IDs.',
      ],
      observation:
        'Character count, byte count and token count differ. Merging changes the segmentation; it does not change the underlying decoded bytes.',
    },
  },
  {
    id: 'embeddings',
    title: 'Embedding lookup and vocabulary identity',
    category: 'Input',
    summary: 'A token ID selects a learned vector; its numeric magnitude does not express meaning.',
    aliases: [
      'lookup',
      'gather',
      'embedding table',
      'tied weights',
      'vocabulary',
      'token features',
    ],
    kinds: ['embedding', 'lookup', 'shifted', 'embed1', 'embed2'],
    graphs: [],
    prerequisites: ['tokenization', 'tensors'],
    related: ['linear-maps', 'residual', 'logits', 'trained-model'],
    sources: ['attention', 'stories'],
    sections: [
      section(
        'gather',
        'Gather a row',
        [
          'An embedding matrix E[V,D] has one D-feature row per vocabulary ID. For token ID i, the model gathers E[i,:]. B sequences with T current IDs each become X[B,T,D]. The ID is an index: ID 100 is not inherently “twice as meaningful” as ID 50, and nearby IDs need not have related meanings.',
          'A lookup can be described mathematically as a one-hot vector times E, but the reference implementation gathers the selected row directly. It does not need to allocate a full one-hot vocabulary vector for each token.',
        ],
        String.raw`X_{b,t,d}=E_{\mathrm{id}_{b,t},d}`,
      ),
      section('context', 'The same token can develop different contextual vectors', [
        'At lookup, repeated IDs select the same parameter row. After attention and feed-forward layers, their hidden states can differ because they occupy different positions and have different visible prefixes. The embedding is the starting representation, not the final meaning of a token in context.',
        'In this foundations profile RoPE rotates projected queries and keys inside attention. It does not add a learned position vector to the embedding. Other architectures use different positional methods, so consult the current model’s authored description.',
      ]),
      section('tying', 'Tied output weights are a specific parameter choice', [
        'For a synthetic lookup, IDs [2,2,0] select rows [E[2,:],E[2,:],E[0,:]] before any contextual processing. Neither averaging the IDs nor interpolating between their numeric values is part of this operation. When output weights are tied, a candidate logit instead compares the final hidden row with one embedding row; it does not invert the original gather.',
        'The vocabulary projection converts final hidden features back to V logits. A tied model uses the embedding matrix’s transpose for this output map; an untied model has a separate output matrix. Tying shares parameters, not input and output activations.',
        'The untrained numerical profile uses independent embedding and output weights. The bundled trained Stories260K profile identifies its tied classifier explicitly. A diagram that has both an embedding box and an output box does not by itself tell you whether the physical parameter storage is shared.',
      ]),
    ],
  },
  {
    id: 'residual',
    title: 'Residual stream and decoder blocks',
    category: 'Foundations',
    summary:
      'A layer adds an attention update and a feed-forward update to an ongoing vector stream.',
    aliases: [
      'skip connection',
      'identity',
      'pre norm',
      'pre-norm',
      'add',
      'block output',
      'layers',
    ],
    kinds: ['input', 'add1', 'add2', 'decoder', 'mainstate', 'hidden', 'context'],
    graphs: ['block'],
    prerequisites: ['tensors'],
    related: ['rmsnorm', 'attention-overview', 'swiglu', 'linear-maps'],
    sources: ['attention', 'residual', 'rms'],
    sections: [
      section(
        'updates',
        'Read each branch as an update',
        [
          'In this sequential pre-norm decoder, the attention sublayer reads a normalized version of X and produces an update added to the original X. The feed-forward sublayer then reads a normalized version of the updated stream Y and adds another update. The skip path carries the unnormalized stream around each sublayer.',
          'Every addition is element-wise at matching batch, token and feature coordinates. Attention can incorporate other visible positions before producing its update; the feed-forward transformation acts independently at each position using shared weights. Both return the residual width D so the additions are well-defined.',
        ],
        String.raw`Y=X+\operatorname{Attn}(\operatorname{RMSNorm}(X)),\qquad Z=Y+\operatorname{FFN}(\operatorname{RMSNorm}(Y))`,
      ),
      section('layers', 'Depth means successive transformations', [
        'Layer 1 receives the output of layer 0. In this profile, each layer has independent attention matrices, feed-forward matrices, normalization gains and K/V state. Expanding the model shows those distinct instances. The layer selector in a study view identifies which instance is being discussed; it does not make all layers share one tensor.',
        'A synthetic element example: a stream coordinate of 2.0 plus an attention update of −0.5 becomes 1.5. A later FFN update of 0.2 makes it 1.7. Addition preserves an identity path but does not guarantee that the input can be perfectly recovered from the output or that a coordinate keeps one meaning across layers.',
      ]),
      section('variants', 'Placement is part of the architecture', [
        'For a vector residual map y = x + f(x), the Jacobian is I + J_f(x), where the identity matrix comes from the bypass. This explains a direct derivative path without guaranteeing a nonvanishing total gradient: the learned branch can reinforce or cancel parts of that identity contribution. The scalar derivative in the deeper treatment is the one-coordinate version of this statement.',
        'Pre-norm, post-norm and parallel attention/FFN blocks have different equations and dependency graphs. The original Transformer paper is a foundation for attention and residual blocks; it is not a claim that its block equation is identical to this app’s RMSNorm/SwiGLU teaching design.',
        'Follow the residual line in the graph instead of assuming every wire carries a newly projected vector. In the microscope, compare an addition output to both its operands: neither branch alone explains the final hidden state.',
      ]),
    ],
  },
  {
    id: 'rmsnorm',
    title: 'RMSNorm, gain and epsilon',
    category: 'Foundations',
    summary: 'Normalize the scale across one token’s features without subtracting their mean.',
    aliases: [
      'normalization',
      'RMS',
      'LayerNorm',
      'variance',
      'mean',
      'epsilon',
      'gamma',
      'gain',
      'root mean square',
    ],
    kinds: ['norm1', 'norm2', 'finalnorm'],
    graphs: [],
    prerequisites: ['tensors'],
    related: ['residual', 'linear-maps', 'tensor-laboratory', 'numerical-precision'],
    sources: ['rms', 'layernorm'],
    sections: [
      section(
        'reduction',
        'Reduce the feature axis, then rescale each feature',
        [
          'For each batch item and position, square the D features, average them, add epsilon and take a square root. Divide each input feature by that common denominator and multiply by its learned feature gain g. The output keeps the input shape. Different tokens use different RMS denominators; the learned gain is shared across token positions.',
          'The reference numerical implementation uses epsilon = 10⁻⁵. Epsilon keeps the denominator nonzero for an all-zero row and affects small-magnitude inputs. Gain is a parameter; the RMS denominator is an activation statistic computed from this row.',
        ],
        String.raw`y_d=g_d\frac{x_d}{\sqrt{\frac{1}{D}\sum_{j=0}^{D-1}x_j^2+\epsilon}}`,
      ),
      section('example', 'Worked row · synthetic', [
        'Take x = [3,4], gain [1,1], and ignore epsilon only for this hand calculation. Mean square is (9+16)/2 = 12.5 and RMS is approximately 3.5355. The output is approximately [0.8485,1.1314]. Its mean is not zero. Real reference calculations include epsilon and Float32 output rounding.',
        'LayerNorm also subtracts the feature mean and normalizes using centered variance. RMSNorm does not. Normalization does not turn the vector into probabilities; negative inputs can remain negative, and output coordinates need not sum to one.',
      ]),
      section('placement', 'Why look at it here?', [
        'The pre-norm branches in this profile operate on a scale-normalized view while the identity path carries the residual stream. A final RMSNorm precedes the vocabulary projection. These are different parameter instances even when they apply the same formula.',
        'In the numerical microscope, inspect the sum of squares, denominator, gain and original coordinate. An observed change in one normalized feature can arise because another feature changed the shared denominator, even if the inspected input coordinate stayed fixed.',
      ]),
      section(
        'compare-layernorm',
        'LayerNorm versus RMSNorm: center first or preserve the mean?',
        [
          'For the same feature row, LayerNorm subtracts its mean and divides by the square root of centered variance plus epsilon. RMSNorm divides the uncentered input by the square root of mean square plus epsilon. In both cases epsilon is inside the square root in this implementation. Here D is the feature count, and variance uses the population denominator D. Both may apply learned gains; a LayerNorm bias is a separate parameter, set to zero in the comparison.',
          'Adding the same offset to every coordinate leaves the centered LayerNorm result unchanged in exact arithmetic. RMSNorm generally changes because its statistic includes that offset. For the constant row [c,c], LayerNorm with zero bias gives [0,0]; RMSNorm need not. Neither result is a probability distribution, and choosing one requires the model’s actual trained architecture.',
          'The expanded lab compares both operations on the same shifted row with shared scalar gain and epsilon. It isolates the centering difference, while the original RMSNorm controls retain their step-by-step calculation. It does not replace the running model’s normalization or make a speed benchmark.',
          'Normalization statistics describe an intermediate calculation, not a universal guarantee about the final output. Epsilon changes exact scale identities; feature-dependent gains can change scale and mean, and an additive bias can shift them further. For LayerNorm, the centered vector has zero mean before gain/bias; unequal gains need not preserve that mean afterwards.',
        ],
        String.raw`\mu=D^{-1}\sum_jx_j,\quad v=D^{-1}\sum_j(x_j-\mu)^2,\quad y_i=g_i\frac{x_i-\mu}{\sqrt{v+\epsilon}}+b_i`,
      ),
    ],
  },
  {
    id: 'swiglu',
    title: 'SwiGLU and position-wise feed-forward networks',
    category: 'Foundations',
    summary:
      'Follow two feature projections, a smooth gate, an element-wise product and a projection back to the residual stream.',
    aliases: [
      'FFN',
      'MLP',
      'feedforward',
      'feed-forward',
      'SiLU',
      'swish',
      'Hadamard',
      'gate',
      'up',
      'down',
      'intermediate width',
    ],
    kinds: ['swiglu', 'gate', 'up', 'silu', 'multiply', 'down'],
    graphs: ['feedforward'],
    prerequisites: ['linear-maps'],
    related: ['residual', 'rmsnorm', 'mixture-of-experts'],
    sources: ['swiglu'],
    sections: [
      section(
        'branches',
        'Two branches meet coordinate by coordinate',
        [
          'The normalized input X[B,T,D] feeds a gate matrix Wg[D,F] and an up matrix Wu[D,F]. The gate branch applies SiLU(z) = z·sigmoid(z); the result multiplies the up branch element by element. A down matrix Wd[F,D] returns to the residual width.',
          'Both branches are learned projections. The gate is not an attention probability, a choice of token position, or a hard expert selector. SiLU can be negative and is not bounded to [0,1]. Each product coordinate uses one gate coordinate and its corresponding up coordinate, with no reduction until the down projection.',
        ],
        String.raw`\operatorname{FFN}(X)=\bigl(\operatorname{SiLU}(XW_g)\odot XW_u\bigr)W_d,\qquad \operatorname{SiLU}(z)=\frac{z}{1+e^{-z}}`,
      ),
      section('example', 'One intermediate coordinate · synthetic', [
        'Suppose a gate projection coordinate is 2 and the matching up coordinate is −3. SiLU(2) ≈ 1.7616, so their product is approximately −5.2848. The down projection mixes this and the other F intermediate coordinates into each D-dimensional update feature.',
        'Changing F changes the sizes of all three matrices and the width of the intermediate activation. It does not change the number of token positions. “Position-wise” means that this sublayer does not directly mix different positions; it can still use context already carried in its input by earlier attention.',
      ]),
      section('reading', 'Read the fork and join', [
        'A useful distinction is coordinate-wise versus position-wise. The Hadamard gate is coordinate-wise: it combines matching intermediate features. The whole feed-forward sublayer is position-wise: its matrices mix features within each token row while sharing parameters across rows. Neither phrase means that each final output feature depends on only one input feature.',
        'Follow the fork after normalization and identify which branch receives the nonlinear activation. Then inspect the Hadamard-product connection and compare it with a matrix-product connection. The displayed graph makes those two kinds of multiplication explicit.',
        'A dense SwiGLU executes the same set of matrices for each position. Replacing it with mixture-of-experts introduces a router and selected expert paths; merely drawing extra branches does not implement token-specific routing. This app’s numerical foundations profile implements dense SwiGLU.',
      ]),
    ],
    experiment: {
      title: 'Follow a gate’s operands',
      steps: [
        'Open the tensor laboratory’s question about what a gate does.',
        'Select a product cell and follow its SiLU operand.',
        'Return and follow its up-projection operand, then inspect the down projection.',
      ],
      observation:
        'The product is local to an intermediate coordinate; the down projection performs the later sum over intermediate features.',
    },
  },
]
