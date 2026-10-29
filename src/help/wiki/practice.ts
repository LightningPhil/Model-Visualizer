import type { WikiArticle } from './types'
import { section } from './foundations'

export const practiceArticles: WikiArticle[] = [
  {
    id: 'logits',
    title: 'From hidden state to vocabulary logits',
    category: 'Output & training',
    summary:
      'Project the final feature vector into one score per output ID, then distinguish scores from probabilities.',
    aliases: [
      'output head',
      'unembedding',
      'vocabulary projection',
      'classifier',
      'logit',
      'distribution',
      'final norm',
    ],
    kinds: ['output', 'lmhead', 'sharedhead', 'head1', 'head2'],
    graphs: ['output'],
    prerequisites: ['linear-maps', 'embeddings'],
    related: ['softmax', 'sampling', 'training-objective'],
    sources: ['attention', 'stories'],
    sections: [
      section(
        'projection',
        'One score per vocabulary ID',
        [
          'After the final normalization, a hidden row h[D] multiplies an output matrix Wout[D,V] to produce z[V]. Each logit is a sum of feature/weight products for one candidate ID. Logits can be negative, unbounded and shifted by a common constant without changing their softmax probabilities. They do not sum to one.',
          'The projection is applied to the positions being studied. Ordinary next-token generation uses the last current position’s row to choose the following ID. Training can compare many positions’ rows with shifted targets. A high logit is a model preference among vocabulary IDs, not evidence that a natural-language assertion is true.',
        ],
        String.raw`z_v=\sum_{d=0}^{D-1}h_dW^{\mathrm{out}}_{d,v}`,
      ),
      section('tying', 'Check whether the output matrix is tied', [
        'The trained Stories260K profile ties its classifier to the embedding matrix’s transpose. The untrained laboratory has independent embedding and output matrices. This changes the parameter relationship, not the meaning of the vocabulary axis.',
        'The tokenizer vocabulary must match the checkpoint. A logit at ID 42 is meaningful only with that vocabulary’s piece mapping. The untrained lab deliberately uses opaque IDs and should not be read as a language generator.',
      ]),
      section('inspect', 'Read the entire selection chain', [
        'A synthetic two-candidate check: logits [2,−1] have a gap of 3, so at temperature 1 their odds ratio is exp(3) and the first probability is about 0.9526. Adding 100 to both logits leaves that result unchanged. One logit is not itself a probability or a unique log-odds value; differences between logits determine pairwise log odds under softmax.',
        'First inspect logits. Then inspect the configured temperature and probability row, and finally the selection method and chosen ID. The highest-probability ID need not be selected under categorical sampling. Greedy selection does not require computing softmax to find the argmax, although showing probabilities can aid inspection.',
        'Decoding joins the selected IDs’ token pieces into text. It does not rerun attention or verify factual correctness. The last confirmed output may not yet be present in K/V until it is consumed in another forward call.',
      ]),
    ],
  },
  {
    id: 'sampling',
    title: 'Greedy choice, sampling and temperature',
    category: 'Output & training',
    summary:
      'The selection rule is separate from the forward pass; a sampling seed does not change learned weights.',
    aliases: [
      'random',
      'seed',
      'temperature',
      'argmax',
      'categorical',
      'deterministic',
      'top-k',
      'top-p',
      'decode',
    ],
    kinds: ['sample', 'decode', 'next'],
    graphs: ['output'],
    prerequisites: ['logits', 'softmax'],
    related: ['prefill-decode', 'trained-model', 'speculative-decoding'],
    sources: ['attention', 'stories', 'speculative', 'topk', 'nucleus'],
    sections: [
      section('rules', 'Choose a maximum or draw from a distribution', [
        'Greedy chooses a highest-logit ID. The reference implementation breaks ties using the lowest ID. Categorical sampling draws an ID according to the probability row using a separate random stream. A lower-probability ID can therefore be a valid sampled output.',
        'A sampling seed makes draws reproducible for the same probability sequence and implementation. It does not reinitialize a trained model’s weights. In the untrained lab, the parameter seed is a different control: changing it constructs a different set of untrained parameters and can change every subsequent activation.',
      ]),
      section(
        'temperature',
        'Temperature changes relative gaps',
        [
          'For positive temperature τ, vocabulary probabilities use softmax(z/τ). Smaller τ sharpens differences; larger τ flattens them. In the limit τ→0 with a unique maximum, its probability approaches one. The τ = 0 formula itself is undefined, so greedy is treated as an explicit rule rather than a division by zero.',
          'Synthetic two-logit example: z = [0,1] gives the second candidate probability about 0.731 at τ = 1, 0.881 at τ = 0.5, and 0.622 at τ = 2. Positive temperature preserves logit ordering but changes the sampling distribution. In the greedy path, rescaling by a positive number does not change the unique maximum.',
        ],
        String.raw`p(v\mid\text{prefix})=\frac{\exp(z_v/\tau)}{\sum_u\exp(z_u/\tau)},\qquad \tau>0`,
      ),
      section('limits', 'Selection controls do not add knowledge', [
        'Temperature changes output variation, not the parameters, tokenizer, context capacity or truthfulness of the model. Top-k and top-p filtering are additional sampling techniques; the reference workbenches here expose greedy and seeded categorical selection rather than claiming every production sampler.',
        'The selected ID feeds back into the next call, so a changed draw can alter the entire later trajectory. For a clean experiment, compare the same prefix’s logits before following divergent generated continuations.',
      ]),
      section('categorical-draw', 'Turn a probability row into one draw', [
        'For synthetic probabilities [0.2,0.3,0.5], partition a unit interval into cumulative ranges [0,0.2), [0.2,0.5) and [0.5,1). A uniform draw of 0.6 selects candidate 2. This explains both why a nonmaximum can be selected and why the random seed matters only after the probability row is known. Implementations must handle normalization, boundary conventions and floating-point tails explicitly.',
        'Background filtering methods first choose a permitted candidate set, then renormalize: top-k retains a fixed number of highest-scoring candidates; nucleus/top-p retains a smallest score-sorted prefix reaching a chosen probability mass, under the implementation’s tie convention. The set in top-p can change size from one row to another. Temperature and filtering order are part of the sampling contract. These filters are explained here, not added as hidden workbench modes.',
      ]),
    ],
  },
  {
    id: 'training-objective',
    title: 'Next-token targets and cross-entropy',
    category: 'Output & training',
    summary:
      'A causal position predicts the following token; training changes parameters while inference reads them.',
    aliases: [
      'loss',
      'training',
      'teacher forcing',
      'labels',
      'shifted targets',
      'negative log likelihood',
      'gradient',
      'backpropagation',
    ],
    kinds: ['loss', 'shifted'],
    graphs: ['mtp'],
    prerequisites: ['logits', 'causal-mask'],
    related: ['mtp', 'sampling', 'evidence', 'learning-loop', 'post-training'],
    sources: ['attention', 'mtp', 'deepseek'],
    sections: [
      section(
        'shift',
        'Input and target are offset by one position',
        [
          'For a sequence of IDs x₀,x₁,x₂,…, the causal hidden state at position t uses input through xₜ and predicts target xₜ₊₁. Teacher forcing supplies the observed prefix during training instead of sampling the model’s own next choice for every position.',
          'For a one-hot target y, cross-entropy at a position is −log p(y|prefix), using the natural logarithm. A smaller probability assigned to the correct target yields a larger loss. Multiple positions’ losses can be aggregated according to the training design, with padding and other exclusions specified by that design.',
        ],
        String.raw`\mathcal L_t=-\log p(x_{t+1}\mid x_{\le t})`,
      ),
      section('example', 'Loss is about the target probability · synthetic', [
        'The logit gradient makes the objective concrete. With probabilities [0.25,0.75] and target ID 0, p minus the target indicator is [−0.75,+0.75]. A small gradient-descent step on these two logits alone raises the target score relative to its competitor. Updating shared model parameters requires backpropagating through the rest of the graph, so this local example is not a full training update.',
        'For N scored target tokens, mean negative log likelihood is measured in nats per token; perplexity is exp(mean loss). It is the reciprocal geometric mean of the assigned target probabilities, not an accuracy percentage. Compare perplexities only with compatible tokenization, data and target masking, since changing the units called tokens changes the metric.',
        'If the correct target receives probability 0.5, loss is approximately 0.6931 nats. At 0.1 it is approximately 2.3026. The selected greedy ID could stay the same while its loss changes because probabilities change; prediction accuracy and cross-entropy measure different things.',
        'Backpropagation differentiates the training objective and an optimizer updates parameters. None of this app’s inference workbenches trains a model or updates its learned weights. Recording an activation or changing a sampling seed is not a training step.',
      ]),
      section('separate', 'Keep objectives separate from inference state', [
        'An auxiliary MTP loss supervises farther-future targets during training. A KV cache carries inference activations across calls. Neither should be described as the other. A model can be trained with an auxiliary objective while using an inference procedure that ignores the auxiliary prediction path.',
        'The MTP lab aligns targets to explain the offsets. Its displayed target alignment is a teaching example rather than a gradient computation or a measured loss from a trained MTP checkpoint.',
      ]),
    ],
  },
  {
    id: 'mtp',
    title: 'Multi-token prediction: objectives and designs',
    category: 'Output & training',
    summary:
      'Auxiliary predictions of farther-future tokens can use different architectures; training and speculative inference remain separate.',
    aliases: [
      'MTP',
      'multi-token',
      'future depth',
      'auxiliary',
      'sequential MTP',
      'independent heads',
      'draft',
    ],
    kinds: [
      'mtp',
      'mainstate',
      'shifted',
      'combine',
      'mtpblock',
      'sharedhead',
      'loss',
      'combine1',
      'combine2',
      'mtp1',
      'mtp2',
      'head1',
      'head2',
    ],
    graphs: ['mtp'],
    prerequisites: ['training-objective', 'residual'],
    related: ['speculative-decoding', 'prefill-decode', 'evidence'],
    sources: ['mtp', 'deepseek'],
    sections: [
      section('objective', 'Predict beyond the immediate next token', [
        'The ordinary hidden state at t predicts target t+1. An MTP objective adds supervision for farther positions, such as t+2 and t+3. The exact branches, input embeddings, shared parameters and loss weights depend on the design. “MTP” names a family of ideas rather than one universal graph.',
        'Gloeckle et al. study independent future-prediction heads. The DeepSeek-V3 report describes sequential MTP modules that combine a preceding representation with shifted token embeddings and use a shared output head. The authored MTP study here explains the sequential design and cites the distinction.',
      ]),
      section('training', 'Observed shifted tokens versus generated proposals', [
        'At base position t, a sequential first auxiliary module can consume the observed ID at t+1 and predict t+2. An independent head reading only the base representation can instead predict t+2 from input through t. These are different dependency graphs and conditional prediction tasks. Reading an observed t+1 is not leakage into a t+2 target, but reading the target itself through an unmasked path would be. Explicit target offsets and input masks are essential.',
        'During teacher-forced training, the shifted embedding can come from the observed sequence. During speculative generation, the corresponding input may be a model proposal. Replacing observed inputs with proposals changes the inference procedure and requires explicit verification rules.',
        'The MTP alignment lab shows which future target belongs to each depth. It does not execute training or prove that a particular prediction is accurate. The graphical simulator uses scripted proposals and matches; it does not run learned MTP weights. Numerical MTP is not implemented in either numerical profile.',
      ]),
      section('inference', 'Several predictions are not automatically committed text', [
        'Auxiliary proposals can be wrong. A target-model verification process must establish which tokens can be committed, and the cache must retain a valid state for the accepted prefix. A single mismatch invalidates later draft tokens conditioned on the rejected history.',
        'Speedup is an empirical property depending on draft cost, acceptance, verification batching and the runtime. Playback animation and a count of proposed tokens do not measure it. Use the generation study for the algorithmic transaction and the primary papers for their own measured systems.',
      ]),
    ],
  },
  {
    id: 'speculative-decoding',
    title: 'Draft, verify, accept and roll back',
    category: 'State & execution',
    summary:
      'Understand why a verified prefix can be committed and why rejected trial cache state must be discarded safely.',
    aliases: [
      'speculative',
      'verification',
      'accept prefix',
      'reject suffix',
      'rollback',
      'checkpoint',
      'correction',
      'bonus token',
    ],
    kinds: [
      'verify',
      'compare',
      'reject',
      'commit',
      'feedback',
      'embed1',
      'embed2',
      'mtp1',
      'mtp2',
    ],
    graphs: ['generation'],
    prerequisites: ['prefill-decode', 'kv-cache'],
    related: ['mtp', 'cache-retention', 'sampling'],
    sources: ['speculative', 'deepseek'],
    sections: [
      section('transaction', 'A draft is a proposal', [
        'A draft path proposes multiple IDs. A main decoder verifies their causal predictions in a batch. Under the graphical simulator’s greedy match rule, accept the consecutive matching prefix, then stop at the first mismatch. Later matches do not rescue the suffix because they were proposed under a history that is no longer accepted.',
        'Synthetic example: draft [a,b,c], main greedy agreement on a and disagreement on b. Commit a and the main correction for b; discard draft b and c and their invalid trial state. The chosen correction still needs to be consumed to acquire its own K/V. Fully matching scenarios can produce an additional main-model token.',
      ]),
      section('rollback', 'Keep a recoverable pre-round state', [
        'Verification writes trial K/V for proposed positions. At rejection, preserve state for the valid accepted prefix and remove invalid suffix state. Sliding or sparse retention complicates this: trial writes must not irreversibly evict checkpoint values still needed after rollback.',
        'The simulator pins required pre-round/trial positions before committing retention. A production runtime must provide an equivalent guarantee with its actual storage scheme. A ring buffer that overwrites old slots without snapshots or recoverable copies cannot simply “undo” those values from position counts alone.',
      ]),
      section('stochastic', 'Greedy matching is not the full stochastic algorithm', [
        'Leviathan et al. describe probability-ratio acceptance and a corrected rejection distribution that preserve the target sampling distribution. Comparing a draft ID with one greedy target ID is not a substitute for that rule under categorical sampling.',
        'This app’s graphical scenarios teach causal verification, accepted-prefix handling and state rollback using scripted choices. They make no claim to execute the paper’s stochastic algorithm, numerical MTP or a measured speedup. Read the timeline’s provenance before interpreting it as a trained run.',
      ]),
    ],
    experiment: {
      title: 'Study a rejection transaction',
      steps: [
        'Open Run simulation and choose an MTP rejection scenario.',
        'Step through checkpoint, proposals, verification and the first mismatch.',
        'Compare pinned state, retained absolute positions and committed output before and after rollback.',
      ],
      observation:
        'Committed text and cached input positions advance at different moments. Rejected suffix state must disappear without losing the valid prefix.',
    },
  },
  {
    id: 'conditional-memory',
    title: 'N-gram lookup and conditional memory',
    category: 'Architecture extensions',
    summary:
      'A hashed token-window lookup is a distinct memory mechanism, with explicit collision and fusion semantics.',
    aliases: [
      'Engram',
      'n-gram',
      'hash',
      'lookup',
      'address',
      'retrieval',
      'fusion',
      'memory table',
      'collision',
    ],
    kinds: ['memory', 'ngrams', 'hash', 'retrieve', 'fusion'],
    graphs: ['memory'],
    prerequisites: ['embeddings', 'residual'],
    related: ['kv-cache', 'swiglu', 'evidence'],
    sources: ['engram'],
    sections: [
      section('lookup', 'Token windows determine addresses', [
        'A causal n-gram is a window of n IDs ending at the current position. A hash maps that tuple to a bounded address. Looking up an embedding at that address retrieves a vector associated with the slot. This is not attention over all previous activations and is not an external search request.',
        'Different n-grams can collide at the same bounded address. A hashed slot is therefore not a guaranteed unique symbolic fact. A full architecture specifies how to combine multiple orders/heads, resolve or tolerate collisions, and fuse retrieved features with contextual hidden states.',
      ]),
      section('fusion', 'Retrieval and contextual use are separate steps', [
        'In the cited Engram design, canonicalized token-window IDs select embeddings through several hash heads and n-gram orders; their retrieved vectors are concatenated. The current hidden representation then provides a query for a scalar gate over projected retrieved content. A short depthwise causal convolution adds further processing. These steps explain why one displayed hash address is only the beginning of that published module.',
        'An address can be deterministic without the retrieved contribution being constant across contexts: the same selected parameter vector can be gated differently by different hidden states. Learning can update selected embeddings and projection parameters even though the integer hash itself is not a differentiable language-model prediction. The workbench illustrates addressing only; it does not execute this full trained retrieval/gating path.',
        'The Engram paper motivates conditional memory using token patterns and context-aware fusion. The graph’s lookup path and fusion path are distinct so you can ask what is retrieved and how the current hidden state controls its influence.',
        'The teaching memory lab shows explicit n = 2/3/4 windows and illustrative hash addresses. Its constants and outputs are examples, not trained Engram activations. It does not implement the full published architecture, and this path is inactive in the graphical simulator and absent from numerical execution.',
      ]),
      section('distinction', 'Three uses of the word memory', [
        'Parameter memory is fixed learned model storage during inference. A KV cache holds input-dependent attention state across calls. An n-gram lookup module retrieves parameterized features using token-window addresses. They have different indexing, lifetimes and update rules.',
        'Notebook text and browser storage are study persistence, not model memory mechanisms. When reading an unfamiliar architecture, identify which state is learned, which depends on the current sequence, and which is merely application data.',
      ]),
    ],
  },
  {
    id: 'mixture-of-experts',
    title: 'Mixture-of-experts and routed computation',
    category: 'Architecture extensions',
    summary:
      'Expert routing selects token-dependent computation; the current dense numerical profiles do not execute it.',
    aliases: [
      'MoE',
      'router',
      'top-k experts',
      'dispatch',
      'combine',
      'activated parameters',
      'sparse experts',
    ],
    kinds: [],
    graphs: [],
    prerequisites: ['swiglu', 'linear-maps'],
    related: ['conditional-memory', 'storage-kernels', 'mla', 'learning-loop'],
    sources: ['deepseek'],
    sections: [
      section('route', 'A router makes a per-token choice', [
        'A mixture-of-experts feed-forward layer uses routing scores to choose experts for each token. A full description needs the score function, selection rule, routing weights, dispatch, expert transformations and weighted combination. Some designs add experts that are always shared.',
        'Multiple drawn FFN branches alone do not demonstrate MoE. The crucial behavior is which branches execute for a particular token, with what weights and what capacity/dispatch constraints. Different tokens can choose different paths.',
      ]),
      section('accounting', 'Total and activated parameters differ', [
        'For a hypothetical bias-free layer with E dense SwiGLU experts of common widths D and F, expert matrices contain E·3DF parameters. Selecting k experts uses k·3DF expert parameters for that token, while the router and any shared experts add their own work and weights. This accounting is about which stored coefficients participate, not a claim that parameter count equals FLOPs or latency. Imbalanced routing can leave some devices waiting while others receive too many tokens.',
        'Total expert parameters count all stored expert matrices. Activated parameters count the subset used for a token, together with any shared computation under the chosen definition. A low activated count does not mean the inactive weights disappear from storage.',
        'Runtime cost also depends on routing balance, communication, batch structure and implementation. Neither a parameter count nor an architecture diagram alone is a measured latency estimate. The DeepSeek-V3 report is a primary example of a specific routed/shared expert design, not a universal specification.',
      ]),
      section(
        'dense-routed-comparison',
        'Compare a dense path with routed experts explicitly',
        [
          'A dense feed-forward sublayer applies its specified transformation to every token. A routed expert layer computes router scores, selects expert paths, dispatches inputs, and combines selected outputs. Keeping a large collection of experts does not mean every expert executes for every token. Router work, shared experts and communication remain part of the complete cost.',
          'The new lab uses three fictional scalar affine experts: x, −x+1, and 2x−1. It selects the highest k router scores, renormalizes exponentials over that selected set, and weights only those expert outputs. The displayed full set of expert formulas is explanatory availability, not a claim that a sparse runtime evaluates every branch. Ties select lower expert IDs deterministically.',
          'With two scalar parameters per toy expert, stored expert parameters are six and activated expert parameters are twice k; these counts explicitly exclude the router. A separate dense reference evaluates only 2x−1 with two parameters. The comparison demonstrates routing and accounting, not equal capacity, training, output quality or latency. DeepSeek uses architecture-specific routing and shared experts beyond this toy rule.',
        ],
        String.raw`y=\sum_{i\in\operatorname{TopK}(s)}\frac{e^{s_i}}{\sum_{j\in\operatorname{TopK}(s)}e^{s_j}}E_i(x)`,
      ),
      section('status', 'Scope of this application', [
        'The foundations and both numerical model profiles use dense feed-forward computation; they do not execute learned token-specific expert routing. The standalone scalar wiki lab does execute its explicitly defined top-k arithmetic. That toy should be distinguished from a trained MoE runtime with a learned router, full expert matrices and dispatch constraints.',
        'Adding it would require renderer-independent routing semantics, expert-specific tensor provenance, explicit total/activated accounting, and model-specific sources. Imported descriptions can still be inspected as authored graph content without implying that the local runtime executes them.',
      ]),
    ],
  },
  {
    id: 'mla',
    title: 'MLA and alternative attention state',
    category: 'Architecture extensions',
    summary:
      'Multi-head latent attention needs its own cache representation; reducing KV heads is not a universal substitute.',
    aliases: [
      'multi-head latent attention',
      'latent cache',
      'compressed KV',
      'DeepSeek',
      'linear attention',
      'recurrent state',
    ],
    kinds: [],
    graphs: [],
    prerequisites: ['head-sharing', 'kv-cache'],
    related: [
      'cache-bytes',
      'storage-kernels',
      'mixture-of-experts',
      'attention-comparison',
      'linear-attention',
    ],
    sources: ['deepseek'],
    sections: [
      section('latent', 'Compressing state changes its representation', [
        'DeepSeek’s MLA uses low-rank latent compression of key/value information with a separate positional component and architecture-specific projection/absorption choices. Its persistent state cannot be accurately described by simply changing Hkv in the dense K/V formula.',
        'GQA shares explicit key/value heads among queries. MLA changes what information is stored and how it is used to recover or algebraically combine the required attention calculations. Both can reduce decoding-state costs, but they do so through different mathematical designs.',
      ]),
      section('accounting', 'Start from the actual stored axes', [
        'To estimate memory for a new architecture, identify each persistent tensor, its per-position or recurrent dimensions, dtype and positional metadata. Sum their payloads. Do not reuse 2×L×B×Hkv×R×Dh unless those axes actually describe the state.',
        'Likewise, a recurrent or linear-attention state may summarize a prefix rather than retain separate K/V entries for each position. Such a system needs an explicit update/read equation and a different visualization; it is not a dense cache with a relabeled sequence axis.',
      ]),
      section(
        'latent-algebra',
        'See what is compressed and what can be rearranged',
        [
          'Use this app’s row-vector convention for the content part of one head. A latent row c of width dc produces key content cUk and value content cUv, where Uk and Uv have shapes [dc,Dk] and [dc,Dv]. A content query q has width Dk. Its key score can be evaluated as (qUkᵀ)cᵀ, while a probability-weighted value sum can be formed in latent coordinates before applying Uv. These are matrix-associativity identities, not an approximate recovery of an arbitrary independently trained MHA cache.',
          'The decoupled positional key is an additional stored component in the cited MLA architecture. Its query/key score contribution and rotation conventions must be included as well; a position-dependent rotation cannot simply be absorbed into one fixed content matrix. This is why MLA accounting includes the joint latent and positional state instead of just changing the dense KV-head count.',
        ],
        String.raw`q(cU_K)^\top=(qU_K^\top)c^\top,\qquad \sum_u A_u(c_uU_V)=\left(\sum_u A_uc_u\right)U_V`,
      ),
      section('status', 'An extension boundary, not a hidden mode', [
        'This workbench’s executable attention families are dense MHA/GQA/MQA with the documented retention policies. MLA is a research extension here. Its wiki entry is background for understanding distinctions, not a claim of implemented latent-state inference.',
        'A faithful extension would introduce a configuration profile, actual persistent-state contract, tensor equations, cache accounting and matching numerical validation. The DeepSeek report provides a concrete primary architecture to start from.',
      ]),
    ],
  },
  {
    id: 'tensor-laboratory',
    title: 'Learn with the tensor microscope',
    category: 'Using the workbench',
    summary:
      'Inspect actual arrays, select a scalar, follow its operands and compare controlled experiments.',
    aliases: [
      'numerical',
      'lab',
      'microscope',
      'cell',
      'operand',
      'provenance',
      'baseline',
      'transpose',
      'parameter library',
      'trace',
    ],
    kinds: [],
    graphs: [],
    prerequisites: ['evidence', 'tensors'],
    related: ['linear-maps', 'attention-scores', 'rmsnorm', 'study-method', 'trained-model'],
    sources: [],
    sections: [
      section('profile', 'The separate reference model', [
        'The untrained laboratory uses B = 1, two layers, D = 16, four query heads, Dh = 4, F = 32 and V = 256. GQA has two KV heads, MQA one and MHA four. Parameters come from seeded initialization, not training. Inputs and outputs are opaque IDs; this profile is independent of the architecture builder’s current dimensions.',
        'It executes the causal pre-RMSNorm/RoPE/SwiGLU design with no biases, dropout, weight updates, numerical MTP or n-gram memory. The graph comes from the authored foundations templates; importing another architecture does not silently change the numerical runtime.',
      ]),
      section('cell', 'Follow a single calculation', [
        'Distinguish a scalar recipe from a sentence about behavior. A reproducible score observation needs the model/seed or checkpoint, call, layer, query head, query absolute position and key absolute position. Following its operands should recover the contracted feature products at those coordinates. If you transpose the display, those semantic coordinates remain the identity of the observation.',
        'Choose a question or operation, run to it if necessary, then select a tensor and cell. The calculation panel explains its recipe: product summands, RMS statistics, rotary coefficients, visibility decisions, exponentials or element-wise operands. Follow an operand to inspect its original tensor coordinate, then use Back through the calculation.',
        'Axis selectors choose slices. Transpose view changes display orientation, not the stored axis order or execution. Check the selected layer, call, head and absolute position before comparing cells. A component that has not executed may offer Run to this operation; an older available binding is explicitly labelled with its call.',
      ]),
      section('experiment', 'Change one thing and preserve the comparison', [
        'Pin baseline saves settings so the baseline can be recomputed. Change one setting, then inspect Δ baseline only where shape, axes, dtype and absolute position maps match. Masked-score comparisons additionally require compatible visibility. If masks differ, finite attention probabilities are often a better comparison than subtracting scores containing −∞.',
        'All parameter tensors opens matrices and gain vectors with counts and Float32 payload bytes. Parameters exist from the beginning; activations appear after their recorded operation. Export trace saves actual arrays and provenance separately from the workspace. Playback is inspection of a precomputed capture, not live training.',
      ]),
      section('saving', 'Keep the study separate from the architecture', [
        'The laboratory saves settings, its own layouts, bookmarks, notes and optional baseline settings in a versioned workspace extension. It does not overwrite the architecture notebook or main model configuration. Export model carries the study settings and notes; Export trace carries calculated arrays.',
        'Changing setup calculates a fresh trace. Optional Follow flow moves the camera during playback; otherwise it stays where you put it. Opening contextual wiki help pauses playback so you can read and return to the same study.',
      ]),
    ],
    experiment: {
      title: 'Explain a value without guessing',
      steps: [
        'Choose the attention-score question and select a result cell.',
        'Follow one query operand back through RoPE and its projection.',
        'Return through the calculation and inspect the matching key operand.',
        'Bookmark the operation and write down which axis was summed.',
      ],
      observation:
        'A complete arithmetic explanation names operands, coordinates and operation. It is more specific than saying that the model “focused on” a token.',
    },
  },
  {
    id: 'trained-model',
    title: 'Study the local trained checkpoint',
    category: 'Using the workbench',
    summary:
      'Load the matching tokenizer and pinned Stories260K weights, explicitly generate, and inspect learned activations.',
    aliases: [
      'Stories260K',
      'llama2.c',
      'checkpoint',
      'SHA-256',
      'trained weights',
      'load',
      'generate',
      'completion',
    ],
    kinds: [],
    graphs: [],
    prerequisites: ['evidence', 'tokenization'],
    related: ['tensor-laboratory', 'sampling', 'embeddings', 'attention-interpretation'],
    sources: ['stories'],
    sections: [
      section('identity', 'A pinned model, not arbitrary checkpoint execution', [
        'The bundled Stories260K profile has five layers, D = 64, eight query heads, four KV heads and a 512-token vocabulary. Its embedding/output classifier weights are tied. The manifest pins the checkpoint revision, tokenizer and asset hashes, which the local loader verifies before running.',
        'Open Local trained model, explicitly load the assets, then generate a bounded capture. Weights and prompt processing stay in the browser worker. The viewer supports up to 48 prompt tokens and 16 new IDs for inspection; this bounded capture should not be confused with an arbitrary model’s context limit.',
      ]),
      section('inspect', 'Actual learned arrays with reference arithmetic', [
        'Start with a mechanism question that this capture can answer: whether a masked cell is excluded, how a selected logit was calculated, or whether a cache entry matches its creation call. A fluent completion is not a test of every internal operation, and an unusual completion is not by itself evidence of arithmetic failure. Comparing this checkpoint with an untrained model of another shape is a comparison of different models, not a controlled removal of training alone.',
        'The shared numerical workbench can inspect real parameter values, per-head attention and activations from this trained profile. Its traces identify the checkpoint and arithmetic provenance; they do not substitute seeded untrained tensors. The small base completion model can generate weak or odd text and is not an instruction-following chat assistant.',
        'Changing the sampling seed changes draws, not weights. Changing the prompt changes tokenization, activations and K/V. The architecture builder’s width/head controls describe the teaching document and do not resize this fixed trained checkpoint.',
      ]),
      section('persistence', 'Reopening does not run generation', [
        'Its separate study extension saves prompt, sampling settings, layouts, bookmarks and notes. Importing a workspace preserves those settings without loading or running a model. Each new opening needs visible load/generate actions. Exported traces contain arrays and model identity separately.',
        'Use a short prompt and a small output budget when learning an operation. Compare early layers and later layers, then follow a selected output logit back through the actual hidden state. Treat the observations as evidence about this checkpoint and run.',
      ]),
    ],
  },
  {
    id: 'study-method',
    title: 'A practical way to investigate a model',
    category: 'Using the workbench',
    summary:
      'Move from a concrete question to a prediction, an observable calculation and a saved explanation.',
    aliases: [
      'learn',
      'lesson',
      'experiment',
      'how to',
      'notebook',
      'questions',
      'compare',
      'learning path',
    ],
    kinds: [],
    graphs: [],
    prerequisites: [],
    related: ['reading-the-model', 'tensor-laboratory', 'attention-interpretation', 'workspace'],
    sources: [],
    sections: [
      section('question', 'Make the question small enough to answer', [
        'Write the expected observation and what would contradict it before changing a control. “Future-key probabilities are excluded under this mask” is an implementation invariant. “This head consistently tracks a linguistic feature” is an empirical hypothesis needing many varied inputs and controls. Both are useful, but they require different tests and justify different conclusions.',
        'Start with one observable question: “Which axis disappears in QKᵀ?”, “Why is this future cell zero?”, “Which K/V head does Q3 read?”, or “Why can confirmed output be uncached?” Select the relevant component or connection and open Explain this from the inspector.',
        'Read the authored description first, then the linked concept. Predict a shape, visibility rule or arithmetic result before opening the microscope. A prediction gives you something specific to verify rather than just watching an animation.',
      ]),
      section('evidence', 'Choose evidence that can answer it', [
        'Use architecture mode for structure, the synthetic lab for a labelled illustration, untrained numerical execution for exact reference arithmetic, and the trained checkpoint for actual learned behavior. Keep those kinds of evidence separate in your notes.',
        'Change one variable at a time where possible. If changing head sharing reconstructs the untrained experiment with differently shaped parameters, do not describe the result as a pure same-weight intervention on a trained model. Check position alignment before interpreting baseline deltas.',
      ]),
      section('write', 'Record the mechanism and its limit', [
        'A useful note names the model/profile, decoder call, layer/head, input setting, tensor coordinates, operation and observation. Add a source link for a general mathematical claim. Distinguish “this coordinate changed” from “this mechanism caused the final sentence.”',
        'Return to the graph and explain the path in your own words. Use prerequisites when terminology is missing, related articles when two mechanisms are being confused, and primary references for architecture details beyond the supported profile.',
      ]),
    ],
    experiment: {
      title: 'Three questions to begin with',
      steps: [
        'Explain why query 0 cannot see key 1 during causal prefill.',
        'Explain why two query heads sharing K/V can have different attention rows.',
        'Explain why a cache estimate uses Hkv rather than Hq.',
      ],
      observation:
        'A strong answer connects the dependency rule, the equation and the named tensor axes. The wiki’s attention and state articles provide the necessary prerequisites.',
    },
  },
  {
    id: 'workspace',
    title: 'Navigation, notes and portable model documents',
    category: 'Using the workbench',
    summary:
      'Keep layouts and notebooks, understand import validation, and distinguish a model document from a numerical trace.',
    aliases: [
      'help',
      'wiki',
      'field guide',
      'save',
      'export',
      'import',
      'JSON',
      'schema',
      'format',
      'search',
      'layout',
      'keyboard',
      'theme',
    ],
    kinds: [],
    graphs: [],
    prerequisites: [],
    related: ['study-method', 'evidence', 'tensor-laboratory'],
    sources: [],
    sections: [
      section('navigate', 'Explore without losing your view', [
        'Click a component for its authored description, tensors, equations and citations. Click a connection label for its operation and carried/result shape. Explore inside opens a child study graph. Press / outside a text field to search components across graphs. Each graph remembers its own layout and viewport.',
        'The wiki’s search covers concept articles and authored components and connections, including imported document text. Explain this opens the current selection with related concepts. Back/forward buttons retrace wiki reading history. View on canvas closes help and selects the referenced element when that graph is available in the underlying workbench.',
      ]),
      section('layout', 'Layout tools change the view', [
        'Drag nodes, pan, zoom and fit. Auto-layout replaces the active graph’s arrangement and routes all its connections; Undo layout restores the previous arrangement while that graph stays open. Direction switches horizontal or vertical flow. A component’s position does not change its mathematics.',
        'On narrow screens use navigation and inspector controls to reveal the respective panels. Dark/light themes share the same content, and status text, icons and shapes reinforce color. Tooltip settings adjust delayed control explanations; wiki reading does not depend on hovering.',
      ]),
      section('document', 'What a workspace contains', [
        'The versioned JSON contains model identity, documentation, configuration, tokenizer description, authored graphs, equations, source references, layouts, theme, prompt and notebook. Namespaced extensions carry independent numerical/trained study settings and supported metadata. The authored example JSON is the semantic source; expanded graphs are derived from its recipes rather than saved as duplicate model definitions.',
        'Browser-local saving is specific to the origin and device. Export model produces a portable copy. Open validates schema, references, dimensions and supported profile rules before replacing the workspace; invalid files leave the current document intact. Validation does not prove arbitrary imported equations mathematically correct.',
      ]),
      section('traces', 'A model document is not a tensor capture', [
        'A practical recovery check is to export a workspace, reopen it, and verify a saved graph position and notebook entry before explicitly rerunning a numerical study. Export the trace separately if the actual arrays must be retained. Wiki bookmarks and reading positions are device preferences, so they are not a substitute for the portable model/study export or the numerical capture.',
        'Large numerical arrays belong in Export trace, alongside their dtype, coordinates, position maps and provenance. Workspace exports preserve study settings so runs can be reproduced with the supported profile. Numerical trace import is not implemented; reopening a study uses the workspace format.',
        'Use a namespaced extensions object for custom metadata. Breaking document meanings require versioning and migration that preserves IDs, notes and layouts. Reading wiki help does not alter any of these document fields.',
      ]),
    ],
  },
]
