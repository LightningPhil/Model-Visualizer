import { detail, introduction, type TopicPerspectives } from './perspective-types'

export const practicePerspectives: Record<string, TopicPerspectives> = {
  logits: {
    introduction: introduction(
      'The output head turns the final hidden representation into preferences over vocabulary IDs. Those preferences are called logits. A later normalization or selection rule determines which ID is chosen; the output head itself does not decide what the text means or whether it is true.',
      'Picture ranking possible next pieces of a message. The ranking comes from the current internal representation and the model’s learned output map. The candidates are token pieces rather than necessarily whole words, and the ranking is not a list of verified factual claims.',
      [
        'Finish transforming the current hidden state and apply the final normalization.',
        'Compare its features with the vocabulary projection’s parameters.',
        'Inspect the resulting preference for every candidate ID.',
        'Apply the configured selection rule and decode the selected piece with the matching tokenizer.',
      ],
      'Raw preferences are not probabilities. The output matrix may share parameters with the input embedding, but that sharing is explicit and model-specific. A chosen token’s K/V only exists after a later call processes it as input.',
    ),
    technical: [
      detail(
        'technical-logit-sensitivity',
        'Follow a vocabulary score back to hidden features',
        [
          'For a fixed output matrix, each vocabulary logit is a linear functional of the final normalized hidden row. A change to one hidden coordinate contributes to every vocabulary score according to that coordinate’s output weights.',
          'The difference between two logits is especially informative for selection: common shifts cancel under softmax. Nevertheless, a single coordinate contribution is only a local arithmetic decomposition; upstream normalization and nonlinear layers determine how an input intervention changes the hidden row.',
        ],
        String.raw`z_v=h^\top w_v,\quad \Delta z_v=\Delta h^\top w_v,\quad z_i-z_j=h^\top(w_i-w_j)`,
      ),
    ],
  },
  sampling: {
    introduction: introduction(
      'After the model assigns preferences, a selection rule chooses a next token. Greedy selection takes a maximum. Sampling draws according to a distribution, so less-preferred candidates can still be chosen. Temperature changes how concentrated that distribution is.',
      'Think of either always choosing the leading candidate or drawing from a weighted collection of candidates. Sharper weighting makes leading candidates more dominant; flatter weighting spreads chances more widely. This picture explains selection behavior, not the model’s knowledge or the correctness of the output.',
      [
        'Keep the input prefix and its calculated logits fixed while studying a selection rule.',
        'Choose whether to take a maximum or draw from the distribution.',
        'Change temperature to compare concentration without changing those logits.',
        'Remember that the chosen ID becomes part of the next call’s input, so later trajectories can diverge.',
      ],
      'A sampling seed controls draws, while an untrained parameter seed controls weights. Temperature cannot add knowledge or verify a statement. The app exposes specific supported rules rather than implying that every production filtering algorithm is implemented.',
    ),
    technical: [
      detail(
        'technical-temperature-odds',
        'Temperature acts on log-odds',
        [
          'For two candidates with finite logits and positive temperature, their probability ratio depends exponentially on their logit gap divided by temperature. A common logit shift cancels, and positive temperature preserves ordering. Categorical outcomes still depend on a draw.',
          'As temperature approaches zero, a unique maximum takes all probability in the limit. Directly substituting zero is undefined, so greedy is an explicit selection rule. The reference sampler normalizes the stored probability row by its actual finite-precision sum before drawing.',
        ],
        String.raw`\log\frac{p_i}{p_j}=\frac{z_i-z_j}{\tau},\qquad \frac{\partial\log(p_i/p_j)}{\partial\tau}=-\frac{z_i-z_j}{\tau^2}`,
      ),
    ],
  },
  'training-objective': {
    introduction: introduction(
      'During next-token training, a known prefix is used to predict the observed token that follows it. The training objective penalizes assigning too little probability to that target. An optimizer can use the resulting derivatives to adjust parameters; the inference workbenches here do not perform those updates.',
      'Imagine repeatedly comparing a proposed continuation with an observed continuation and adjusting a reusable prediction procedure. Training uses the observed sequence as context at each position. This is different from generation, where the model’s own chosen outputs become later inputs.',
      [
        'Align each current input position with the following observed target.',
        'Calculate the vocabulary distribution from the causal hidden state.',
        'Measure the penalty attached to the target’s assigned probability.',
        'During actual training, differentiate and update parameters according to the optimizer and training design.',
      ],
      'Selecting the correct maximum and assigning a good probability are different criteria. An auxiliary future-token objective is also different from inference cache state. This app explains target alignment and forward arithmetic without claiming to run a training loop.',
    ),
    technical: [
      detail(
        'technical-cross-entropy-gradient',
        'Derive the logit gradient for a one-hot target',
        [
          'Substitute the softmax probability into negative log likelihood. Differentiating gives the predicted probability minus the target indicator for each logit. This is a local derivative of the objective with respect to logits, not a full parameter gradient or a training implementation.',
          'The full gradient must follow the entire forward graph, including shared parameters and normalization. Loss aggregation also specifies valid target positions and exclusions. A causal mask alone does not fix label alignment or padding conventions.',
        ],
        String.raw`\mathcal L=-z_y+\log\sum_v e^{z_v},\qquad \frac{\partial\mathcal L}{\partial z_v}=p_v-\mathbf 1[v=y]`,
      ),
    ],
  },
  mtp: {
    introduction: introduction(
      'Multi-token prediction adds training signals for farther-future tokens, rather than supervising only the immediate next token. Different designs build those extra predictions in different ways. They can provide useful auxiliary representations or proposals, but the phrase does not describe one universal architecture.',
      'Picture practicing not just the next move in a sequence but several later moves too. In one design the extra predictions branch independently; in another they form a sequence of modules that incorporate shifted token embeddings. This is about supervised predictions, not permission to commit several unverified tokens during inference.',
      [
        'Identify the target offset assigned to each prediction depth.',
        'Check whether the modules are independent or sequential and which parameters they share.',
        'Separate observed shifted inputs during training from proposed inputs during generation.',
        'Use explicit verification and state rules if auxiliary predictions become inference proposals.',
      ],
      'The graphical MTP scenarios are scripted teaching examples. The local numerical profiles do not execute learned MTP modules. Training with an auxiliary objective does not by itself specify a correct stochastic sampling or rollback algorithm.',
    ),
    technical: [
      detail(
        'technical-mtp-alignment',
        'Specify offsets, inputs and sharing before writing a loss',
        [
          'At main position t, the ordinary target is t+1; an additional depth k predicts t+k+1 under the chosen convention. A training design must state the representation supplied to that depth, any shifted observed embedding, its decoder transformation, output-head sharing and loss weight.',
          'In the displayed objective, K is the number of auxiliary depths, k starts at one, lambda_k is that depth’s loss weight, and L_future,k aggregates its valid shifted targets. The formula is a schematic weighted training objective; it does not imply that the heads share inputs, parameters, target masks or conditional distributions across different MTP designs.',
          'Independent-head MTP and DeepSeek’s sequential modules therefore require different dependency graphs even if both have farther-future targets. At inference, substituting generated proposals for observed shifted embeddings changes the inputs and needs an explicit verification procedure. Numerical MTP remains an extension boundary here.',
        ],
        String.raw`\mathcal L_{\rm total}=\mathcal L_{\rm next}+\sum_{k=1}^{K}\lambda_k\mathcal L_{{\rm future},k}`,
      ),
    ],
  },
  'speculative-decoding': {
    introduction: introduction(
      'Speculative generation separates proposing text from verifying it. A draft path suggests several tokens, and a main model checks them. Only a valid prefix is committed; after a mismatch, the later draft history cannot simply be kept.',
      'Suppose a draft proposes token IDs A, B and C. A verifier accepts A but chooses D in place of B. C was proposed after B, so it belongs to a history that is no longer accepted; a later apparent match cannot make that old suffix valid. This concrete prefix rule is separate from how greedy or stochastic verification decides whether a proposal passes.',
      [
        'Preserve a recoverable state before the trial round.',
        'Produce a sequence of proposals and verify their causal predictions.',
        'Accept the valid prefix and stop at the first rejection under the specified rule.',
        'Discard invalid suffix state, keep the valid prefix, and process any selected correction in a later call.',
      ],
      'Greedy token matching is not the full distribution-preserving stochastic algorithm. Rolling storage alone also cannot recover values it has overwritten. The app’s graphical scenarios teach these state transitions without claiming numerical MTP or a measured speedup.',
    ),
    technical: [
      detail(
        'technical-stochastic-acceptance',
        'Greedy comparison and stochastic acceptance are different algorithms',
        [
          'In the speculative sampling paper, a draft token sampled from q can be accepted using a target-to-draft probability ratio, followed by an adjusted distribution upon rejection. This is what supports preservation of the target distribution; equality with a greedy ID is not an equivalent stochastic rule.',
          'Here p and q are normalized target and draft distributions at the same accepted prefix, after the specified sampling transformations. For a rejected proposal, draw the replacement from r(x) = max(p(x)−q(x),0) divided by the sum of those positive differences. The accepted mass for x is min(p(x),q(x)); rejection contributes the missing mass p(x)−min(p(x),q(x)), recovering p(x). If p=q, rejection has zero probability and the residual distribution is never used.',
          'Synthetic check: p=[0.8,0.2], q=[0.5,0.5]. The first candidate is accepted with probability 1 and the second with probability 0.4. Accepted masses are [0.5,0.2]; the remaining 0.3 comes entirely from replacement candidate 0. The resulting output distribution is exactly [0.8,0.2]. This is a one-step mathematical check, not an implementation of stochastic speculative decoding in the viewer.',
          'The graphical simulator implements its documented scripted greedy matching, accepted-prefix handling and recoverable K/V transaction. It does not execute the following stochastic acceptance formula. Real kernels additionally need actual state restoration when retention would otherwise overwrite required checkpoint entries.',
        ],
        String.raw`\alpha(x)=\min\left(1,\frac{p(x)}{q(x)}\right)\quad\text{for a proposal sampled from }q\text{ with }q(x)>0`,
      ),
    ],
  },
  'conditional-memory': {
    introduction: introduction(
      'Conditional n-gram memory retrieves features using short token patterns and then decides how to combine those features with the current context. It is a different mechanism from attention over earlier hidden states or an external search service.',
      'The same short token pattern can select the same stored parameter features in two different contexts. A context-dependent gate can then give those retrieved features different influence. The address comes from the pattern, while its use depends on the current representation. Different patterns can also collide at one address, so the lookup is not a guaranteed unique dictionary entry for each meaning.',
      [
        'Build a causal token window ending at the current position.',
        'Convert that pattern into a bounded lookup address.',
        'Retrieve the slot’s features and use the contextual representation to control their contribution.',
        'Fuse the result according to the architecture’s specified rule.',
      ],
      'The memory lab displays illustrative windows and addresses, not trained Engram activations or a full implementation of its published module. Hash collisions and context-dependent fusion need explicit treatment. This path is not executed by the numerical profiles.',
    ),
    technical: [
      detail(
        'technical-memory-contract',
        'Addressing, retrieval and fusion have different contracts',
        [
          'A complete conditional-memory definition specifies token compression, n-gram orders, hash heads, slot sizes, retrieved widths and context-dependent fusion. The bounded-address operation alone does not establish a unique mapping from linguistic meaning to stored content.',
          'Engram’s publication describes additional architecture-specific processing beyond this app’s small hash demonstration. Distinguish learned lookup parameters from sequence-dependent retrieved activations and from K/V state. Their indexing and update lifetimes are different even though all may be described informally as memory.',
          'For the cited context-gating step, use a retrieved row e and current hidden row h of width D. Project e to key k and value v of width D, then form a sigmoid gate from normalized h and k. In the row-vector convention below, W_K and W_V map retrieved width to D. This scalar gate lies between zero and one for finite inputs, unlike the unbounded signed SiLU branch in SwiGLU. The equation describes this published gating component; the later convolution/fusion remains a separate step and is not executed by the addressing illustration.',
        ],
        String.raw`k=eW_K,\quad v=eW_V,\quad \alpha=\sigma\!\left(\frac{\langle\operatorname{RMSNorm}(h),\operatorname{RMSNorm}(k)\rangle}{\sqrt D}\right),\quad \widetilde v=\alpha v`,
      ),
    ],
  },
  'mixture-of-experts': {
    introduction: introduction(
      'A mixture-of-experts layer chooses which learned expert transformations process each token. A router makes a token-dependent selection, selected experts compute outputs, and a combine rule merges their contributions. The model can store more expert parameters than it uses for a particular token.',
      'Picture directing different tasks to different specialists while retaining access to the whole team. A dense layer takes its specified path for every token; routed experts conditionally choose paths and combine their results. Experts need not correspond to human-readable skills. Their behavior depends on training and the routing design, rather than the label on a diagram branch.',
      [
        'Calculate router scores from the token’s current representation.',
        'Select the permitted expert paths using the architecture’s rule.',
        'Dispatch token features to those experts and run their transformations.',
        'Combine outputs with the specified weights and any shared-expert contribution.',
      ],
      'Extra branches alone do not implement MoE. Routing, dispatch, capacity and combination must all be specified. The current executable profiles are dense, so this article explains an extension rather than showing a hidden expert runtime.',
    ),
    technical: [
      detail(
        'technical-expert-accounting',
        'Parameter counts do not determine routed runtime cost',
        [
          'A routed block needs router parameters, expert matrices, a selection/weighting rule and dispatch/combine semantics. Some designs include experts evaluated for every token in addition to selected routed experts. State the exact definition used for activated-parameter counts.',
          'Total storage includes inactive experts; token-specific arithmetic uses a subset. Communication, load imbalance, batch organization and hardware scheduling can dominate performance. The cited DeepSeek report is one explicit architecture, not a universal router equation or a latency model for this app.',
        ],
      ),
    ],
  },
  mla: {
    introduction: introduction(
      'Multi-head latent attention changes what attention state is stored. In the cited design, each retained position keeps a compact joint content representation for keys and values, plus a separate positional key. Heads use compatible learned transformations to read that representation rather than retaining the usual explicit content key and value for every head.',
      'Follow one past position: its shared latent content can contribute differently to different query heads, while its positional key supplies a separate part of their comparison. This saves stored coordinates when the chosen latent widths are smaller, but still keeps state for each retained position. It is a trained architecture design, rather than ordinary file compression applied to any finished K/V cache.',
      [
        'Identify the actual persistent latent and positional components.',
        'Follow how those components participate in the attention calculation.',
        'Check which projections can be rearranged or absorbed under that architecture.',
        'Count the stored tensors that really exist rather than reusing a dense cache estimate.',
      ],
      'MLA is documented as an extension here and is not executed by the local numerical profiles. Other recurrent or linear-attention state also needs its own update/read semantics. A familiar name cannot justify squeezing a different representation into dense K/V axes.',
    ),
    technical: [
      detail('technical-latent-accounting', 'Account for every persistent component explicitly', [
        'A latent cache estimate sums each stored component’s scalar count and dtype over its actual layer, batch and position structure, including any separately retained positional representation. The dense K/V formula only applies when its axes really describe that state.',
        'Algebraic projection absorption can change which matrices are multiplied at read time without changing the intended architecture-level calculation. To reproduce such a model, specify both the latent-state contract and the chosen execution form. This requires a new compatible profile and numerical validation rather than relabeling GQA.',
      ]),
    ],
  },
  'tensor-laboratory': {
    introduction: introduction(
      'The tensor laboratory lets you inspect a small transformer’s actual calculations. Its parameters are seeded and untrained, so it is a controlled specimen for learning arithmetic, not a language model that has learned to answer questions.',
      'Treat it like a microscope slide prepared to expose a mechanism. You can select a value, see the operation that produced it and trace its inputs. The simplicity helps you follow a full calculation, while the provenance label prevents the specimen from being mistaken for a trained-model observation.',
      [
        'Choose a question, chapter or operation in the recorded run.',
        'Select an available tensor and a cell.',
        'Follow the operands to the earlier values and parameters that contributed.',
        'Change a controlled setting, compare aligned results and record what the calculation shows.',
      ],
      'Playback reveals a capture calculated ahead of time; it is not pausing a live GPU kernel. Transpose changes the display, and saved baseline comparisons require matching coordinate meanings. An arbitrary imported architecture does not become executable merely because its graph can be viewed.',
    ),
    technical: [
      detail('technical-trace-recipes', 'A scalar explanation is a typed dependency recipe', [
        'The reference trace associates stable operation/component IDs with output tensors and recipes. A cell explanation follows the corresponding contraction, normalization, rotation, mask or element-wise rule at original coordinates, while parameters remain available independently of playback position.',
        'Inspection must respect creation time and decoder-call provenance. A compatible baseline also requires shapes, semantic axes, dtype, absolute positions and, for masked-score subtraction, the visibility pattern. Historical trace retention is separate from the decoder’s live K/V payload.',
      ]),
    ],
  },
  'trained-model': {
    introduction: introduction(
      'The local trained path loads a small checkpoint and its matching tokenizer, then records the actual calculations behind a short completion. This is where the viewer can show learned parameter values and input-dependent activations from a trained model.',
      'Use it as a real but deliberately small specimen. You can investigate what this model did for a particular prompt without assuming it behaves like a large instruction-following assistant. The shared microscope makes it possible to compare familiar mechanisms while keeping trained and untrained provenance separate.',
      [
        'Explicitly load the pinned weights and tokenizer.',
        'Choose a short prompt and a bounded output budget, then generate.',
        'Inspect a recorded operation, parameter matrix or attention head.',
        'Trace selected values and keep the model identity and settings with your observations.',
      ],
      'The architecture builder does not resize this fixed checkpoint. Weak or odd completions are possible because it is a tiny base model. Opening a saved workspace restores study settings but does not silently load weights or run generation.',
    ),
    technical: [
      detail(
        'technical-checkpoint-identity',
        'Identity and arithmetic both constrain reproducibility',
        [
          'The manifest pins architecture, revision, tokenizer and asset hashes. The loader validates those assets before the reference worker runs them. A matching integer ID is insufficient if tokenizer identity differs, and a matching checkpoint is insufficient if the input and positional conventions differ.',
          'This profile uses the shared reference arithmetic with learned parameters and identifies tied classifier weights explicitly. Trained studies keep their own layouts, bookmarks and notes. Exporting a trace records arrays and provenance separately from the model document’s study configuration.',
        ],
      ),
    ],
  },
  'study-method': {
    introduction: introduction(
      'The most useful exploration starts with a question small enough to answer from the available evidence. Instead of watching the whole animation and guessing, choose one mechanism, predict what should change, inspect the relevant values and write down the limit of what you learned.',
      'Think of changing a single control in an experiment while keeping the rest of the setup stable. A small controlled question is often more informative than changing the entire prompt and comparing two unrelated continuations. The purpose is to isolate an observable calculation, not to earn a completion score.',
      [
        'State a concrete question and identify the operation that could answer it.',
        'Read the introduction, then move to technical detail when the coordinate rules matter.',
        'Use the scratch lab or actual numerical run to test a prediction.',
        'Record the setup, observation and the strongest conclusion the evidence supports.',
      ],
      'A toy intervention teaches arithmetic and is not automatically a causal experiment on a trained checkpoint. Changing architecture can also change parameter shapes and initialization. Always identify what was held fixed before interpreting a difference.',
    ),
    technical: [
      detail('technical-experimental-record', 'Specify an intervention as a reproducible record', [
        'Record model/profile identity, parameter source, tokenizer/input IDs, decoder call, layer/head, tensor name, semantic coordinates, absolute position map and the altered variable. Also record what was held fixed and where the result was measured.',
        'An observed difference is a function of that complete intervention. Comparing a later completion after an earlier sampling divergence does not isolate a single intermediate cause. Use operand-level arithmetic, aligned baselines and explicit scope to distinguish mechanism demonstrations from claims about model behavior.',
      ]),
    ],
  },
  workspace: {
    introduction: introduction(
      'The workspace keeps the model description and your study organized. It remembers graph views and notes, while the wiki gives you explanations and reading routes. Numerical and trained studies keep their own settings so exploring one does not overwrite another.',
      'Think of keeping a technical atlas, study notes and a collection of experimental recordings together while labeling each kind of material. The atlas explains structure; the notebook records observations; a trace stores calculated values. They should be connected without being mistaken for the same document.',
      [
        'Select a component or connection to open its authored explanation.',
        'Use a concept guide’s introduction, technical detail and lab to choose the right level of study.',
        'Return to matching model operations or open supported numerical examples.',
        'Export a workspace for portable model settings and notes, and export a trace separately for actual arrays.',
      ],
      'Browser-local saves and wiki reading places belong to an origin and device. Model exports are portable, but the separate device reading library is not embedded in them. Import validation protects the document contract without proving arbitrary imported mathematics correct.',
    ),
    technical: [
      detail(
        'technical-document-boundaries',
        'Preservation requires stable identity and separate state domains',
        [
          'The versioned model document owns authored semantics, equations, references, configuration, layouts and notebook content. Generated whole-model instances derive from authored recipes rather than becoming a second saved semantic copy. Stable IDs anchor layout restoration and navigation.',
          'Numerical/trained studies use independently validated namespaces, and wiki reading places use a separate versioned device key. Unknown future metadata is preserved instead of being silently reinterpreted. Routes and temporary educational scratch values are derived UI state rather than changes to the portable model contract.',
        ],
      ),
    ],
  },
}
