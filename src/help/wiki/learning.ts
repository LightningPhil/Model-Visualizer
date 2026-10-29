import type { WikiArticle } from './types'
import { section } from './foundations'

export const learningArticles: WikiArticle[] = [
  {
    id: 'learning-loop',
    title: 'How a model learns: gradients and updates',
    category: 'Learning & interpretation',
    summary:
      'Follow a prediction through loss, the chain rule, a gradient and an optimizer update.',
    aliases: [
      'backpropagation',
      'gradient descent',
      'learning rate',
      'optimizer',
      'training loop',
      'derivative',
      'SGD',
    ],
    kinds: [],
    graphs: [],
    prerequisites: ['linear-maps', 'training-objective'],
    related: [
      'adam',
      'adamw',
      'muon',
      'post-training',
      'numerical-precision',
      'parameters-activations-state',
      'residual',
    ],
    sources: ['backprop', 'attention'],
    sections: [
      section(
        'learning-forward',
        'A training example supplies both inputs and targets',
        [
          'The forward pass uses the current parameters to produce predictions. A loss compares them with targets under an explicit objective. For next-token training the input/target alignment matters: a causal hidden state predicts a following observed token, rather than reading that target through an unmasked future dependency.',
          'The lab uses one scalar regression example ŷ=wx+b with half squared error. It is deliberately small enough to expose every derivative. It is not a transformer training engine or a demonstration that next-token cross-entropy behaves identically to squared error. The trained and untrained workbenches continue to run forward inference only.',
        ],
        String.raw`\widehat y=wx+b,\quad e=\widehat y-y,\quad \mathcal L=\tfrac12e^2`,
      ),
      section(
        'learning-backward',
        'Backpropagation applies the chain rule to the executed graph',
        [
          'For the toy loss, the derivative with respect to the prediction is e. Multiplying by the prediction’s derivative gives dL/dw=ex and dL/db=e. Gradients describe local sensitivity at the current parameters. Their signs and magnitudes depend on the example, objective and all preceding operations.',
          'Reconstruct the default synthetic example: x=2, target y=3, w=0 and b=0 give prediction 0, error −3 and loss 4.5. The prediction derivative of the loss is −3; multiplying by x gives weight gradient −6, while the bias gradient is −3. A simultaneous step with learning rate eta=0.1 gives w=0.6 and b=0.3. The new prediction is 1.5 and the loss is 1.125. Both updates use the old error; updating one parameter and recomputing the other gradient would be a different algorithm.',
          'For a shared parameter, contributions from every use add. A complete transformer backward pass follows projections, normalization, attention, nonlinearities and shared embeddings. Backpropagation is the derivative calculation; it is not itself the optimizer. Some executed operations have piecewise derivatives or require conventions at nondifferentiable points.',
        ],
        String.raw`\frac{\partial\mathcal L}{\partial w}=ex,\quad \frac{\partial\mathcal L}{\partial b}=e,\quad w'=w-\eta ex,\quad b'=b-\eta e`,
      ),
      section('learning-step', 'An update is a choice, not a guaranteed improvement', [
        'The lab takes one simultaneous SGD step from a snapshot of the current gradients. Both new parameters use the old forward pass. It records before/after loss for that same example; a large learning rate can overshoot. Repeatedly fitting one example is not evidence of generalization.',
        'Real training aggregates examples and may use momentum, adaptive optimizer statistics, weight decay, gradient clipping and a learning-rate schedule. These are distinct mechanisms with state and conventions of their own. An optimizer step must not be confused with generating another token or appending to a KV cache.',
      ]),
      section(
        'learning-validation',
        'Check a derivative and check what you learned',
        [
          'A central finite difference estimates the derivative by evaluating slightly above and below the current parameter, holding the other operands fixed. For a general smooth nonlinear loss, a finite perturbation introduces truncation error while an extremely small one can suffer subtraction and rounding error. The scalar quadratic here has no central-difference truncation error in exact arithmetic; numerical agreement still depends on finite precision. This is a local check, not a proof for every implementation or input.',
          'Track held-out behavior separately from the loss used to choose the update. Training, validation and test data have different roles. A lower training loss does not establish factual correctness, robust reasoning or absence of memorization. The local toy’s history is disposable scratch state and never edits checkpoint parameters.',
        ],
        String.raw`\partial_w\mathcal L\approx\frac{\mathcal L(w+h)-\mathcal L(w-h)}{2h}`,
      ),
    ],
  },
  {
    id: 'position-methods',
    title: 'Compare positional encoding methods',
    category: 'Learning & interpretation',
    summary:
      'Compare learned absolute positions, sinusoidal additions, RoPE rotations and ALiBi score biases.',
    aliases: [
      'positional embeddings',
      'absolute positions',
      'sinusoidal encoding',
      'ALiBi',
      'position encoding',
      'length extrapolation',
    ],
    kinds: [],
    graphs: [],
    prerequisites: ['embeddings', 'attention-scores', 'rope'],
    related: ['attention-patterns', 'cache-retention', 'linear-maps'],
    sources: ['attention', 'rope', 'alibi'],
    sections: [
      section('position-entry', 'The method determines where position enters the graph', [
        'A learned absolute-position table supplies a parameter row indexed by position, often added to token embeddings before later transformations. Sinusoidal encoding adds deterministic frequency-based coordinates. RoPE rotates query/key feature pairs. ALiBi adds head-dependent distance biases to attention scores. These interventions occur at different locations and are not interchangeable checkpoint settings.',
        'The lab holds a two-feature content query and key fixed. Its sinusoidal-addition row uses one sine/cosine pair; its learned-addition row uses an explicitly fictional four-position table; its RoPE row rotates the pair; its ALiBi row penalizes causal query/key distance. Identity Q/K projections are assumed for the addition rows. Real embeddings pass through many transformations, so these numbers isolate entry points rather than reproduce a trained model.',
      ]),
      section(
        'position-sine',
        'Frequency pairs give deterministic coordinates',
        [
          'For an even width D, the original sinusoidal construction uses sine and cosine at geometrically spaced frequencies. An additive position vector changes the representation before later projections. Having a defined function at a large index is different from demonstrating useful behavior at lengths not seen during training.',
          'Learned absolute tables instead need an explicit supported index range and a policy for any extension. Interpolating, resizing or extrapolating position mechanisms can change model behavior and requires model-specific validation. The four-row teaching table is not a learned checkpoint table.',
        ],
        String.raw`P_{p,2i}=\sin(p/10000^{2i/D}),\quad P_{p,2i+1}=\cos(p/10000^{2i/D})`,
      ),
      section(
        'position-relative',
        'Relative effects can enter through rotations or biases',
        [
          'For a shared rotary frequency, the query/key dot product after rotations depends on their relative displacement through R(p)ᵀR(r)=R(r−p). This statement concerns a fixed content pair at that frequency, not invariance of an entire trained forward pass when all position conventions change.',
          'In causal ALiBi, a positive per-head slope m penalizes distance p−r for r≤p. The ordinary content score remains, and the causal mask still excludes future positions. The lab exposes one positive slope; the paper specifies a multi-head slope construction. Do not interpret the bias as learned evidence that recent tokens are always more important.',
        ],
        String.raw`s_{p,r}=q_p^\top k_r/\sqrt{D_h}-m_h(p-r),\qquad r\le p`,
      ),
      section('position-cache', 'Position identity survives cache compaction', [
        'The physical slot occupied by retained K/V is not its original absolute position. Reindexing compacted state as if it started at zero can change rotations, biases and visibility. Reuse therefore depends on compatible position conventions as well as matching token IDs and parameters.',
        'A model’s advertised maximum length, direct attention window and measured long-context task performance answer different questions. This comparison does not implement context extension or swap the running model’s positional method. Use the existing RoPE laboratory for the supported numerical profile.',
      ]),
      section(
        'position-cross-terms',
        'Why adding sinusoidal vectors differs from rotating content',
        [
          'In the lab’s identity-projection, one-frequency example, position vectors P_p=[sin(p),cos(p)] have inner product cos(p−r). That positional term alone depends only on displacement. But adding positions to content also creates the cross terms q·P_r and P_p·k. Those can depend on the absolute positions even when the displacement is fixed, explaining why the lab’s sinusoidal-addition score need not share RoPE’s fixed-content shift identity.',
          'The symbols q and k in this expansion are the fixed content vectors, while P_p and P_r are positional additions. Real learned Q/K projections and multiple layers make the full comparison architecture-specific; this expansion isolates the point where addition introduces interactions rather than claiming equivalence between complete models.',
        ],
        String.raw`\langle q+P_p,k+P_r\rangle=\langle q,k\rangle+\langle q,P_r\rangle+\langle P_p,k\rangle+\langle P_p,P_r\rangle`,
      ),
    ],
  },
  {
    id: 'feature-representations',
    title: 'Features, neurons and superposition',
    category: 'Learning & interpretation',
    summary:
      'Distinguish a coordinate, an activation pattern and a hypothesized feature before assigning meaning.',
    aliases: [
      'neuron meaning',
      'polysemanticity',
      'distributed representation',
      'superposition',
      'feature directions',
      'sparse autoencoder',
      'SAE',
    ],
    kinds: [],
    graphs: [],
    prerequisites: ['tensors', 'linear-maps'],
    related: ['causal-interventions', 'attention-interpretation', 'evidence'],
    sources: ['superposition', 'interpretability'],
    sections: [
      section(
        'features-coordinates',
        'A coordinate is observable; a feature is an interpretation',
        [
          'An array coordinate is defined by the model’s actual basis and axes. A proposed feature is a pattern or direction associated with some behavior or input property. One feature may involve many coordinates, and one coordinate may respond to multiple properties. A high activation on one example is not enough to give it a reliable human label.',
          'The lab defines three fictional nonnegative feature strengths and three directions in two dimensions. Their sum is the hidden vector. Because three nonzero directions cannot all be mutually orthogonal in two dimensions, a simple dot-product readout can receive interference from other features. This is a declared linear dictionary illustration, not a trained sparse autoencoder or a reproduction of the cited toy-model optimization.',
        ],
        String.raw`h=\sum_i a_i d_i,\qquad d_0^\top h=a_0\|d_0\|^2+\sum_{i\ne0}a_i d_0^\top d_i`,
      ),
      section('features-superposition', 'Sparse use can make overlapping representations useful', [
        'The default toy dictionary gives a precise ambiguity: with directions at 0°, 60° and 120°, d1=d0+d2. Feature strengths [0,1,0] and [1,0,1] therefore produce the same hidden vector, despite different F0 coefficients. Both strength vectors are nonnegative. A linear readout of that hidden vector cannot tell those two explanations apart without additional assumptions or information; naming one coordinate does not resolve the ambiguity.',
        'Superposition research investigates how more features can be represented than there are available dimensions, with interference and sparsity affecting the tradeoff. Polysemanticity describes a unit responding to multiple interpretable properties. Neither term licenses assigning semantic labels to arbitrary coordinates from a single heatmap.',
        'Sparsity of feature occurrence is different from sparse model weights, sparse attention visibility or expert routing. A nonlinear decoder can mitigate some interference in a particular learned system. This lab only shows vector sums and a linear readout, so it cannot establish the feature structure of the loaded checkpoint.',
      ]),
      section(
        'features-basis',
        'A consistent basis change can preserve a linear computation',
        [
          'If an orthogonal matrix R rotates both a hidden vector and its readout direction, their dot product is unchanged. Displayed coordinates can change while that particular scalar readout remains fixed. The lab’s view rotation makes this directly observable.',
          'This identity does not mean an arbitrary neuron-wise nonlinear network can be freely rotated without changing its behavior. A model’s projections, normalization and nonlinearities constrain what basis transformations preserve the complete computation. The example only rotates the vectors participating in the declared dot product.',
        ],
        String.raw`(Rd)^\top(Rh)=d^\top h,\qquad R^\top R=I`,
      ),
      section('features-evidence', 'A useful feature hypothesis needs more than a name', [
        'Inspect varied activating and nonactivating inputs, quantify the association, check alternative explanations and test targeted interventions with controls. Read out effects at specified tensors or output metrics. Correlation, a probe’s predictive success and a causal effect under intervention are different kinds of evidence.',
        'Learned dictionaries and sparse autoencoders are tools for proposing representations, with their own objectives, reconstruction errors and interpretation limits. This wiki does not claim to extract trained features automatically. Tensor cells retain their axis/coordinate labels instead of receiving invented semantic names.',
      ]),
    ],
  },
  {
    id: 'causal-interventions',
    title: 'Test a mechanism: ablation and activation patching',
    category: 'Learning & interpretation',
    summary:
      'Define a controlled change, rerun its downstream computation and measure a stated outcome.',
    aliases: [
      'causal tracing',
      'activation patching',
      'ablation',
      'intervention',
      'clean and corrupted',
      'mechanistic interpretability',
    ],
    kinds: [],
    graphs: [],
    prerequisites: ['evidence', 'feature-representations'],
    related: ['attention-interpretation', 'study-method', 'tensor-laboratory'],
    sources: ['rome', 'interpretability'],
    sections: [
      section('intervention-contract', 'Name exactly what changes and what stays fixed', [
        'A causal experiment identifies the model/checkpoint, clean and altered inputs, layer, token position, head or feature coordinates, intervention operator and downstream metric. Record where computation resumes and which quantities are recomputed. A heatmap comparison alone does not specify such an experiment.',
        'Ablation can zero or replace a tensor. Patching can copy a compatible activation from a clean run into an altered run. Restoring one metric under one intervention supports a scoped claim about that setup; it does not establish a universal location for a concept or a complete explanation of every output.',
      ]),
      section(
        'intervention-toy',
        'A small path makes the counterfactual explicit',
        [
          'The lab’s synthetic hidden state is [a,b], with output logit 3a−2b. Its clean state [1,1] gives 1; its altered state [0,1] gives −2. Patching the first coordinate from clean into altered restores 1. Zeroing the second clean coordinate gives 3: deleting a component can increase an output when it had a negative contribution.',
          'These are arithmetic counterfactuals in an authored toy graph. No hidden hooks alter the trained workbench, and the slider is not evidence about any language concept. The displayed sigmoid is just a two-outcome conversion of this scalar logit; the logit and its change remain separately visible.',
        ],
        String.raw`z=3a-2b,\qquad p=\sigma(z),\qquad \Delta z=z_{\rm intervention}-z_{\rm baseline}`,
      ),
      section('intervention-controls', 'Replacement values can create unfamiliar internal states', [
        'Zero, mean replacement and clean-run patching answer different questions. Some replacements lie outside the activation distribution seen during training. Coordinate alignment must include semantic axes, absolute positions and graph identity; matching shape alone is insufficient.',
        'Use negative controls such as irrelevant locations or unrelated source activations, multiple input pairs, and more than one outcome metric. A probability difference can be affected by changes in competing logits; compare a specified logit gap where appropriate. A large intervention effect can also reflect nonspecific disruption.',
      ]),
      section('intervention-scope', 'Necessity and sufficiency are conditional claims', [
        'An ablation that lowers a numerical metric shows an effect under that intervention; it does not by itself establish logical necessity. Define a success criterion before using that word. In the toy, “logit greater than zero” holds for clean [1,1] but fails when its first coordinate is replaced by zero. Restoring that coordinate is sufficient to recover this criterion with the second coordinate held at one. These claims are conditional on the specified background, replacement and criterion; redundant paths and interactions can change them elsewhere.',
        'The cited ROME work includes causal tracing in particular language models and a separate parameter-editing method. Activation replacement, weight editing, training and ordinary prompt changes are distinct interventions. This app’s current reference microscope records forward calculations; the new toy lab does not introduce trained-model patching support.',
      ]),
    ],
  },
  {
    id: 'post-training',
    title: 'From pretraining to assistant behaviour',
    category: 'Learning & interpretation',
    summary:
      'Separate next-token pretraining, supervised instruction tuning, reward-model RL and direct preference optimization.',
    aliases: [
      'SFT',
      'RLHF',
      'DPO',
      'instruction tuning',
      'alignment',
      'pretraining',
      'preference optimization',
      'reward model',
    ],
    kinds: [],
    graphs: [],
    prerequisites: ['training-objective', 'learning-loop'],
    related: ['sampling', 'trained-model', 'evidence'],
    sources: ['instructgpt', 'dpo', 'attention'],
    sections: [
      section('post-data', 'Training stages change data, objectives and behavior', [
        'Next-token pretraining fits observed sequences. Supervised instruction tuning trains on prompt/response demonstrations, with explicit choices about which tokens contribute to loss. These stages can use the same causal forward computation while changing data selection, target masking and optimization.',
        'An inference-time system message or chat template does not itself update weights. The tokenizer and formatting must match the checkpoint. The bundled Stories260K is a small base completion model, not an instruction-tuned assistant; the wiki must not describe its behavior as evidence that post-training was performed here.',
      ]),
      section('post-preferences', 'Preference data are comparisons under a labeling process', [
        'A preference example ranks responses to a prompt. In an RLHF pipeline such as the cited InstructGPT work, a reward model learns from comparisons and policy optimization uses reward feedback with a constraint or penalty against excessive policy drift. Reward is a learned training signal, not a direct measurement of truth.',
        'DPO instead optimizes a loss on preferred/rejected responses using policy and reference log-probability differences. It does not require the same explicit reward-model-plus-RL loop. Both data quality and the exact objective matter; preference optimization is not synonymous with every form of reinforcement learning.',
      ]),
      section(
        'post-objectives',
        'Write target masking and policy constraints explicitly',
        [
          'A demonstration objective can average negative token log probabilities over a declared target mask m. Prompt tokens may provide context while contributing no direct target loss, depending on the recipe. This is different from hiding those tokens from attention. The objective must state its normalization and exclude empty target sets.',
          'Read m_t as a zero/one inclusion flag, pi_theta as the trainable conditional response distribution and pi_ref as a fixed reference in the stated objective. Theta denotes model parameters. At a fixed prompt x, the KL term sums pi_theta(y|x) log[pi_theta(y|x)/pi_ref(y|x)] over complete responses y, rather than comparing just two token logits. Actual policy-optimization implementations estimate their chosen objectives and may use token-level decompositions.',
          'A conceptual reward-regularized policy objective maximizes expected reward while penalizing divergence from a reference policy. The displayed KL is a distribution-level expression over responses, with λ positive; implementations use particular sampling, reward and optimization approximations. It is not the DPO loss and is not executed by the lab.',
        ],
        String.raw`\mathcal L_{\rm SFT}=-\frac{\sum_t m_t\log\pi_\theta(y_t|x,y_{<t})}{\sum_t m_t},\quad J=\mathbb E_{y\sim\pi_\theta}[r(x,y)]-\lambda D_{\rm KL}(\pi_\theta\|\pi_{\rm ref})`,
      ),
      section(
        'post-dpo',
        'Sequence probabilities and reference anchoring have different roles',
        [
          'For a fixed prompt x and preferred/rejected responses y+ and y−, let rθ be the policy log-probability gap and rref the reference gap. The DPO pair loss is negative log sigmoid of β(rθ−rref). Response log probabilities normally sum token log probabilities under the appropriate causal contexts; they are not one arbitrary vocabulary logit.',
          'The sigmoid is sigma(s)=1/(1+exp(−s)); beta is the positive objective scale in this formulation, not a factual-confidence score. Evaluate a response log probability as the sum of its scored tokens’ conditional log probabilities, with the training recipe defining completion boundaries, target masks and end-of-sequence treatment. Computing those log probabilities directly avoids multiplying many tiny probabilities. The displayed ratios assume positive probabilities for the scored responses.',
          'The lab uses an explicitly normalized two-completion universe to expose this objective alongside a demonstration negative log likelihood. It shows loss sensitivity to a scalar policy gap, reference gap and β. It does not train an assistant, execute RL or estimate real response preferences. Comparing objective values does not establish which training method is better.',
        ],
        String.raw`\mathcal L_{\rm DPO}=-\log\sigma\!\left(\beta\left[\log\frac{\pi_\theta(y^+|x)}{\pi_\theta(y^-|x)}-\log\frac{\pi_{\rm ref}(y^+|x)}{\pi_{\rm ref}(y^-|x)}\right]\right)`,
      ),
      section(
        'post-evaluation',
        'Behavioral evaluation is separate from fitting a training signal',
        [
          'Instruction following, factuality, calibration, robustness and preference ratings require stated evaluation procedures. Improving a preference loss need not improve every measure. A model can confidently produce an incorrect answer, and decoding settings cannot add missing training evidence.',
          'Keep the checkpoint’s provenance and actual supported stages visible. This repository adds explanatory objectives and local synthetic calculations, without downloading training datasets, modifying the checkpoint or sending prompts to a service.',
        ],
      ),
    ],
  },
  {
    id: 'numerical-precision',
    title: 'Floating-point precision and numerical stability',
    category: 'Learning & interpretation',
    summary:
      'Understand representation error, accumulation order, cancellation, overflow and stable equivalent formulas.',
    aliases: [
      'Float32',
      'FP32',
      'FP16',
      'BF16',
      'rounding',
      'overflow',
      'underflow',
      'NaN',
      'mixed precision',
      'numerical stability',
    ],
    kinds: [],
    graphs: [],
    prerequisites: ['linear-maps', 'softmax'],
    related: ['cache-quantization', 'storage-kernels', 'learning-loop'],
    sources: ['floating', 'bfloat', 'flash', 'attention'],
    sections: [
      section('precision-format', 'Storage precision and arithmetic precision are separate', [
        'For normal binary32 numbers in the interval [2^e,2^(e+1)), adjacent values are spaced by 2^(e−23). The lab’s A=100000000 lies in the e=26 interval, where spacing is 8. Adding 1 and rounding back to binary32 cannot preserve that unit. Canceling the two equal stored large values first instead leaves an exactly representable unit. This connects the observed order dependence to the format rather than treating it as an unexplained calculator quirk.',
        'Finite binary floating-point formats represent a bounded set of values using sign, exponent and significand fields. Many decimal fractions cannot be represented exactly. Binary32 has 24 significant binary digits including its implicit leading bit for normal numbers; binary16 has 11; bfloat16 has 8. Binary16 and bfloat16 have different exponent ranges despite equal storage size.',
        'A tensor’s storage label does not specify every arithmetic step. Multiplication, accumulation and output conversion can use different precisions. The numerical reference uses JavaScript number arithmetic for many reductions and stores arrays in Float32; it is not a bitwise simulation of a GPU mixed-precision kernel.',
      ]),
      section(
        'precision-order',
        'Equivalent real-number expressions can round differently',
        [
          'Floating-point addition is not generally associative. The lab compares two parenthesizations of [A,1,−A], rounding each operand and addition through Math.fround for binary32. At A=100000000, (A+1)−A loses the unit, while A+(1−A) can lose it too; pairing A with −A before adding 1 preserves it. The displayed operation order is part of the experiment.',
          'The JavaScript binary64 comparison is a higher-precision computation, not arbitrary-precision exact arithmetic. For these selected integer inputs within its exact range, the real arithmetic target is 1. Large cancellation can expose earlier rounding error. Reordering or compensated summation can help for some workloads but does not make every reduction exact.',
        ],
        String.raw`\operatorname{fl}(\operatorname{fl}(A+1)-A)\ne\operatorname{fl}(\operatorname{fl}(A-A)+1)`,
      ),
      section(
        'precision-softmax',
        'Stability can require changing an equivalent formula',
        [
          'Exponentiating large logits directly can overflow even though their normalized probabilities are well-defined. Subtracting the row maximum before exponentiation preserves real-number softmax and limits its largest exponential to one for a finite nonempty row. Masked or wholly invalid rows still need explicit handling.',
          'The lab compares naive and max-shifted two-score softmax in JavaScript. It reports Infinity or NaN explicitly rather than turning them into meaningful probabilities. Underflow can also make tiny probabilities round to zero. A stable expression controls avoidable error; it does not eliminate representation limits.',
        ],
        String.raw`\frac{e^{z_i}}{\sum_j e^{z_j}}=\frac{e^{z_i-m}}{\sum_j e^{z_j-m}},\qquad m=\max_j z_j`,
      ),
      section('precision-validation', 'Choose a tolerance that matches the calculation', [
        'Bitwise identity and numerical agreement within a tolerance are different tests. Absolute error is useful near zero; relative error needs a meaningful reference scale. Compare aligned semantic coordinates and visibility metadata before interpreting a difference as a precision issue.',
        'Neither this lab nor the cache-byte controls implement FP16/BF16 kernels. Quantization uses a separate encoding design and metadata; floating-point rounding is not the same as arbitrary low-bit affine quantization. Benchmark and validate the actual runtime before making speed, memory or quality claims.',
      ]),
    ],
  },
]
