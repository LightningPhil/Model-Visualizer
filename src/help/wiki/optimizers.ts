import type { WikiArticle } from './types'
import { section } from './foundations'
import { detail, introduction, type TopicPerspectives } from './perspective-types'

export const optimizerArticles: WikiArticle[] = [
  {
    id: 'adam',
    title: 'Adam: adaptive moments and optimizer state',
    category: 'Learning & interpretation',
    summary:
      'Follow gradient averages, squared-gradient averages, bias correction and coordinate-wise parameter updates.',
    aliases: [
      'Adam optimizer',
      'adaptive moment estimation',
      'first moment',
      'second moment',
      'beta1',
      'beta2',
      'bias correction',
      'epsilon',
    ],
    kinds: [],
    graphs: [],
    prerequisites: ['learning-loop'],
    related: ['adamw', 'muon', 'numerical-precision', 'memory-budget'],
    sources: ['adam', 'adamw'],
    sections: [
      section(
        'adam-state',
        'The optimizer remembers gradient history',
        [
          'Backpropagation produces a gradient of the chosen loss. Adam then maintains an exponentially weighted first moment m and an exponentially weighted second raw moment v for each parameter coordinate. The second moment averages squared gradients; it is not the centered variance, and m is not a parameter or an activation from the forward graph.',
          'Initialize both buffers to zero. With beta values in [0,1), update the buffers once per optimizer step. A larger beta retains a longer history. This state belongs to training and is separate from input-dependent inference K/V. Resuming training needs compatible optimizer state and step counts in addition to weights; inference normally does not use these moment buffers.',
        ],
        String.raw`m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\qquad v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2`,
      ),
      section(
        'adam-correction',
        'Zero initialization motivates bias correction',
        [
          'With zero initial buffers and stationary first and second raw gradient moments, the moving averages are initially attenuated by factors involving beta raised to the step count. Adam divides by those factors before constructing the update. The expectation argument requires the corresponding moments to stay fixed, but not statistical independence between steps. It does not make a changing noisy gradient an exact estimate of the loss landscape.',
          'The step index starts at one for the first actual step. An explicit zero gradient still updates and decays the buffers; skipping a missing gradient may have different semantics in a framework. The teaching lab processes every supplied gradient as a step and makes t visible.',
        ],
        String.raw`\widehat m_t=\frac{m_t}{1-\beta_1^t},\quad \widehat v_t=\frac{v_t}{1-\beta_2^t},\quad \theta_t=\theta_{t-1}-\eta\frac{\widehat m_t}{\sqrt{\widehat v_t}+\epsilon}`,
      ),
      section('adam-direction', 'The denominator rescales each coordinate independently', [
        'For vector parameters, squares, square roots and division in these formulas are element-wise. The second raw moment is neither the centered gradient variance nor a Hessian estimate. A gradient and a parameter can have different physical or numerical scales; the adaptive denominator changes the update scaling, while eta remains the explicitly chosen learning-rate multiplier.',
        'The first moment smooths direction, while the square-root second moment supplies an adaptive scale. Epsilon is a positive numerical stabilizer outside the square root in the convention used here. Implementations with different epsilon placement or scaling are not necessarily numerically identical. Zero current gradient does not guarantee zero update because the first moment can retain history.',
        'For a nonzero constant scalar gradient g from an empty state, bias correction gives m-hat=g and v-hat=g² at every step. Ignoring epsilon only for that hand calculation, the adaptive direction is sign(g). This explains why doubling all gradient magnitudes need not double the update. Finite epsilon, nonconstant history and coordinate coupling in the model all matter.',
      ]),
      section('adam-regularization', 'Adam alone does not specify weight decay', [
        'Plain Adam in this guide consumes the supplied loss gradient without a penalty. Adding an L2 term to the objective changes that gradient before it enters both moment buffers. Decoupled weight decay instead applies a separate parameter shrinkage. AdamW explains why these are different under adaptive scaling.',
        'The lab feeds the same prescribed gradient sequence to three separate scalar states: plain Adam, Adam with coupled L2, and AdamW. These are controlled arithmetic comparisons, not three models learning the same loss trajectory. Gradients would normally change as each model’s parameters diverged; a smaller displayed parameter is not evidence of better training.',
      ]),
      section('adam-memory', 'Count optimizer buffers separately from weights and gradients', [
        'For P parameter scalars, Adam’s two dense moment buffers contain 2P scalars before sharding or compression. Their bytes depend on their actual dtype; a parameter storage label does not establish moment precision. Gradients, parameter copies, master weights, step metadata, activations and temporary optimizer work add separate costs.',
        'These counts are not total GPU memory or a speed prediction. Fused updates, distributed sharding, sparse parameter handling and optimizer variants need their own contracts. No optimizer in this wiki modifies the reference or trained checkpoint.',
      ]),
      section(
        'adam-history-window',
        'Read beta as a geometric memory of gradients',
        [
          'Expanding the recurrence gives a gradient from j steps ago a weight proportional to beta^j. The weights do not form a hard sliding window: old contributions decay continuously. The scale 1/(1−beta) is a useful rough horizon, while the exact half-life is log(1/2)/log(beta) for beta strictly between zero and one. Beta=0 keeps only the current value.',
          'For beta=0.9, the rough horizon is ten steps and the half-life about 6.58 steps; for beta=0.999 they are about one thousand and 692.8. These are properties of the moving average, not guarantees about optimization quality. Separate beta values control the first- and squared-gradient histories.',
        ],
        String.raw`m_t=(1-\beta_1)\sum_{j=1}^t\beta_1^{t-j}g_j,\qquad \mathbb E[m_t]=(1-\beta_1^t)\mu\ \text{when }\mathbb E[g_j]=\mu`,
      ),
    ],
  },
  {
    id: 'adamw',
    title: 'AdamW: decoupled weight decay',
    category: 'Learning & interpretation',
    summary:
      'Compare adding a penalty to gradients with shrinking parameters independently of adaptive moments.',
    aliases: [
      'Adam W',
      'weight decay',
      'L2 regularization',
      'coupled decay',
      'decoupled regularization',
    ],
    kinds: [],
    graphs: [],
    prerequisites: ['adam', 'learning-loop'],
    related: ['muon', 'post-training', 'numerical-precision'],
    sources: ['adamw', 'adam'],
    sections: [
      section(
        'adamw-l2',
        'Coupled L2 changes the gradient and its history',
        [
          'With an objective penalty lambda times half the squared parameter norm, the added gradient is lambda theta. Feeding this total gradient into Adam changes both m and v. The penalty contribution is therefore affected by coordinate-wise adaptive scaling and by its accumulated history.',
          'For ordinary SGD without adaptive scaling, this penalty can be written as a corresponding parameter shrinkage under matching coefficient conventions. That algebra does not make coupled L2 and decoupled decay interchangeable in Adam. The reference paper’s distinction concerns the update rule, not merely a new spelling for a regularization hyperparameter.',
        ],
        String.raw`\mathcal L_{\rm penalized}=\mathcal L+\tfrac\lambda2\|\theta\|_2^2,\qquad g_{\rm coupled}=\nabla\mathcal L+\lambda\theta`,
      ),
      section(
        'adamw-update',
        'Decoupling keeps decay out of the moments',
        [
          'AdamW forms moments from the loss gradient and applies a separate shrinkage of the old parameter. In this guide, decay coefficient lambda is multiplied by learning rate eta. The adaptive term and shrinkage both use the pre-update snapshot. State the convention when comparing implementations or schedules, because decay parameters can be normalized or scheduled differently.',
          'The isolated decay factor is 1−eta lambda. In the conventional range 0≤eta lambda≤1, it moves a parameter toward zero without changing its sign, reaching zero at the upper endpoint. Larger products can reverse the sign and eventually increase magnitude. Calling the operation decay is not a guarantee of shrinkage for arbitrary coefficients; the lab stays in a bounded teaching range.',
          'The decay term is not divided by the square-root second moment. Changing lambda can therefore change shrinkage without injecting that change into m and v. This does not mean the final result is independent of learning rate: the displayed shrinkage explicitly contains eta, and the loss gradients depend on the evolving model.',
        ],
        String.raw`\theta_t=(1-\eta\lambda)\theta_{t-1}-\eta\frac{\widehat m_t}{\sqrt{\widehat v_t}+\epsilon},\quad m_t,v_t\text{ use }g_t=\nabla\mathcal L`,
      ),
      section('adamw-example', 'A zero loss gradient makes the difference visible', [
        'Synthetic first step: theta=2, eta=0.1, lambda=0.1, zero initial moments and zero loss gradient. Plain Adam leaves theta at 2. AdamW shrinks it to 1.98. Coupled L2 supplies gradient 0.2; with bias correction its adaptive step is approximately 0.1 when epsilon is small, giving approximately 1.9. The values differ because the penalty passed through Adam’s denominator in only one case.',
        'Use the lab’s zero-gradient preset to reconstruct all three rows. With nonzero history, even a zero current loss gradient can produce an adaptive update. Do not conflate the cold-start example with the general zero-gradient case.',
      ]),
      section('adamw-groups', 'Parameter groups and the training recipe remain explicit choices', [
        'A training implementation may use different decay or learning-rate settings for different parameter groups. Biases, normalization gains, embeddings and output matrices need deliberate treatment; there is no universal exclusion rule implied by the optimizer name. Tied parameters must not accidentally receive multiple independent updates.',
        'AdamW retains Adam’s two moment buffers for the parameters it updates. Decoupling weight decay does not remove that state, compress K/V or change the attention architecture. Compare optimizers under matched data, schedules and tuning budgets; this arithmetic lab cannot rank their final model quality.',
      ]),
    ],
  },
  {
    id: 'muon',
    title: 'Muon: matrix momentum and Newton–Schulz updates',
    category: 'Learning & interpretation',
    summary:
      'Study matrix-aware update shaping, singular directions, finite Newton–Schulz iterations and parameter-group boundaries.',
    aliases: [
      'Muon optimizer',
      'matrix orthogonalization',
      'Newton Schulz',
      'polar factor',
      'singular values',
      'Nesterov momentum',
    ],
    kinds: [],
    graphs: [],
    prerequisites: ['learning-loop', 'linear-maps', 'adamw'],
    related: ['numerical-precision', 'memory-budget', 'feature-representations'],
    sources: ['muon', 'muon-code', 'adamw'],
    sections: [
      section('muon-role', 'Transform a matrix update, not every scalar independently', [
        'Muon builds a momentum-based candidate update for eligible weight matrices, then applies an approximate orthogonalizing matrix transform before the parameter step. Adam’s denominator acts coordinate by coordinate; Muon’s transformation depends on the matrix structure. It does not orthogonalize the model’s weight matrix itself or promise that learned features become mutually independent.',
        'The author reference assigns eligible hidden matrix parameters to Muon and uses another optimizer such as AdamW for other groups, including embeddings, outputs and scalar/vector gains or biases. Eligibility, reshaping, aspect-ratio scaling, decay and distributed implementation are part of a particular recipe. A name alone is not a complete optimizer specification.',
      ]),
      section(
        'muon-momentum',
        'Specify the momentum convention before comparing formulas',
        [
          'The pinned implementation inspected for this wiki uses an exponential moving average buffer M=beta M_old+(1−beta)G. Its optional Nesterov-like candidate is (1−beta)G+beta M using the newly updated buffer. Other presentations use differently scaled momentum recurrences. The lab labels this convention and lets you choose its lookahead candidate or ordinary momentum.',
          'For an idealized full-rank candidate B=UΣVᵀ, a polar-direction target replaces nonzero singular magnitudes by one while preserving the associated left/right directions. Rectangular matrices have a semi-orthogonal interpretation; they do not have both row and column Gram matrices equal to identity unless square. Rank-deficient zero directions need an explicit convention.',
          'The singular value decomposition gives a concrete interpretation: B sends a right singular direction v_i to sigma_i times its corresponding left direction u_i. U and V collect orthonormal directions, and Sigma contains nonnegative stretch magnitudes. The Frobenius norm is the square root of the sum of squared entries, equivalently squared singular values summed before the root; the spectral norm is the largest singular value. They are different norms.',
        ],
        String.raw`M_t=\beta M_{t-1}+(1-\beta)G_t,\quad B_t=(1-\beta)G_t+\beta M_t,\quad B_t=U\Sigma V^\top\ \longrightarrow\ UV^\top`,
      ),
      section(
        'muon-iteration',
        'Finite Newton–Schulz output is not an exact polar factor',
        [
          'The cited implementation Frobenius-normalizes the candidate with epsilon, transposes tall matrices for its chosen multiplication order, and iterates a quintic matrix polynomial with coefficients a=3.4445, b=−4.7750, c=2.0315. A finite number of these iterations is a deliberately approximate update-shaping procedure. Its output singular magnitudes need not be exactly one.',
          'The lab uses a square 2×2 family with fixed left/right singular directions. Gradients and old momentum share those directions, so scalar polynomial recurrences reconstruct the same matrix polynomial in exact arithmetic. It displays every iteration’s signed diagonal factors, their absolute singular magnitudes and the full output matrix, plus a separate ideal partial-polar target. Zero directions stay zero in the lab. Computation uses JavaScript binary64, not the reference’s bfloat16 GPU arithmetic.',
        ],
        String.raw`X_0=\frac{B}{\|B\|_F+\epsilon},\quad A_k=X_kX_k^\top,\quad X_{k+1}=aX_k+bA_kX_k+cA_k^2X_k`,
      ),
      section('muon-step', 'An update still needs learning rate, scaling and decay', [
        'In the pinned implementation, a returned update with m rows and n columns is multiplied by sqrt(max(1,m/n)) after the polynomial transform. These dimensions refer to that implementation’s matrix view; they are not the model’s sequence length or batch size. This is one explicit shape-scaling convention, with value one for a square matrix, rather than a universal definition of every Muon variant.',
        'The transformed direction is multiplied by the chosen learning rate and any recipe-specific shape adjustment before subtracting it from weights. The pinned code also supports decoupled weight decay. In its square case the displayed aspect-ratio multiplier is one; rectangular scaling conventions should not be inferred from this square demonstration.',
        'Muon uses a momentum buffer for its eligible parameters rather than Adam’s two per-coordinate moment buffers. That does not imply half total training memory: auxiliary AdamW groups, gradients, parameter/master copies, iteration scratch, activations and sharding all contribute. Orthogonalizing a candidate also costs computation. The lab demonstrates matrix geometry, not a performance benchmark or a general claim of superiority.',
      ]),
      section('muon-evidence', 'Read a concrete implementation and evaluate the whole run', [
        'The wiki links the author’s design explanation and an immutable revision of the reference code. The displayed coefficients, momentum and normalization describe that reference, not every future Muon variant. Wider models, different parameter groups, precision, schedules and distributed strategies require their own validation.',
        'Nothing here trains the loaded checkpoint, extracts its singular directions or changes its optimizer. For a fair empirical comparison, state the model, dataset, token budget, tuning budget, convergence metric and measured wall-clock/hardware conditions. An attractive matrix plot is not that evidence.',
      ]),
    ],
  },
]

export const optimizerPerspectives: Record<string, TopicPerspectives> = {
  adam: {
    introduction: introduction(
      'Adam turns loss gradients into parameter updates while keeping a history for each parameter coordinate. One moving average smooths direction; another tracks squared gradient size. The update combines those histories instead of treating the current gradient as the whole story.',
      'Imagine adjusting many controls whose readings have different scales and fluctuate over time. You keep a recent directional trend and a recent scale estimate for each control before choosing an adjustment. This picture explains the bookkeeping, not a guarantee that every adjustment improves a model.',
      [
        'Receive the current loss gradient from backpropagation.',
        'Update the directional and squared-gradient histories.',
        'Correct the initial attenuation caused by starting the histories at zero.',
        'Use the corrected values, learning rate and stabilizer to update the parameter.',
      ],
      'Adam’s history is training state, not attention cache or learned knowledge. A current zero gradient can still leave a nonzero historical direction. The lab uses supplied fictional scalar gradients; it does not train the loaded checkpoint or rank optimizers by model quality.',
    ),
    technical: [
      detail(
        'adam-initialization-check',
        'Reconstruct the first corrected step by hand',
        [
          'From empty buffers, the first moments are (1−beta1)g and (1−beta2)g². Dividing by the corresponding initialization factors recovers g and g². The first adaptive step is therefore eta times g/(|g|+epsilon), in the convention used here.',
          'This is not generally eta times the raw gradient. The lab exposes both raw and corrected moments so initialization, sign changes and historical inertia can be checked independently of the final parameter value.',
        ],
        String.raw`\Delta\theta_1=-\eta\frac{g_1}{|g_1|+\epsilon}`,
      ),
    ],
  },
  adamw: {
    introduction: introduction(
      'AdamW adds a separate shrinking operation to Adam’s gradient-based adjustment. The loss gradient controls the adaptive histories, while weight decay pulls the current parameter toward zero through its own term. Keeping those operations separate changes the mathematics.',
      'Think of adjusting a control using a smoothed feedback signal while separately applying a small reset toward its neutral setting. Feeding the reset signal into the feedback filter would change its history as well. That distinction is why adding a squared-weight penalty and applying separate decay can produce different Adam updates.',
      [
        'Compute moments from the loss gradient without adding the decay term.',
        'Calculate the adaptive Adam adjustment with its declared stabilizer.',
        'Calculate the separate shrinkage from the old parameter.',
        'Combine both changes and preserve the loss-gradient histories for the next step.',
      ],
      'Decoupled does not mean the final weights are independent of learning rate or the training trajectory. Parameter-group exclusions are recipe choices. The comparison lab supplies the same gradient sequence to separate scratch states and does not measure generalization or alter checkpoint weights.',
    ),
    technical: [
      detail('adamw-no-gradient', 'Separate a missing gradient from an explicit zero', [
        'The lab’s zero-gradient preset is an actual step from empty buffers. In general, an explicit zero loss gradient still decays existing moment estimates and can leave a nonzero adaptive update. Frameworks can skip parameters with missing gradients, including their state or decay; that is a separate implementation policy.',
        'An optimizer checkpoint must preserve parameter-group settings and correctly aligned buffers/step counts. Restarting moments while keeping weights changes the next update. Inference K/V state cannot substitute for this training state.',
      ]),
    ],
  },
  muon: {
    introduction: introduction(
      'Muon is an optimizer design for eligible hidden weight matrices. It gathers a momentum-based update and transforms that update using the matrix’s geometry before applying it. Other parameter groups can use an optimizer such as AdamW alongside it.',
      'Picture a matrix update as several coordinated directions rather than a collection of unrelated knobs. Some directions may dominate in magnitude. Muon’s matrix transform reshapes those directional magnitudes while retaining their associated orientation. This picture describes the update, not a claim that the learned weights or concepts become orthogonal.',
      [
        'Identify which parameter matrices belong to the Muon group.',
        'Form a momentum candidate under a declared convention.',
        'Normalize it and apply a finite matrix-polynomial iteration.',
        'Apply the chosen step scale and decay, while handling other groups with their own optimizer.',
      ],
      'The finite iteration approximates a directional target; it does not return a perfect orthogonal matrix in every case. The lab uses a restricted fictional two-by-two family and binary64 arithmetic. It does not emulate GPU bfloat16, train the model or establish that Muon is always faster or better.',
    ),
    technical: [
      detail(
        'muon-spectral-polynomial',
        'A matrix polynomial can be inspected one singular direction at a time',
        [
          'For the lab’s fixed orthogonal left/right factors, substituting U diag(s) Vᵀ into the stated polynomial transforms each signed diagonal factor with f(s)=as+bs³+cs⁵. Negative factors carry orientation signs; singular values are their absolute magnitudes. Zero remains zero.',
          'The ideal partial-polar comparison replaces nonzero factors by their signs. It is separate from the finite polynomial output, and rank-deficient zero directions are not filled in as arbitrary orthogonal extensions. The transform need not converge monotonically to unit magnitudes for these coefficients.',
        ],
        String.raw`s_{k+1}=3.4445s_k-4.7750s_k^3+2.0315s_k^5`,
      ),
    ],
  },
}
