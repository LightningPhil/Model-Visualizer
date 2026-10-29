import { detail, introduction, type TopicPerspectives } from './perspective-types'

export const learningPerspectives: Record<string, TopicPerspectives> = {
  'learning-loop': {
    introduction: introduction(
      'Learning changes the reusable parameters of a model. A training example supplies a target, the current model makes a prediction, and a loss measures the mismatch. Derivatives tell us how a small parameter change would affect that loss locally. An optimizer chooses an update using that information.',
      'Picture adjusting the controls of a reusable prediction machine after checking an example with a known result. The derivative tells you a local direction to try, rather than revealing the perfect final settings. Taking too large a step can make the mismatch worse, and practicing one example does not prove you will handle a new one.',
      [
        'Run the current parameters forward to obtain a prediction and loss.',
        'Work backward through the calculation to find parameter sensitivities.',
        'Apply an explicitly chosen optimizer step from the same gradient snapshot.',
        'Measure both the new training loss and performance on separate examples.',
      ],
      'Generating a token normally changes the sequence and cache, not the learned weights. This lab trains only a fictional scalar predictor. It does not modify the loaded model or implement transformer training. A gradient describes local sensitivity, not a complete causal explanation of model behavior.',
    ),
    technical: [
      detail(
        'technical-learning-curvature',
        'Predict when the one-example update overshoots',
        [
          'For this two-parameter scalar predictor, a simultaneous SGD update changes the example’s residual from e to e[1−η(x²+1)]. The same-example squared loss therefore shrinks only when the magnitude of that factor is below one, unless the loss was already zero. This gives a direct check against the displayed before/after loss.',
          'This condition is specific to the stated quadratic example with both weight and bias updated. It is not a universal stable learning rate for a transformer or another optimizer. The lab’s finite-difference check verifies the analytic weight derivative for the current example independently of the step choice.',
        ],
        String.raw`e'=e\left[1-\eta(x^2+1)\right],\qquad 0<\eta<\frac{2}{x^2+1}`,
      ),
    ],
  },
  'position-methods': {
    introduction: introduction(
      'A model needs a way to distinguish where pieces of a sequence occur. Position can enter as an added representation, a rotation of attention features, or a bias on comparisons. These methods give the model different information through different parts of the computation.',
      'Imagine distinguishing identical sounds by when they occur. You could attach a time label, transform a signal according to time, or alter how you compare events based on their separation. These are different ways to make order available; they do not prove the model uses every long-range relationship successfully.',
      [
        'Keep the content vectors fixed so the effect of position is visible.',
        'Identify whether position changes inputs, query/key features or comparison scores.',
        'Move the query and key and compare methods under declared assumptions.',
        'Keep absolute position identity when compacting or reusing cached state.',
      ],
      'The lab uses a fictional position table, one sinusoidal pair and identity projections for addition examples. It does not switch the running checkpoint’s architecture. Defining an encoding at longer positions is different from validating a model beyond its training or configured context.',
    ),
    technical: [
      detail(
        'technical-position-shift',
        'Test a common shift without changing relative displacement',
        [
          'For a fixed rotary content pair at the lab’s single frequency, shifting both positions equally preserves their relative rotation. For the stated ALiBi rule, the distance bias is also unchanged. The fictional absolute table and sinusoidal-addition comparisons need not preserve the score under a common shift.',
          'This is a controlled comparison before the mask and before later layers. The lab shows a future key as excluded in every method; a position bias never overrides the causal mask. Consistent pair conventions and frequencies are part of the rotary identity.',
        ],
        String.raw`R(p+c)^\top R(r+c)=R(r-p),\qquad (p+c)-(r+c)=p-r`,
      ),
    ],
  },
  'feature-representations': {
    introduction: introduction(
      'A feature is a proposed pattern in what the model represents. A neuron or tensor coordinate is one measurable part of the implementation. They need not correspond one-to-one: a pattern can spread across many coordinates, and one coordinate can respond to several different patterns.',
      'Several sounds can be mixed into the same pair of recording channels. A channel’s level is observable, but it does not automatically identify which sound produced it. Likewise, observing a large activation is a starting point for investigation rather than a reliable semantic label.',
      [
        'Identify the exact tensor coordinates and their provenance.',
        'Look across varied examples before proposing a meaning.',
        'Consider overlapping directions and interference between features.',
        'Use controlled interventions and independent tests to challenge the hypothesis.',
      ],
      'The lab mixes three fictional feature directions into two dimensions and rotates the view. It has no learned semantics and does not extract features from the checkpoint. A basis change preserving one dot product does not make arbitrary rotations of a nonlinear model equivalent.',
    ),
    technical: [
      detail(
        'technical-feature-readout',
        'Separate a feature coefficient from a contaminated readout',
        [
          'For unit dictionary direction d0, its dot product with the hidden mixture contains its own coefficient plus contributions from nonorthogonal directions. Changing a different feature can therefore change this readout. The lab displays the desired fictional coefficient and the interference separately.',
          'An observed readout need not be a uniquely recoverable latent coefficient. Recoverability depends on the dictionary, constraints, nonlinear decoder and feature distribution. The example demonstrates a geometric ambiguity without claiming that every trained feature uses this exact dictionary model.',
        ],
      ),
    ],
  },
  'causal-interventions': {
    introduction: introduction(
      'To test a suspected mechanism, deliberately change a specified internal quantity and calculate what happens downstream. Ablation removes or replaces a component. Activation patching copies a compatible internal value from another run. Both require a clearly defined experiment and outcome.',
      'A circuit diagram suggests possible explanations, but a controlled experiment changes one component while keeping the rest of the setup specified. The result tells you what happened under that intervention. It does not automatically explain all behavior of every circuit with a similar-looking part.',
      [
        'Choose clean and altered inputs plus a concrete outcome metric.',
        'Specify which coordinate or tensor to replace and where its replacement comes from.',
        'Recompute the affected downstream calculation and compare with baselines.',
        'Repeat with controls, other examples and alternative interventions.',
      ],
      'An ablation can increase an output if it removes a negative contribution. Patching can also create an unfamiliar internal state. This lab performs explicit arithmetic interventions in a toy graph; it does not alter trained activations or prove facts about a language model’s concepts.',
    ),
    technical: [
      detail('technical-patch-metric', 'State the reference background for every difference', [
        'A clean-baseline delta and an altered-baseline delta answer different questions. The toy patched state matches the clean state, so it has zero clean delta but a positive altered delta. Displaying both avoids treating restoration as a context-free score.',
        'Normalized recovery ratios can be unstable when clean and altered outcomes are nearly equal. Report raw metrics and reference values as well. Sufficiency under a chosen altered background is distinct from necessity across all contexts.',
      ]),
    ],
  },
  'post-training': {
    introduction: introduction(
      'A base language model learns to predict observed sequences. Additional training can encourage responses that follow instructions or match labeled preferences. These stages change parameters through particular data and objectives; they are different from simply changing a prompt during inference.',
      'Practicing general prediction, studying worked response examples and comparing preferred answers are different learning activities. They can share the same underlying prediction machinery while supplying different training signals. None of them makes a preference label a guarantee of truth.',
      [
        'Identify the checkpoint and which training stages actually occurred.',
        'Distinguish demonstrations from rankings of alternative responses.',
        'Follow the objective and reference model used by a specific method.',
        'Evaluate instruction following, factuality and robustness separately from training loss.',
      ],
      'The lab compares demonstration and preference losses in a fictional two-completion universe. It does not run RL, train an assistant or claim that the base Stories checkpoint received instruction tuning. Templates and system messages influence input, but do not themselves update weights.',
    ),
    technical: [
      detail(
        'technical-dpo-gap',
        'Differentiate the preference loss with respect to a policy gap',
        [
          'Holding the reference gap fixed, the derivative of the DPO pair loss with respect to the policy log-probability gap is −β sigmoid(−β(g−gref)). The lab displays this local derivative. A model-parameter gradient additionally follows how both response log probabilities depend on the shared parameters.',
          'In the two-completion teaching universe, the policy gap is a scalar logit gap and the preferred completion has sigmoid(g) probability. For real variable-length responses, use sequence log probabilities with the correct causal contexts, masking and objective conventions. The simplified scalar is not a direct estimate from a real assistant.',
        ],
        String.raw`g=\log\pi_\theta(y^+|x)-\log\pi_\theta(y^-|x),\qquad \partial_g\mathcal L=-\beta\sigma[-\beta(g-g_{\rm ref})]`,
      ),
    ],
  },
  'numerical-precision': {
    introduction: introduction(
      'A computer stores only a finite set of numerical values. Calculations can round, lose tiny contributions, overflow or underflow. The number format, order of operations and precision used for accumulation all affect the result you inspect.',
      'Imagine adding measurements with a calculator that keeps only a limited number of significant digits. A tiny addition to a very large number can disappear. Subtracting the large number afterwards cannot recover the lost information. Changing the order can sometimes preserve it.',
      [
        'Identify the storage format and the precision of intermediate arithmetic.',
        'Inspect rounding and cancellation in a small calculation.',
        'Compare naive and numerically stable forms of the same mathematical rule.',
        'Use suitable tolerances and aligned coordinates when checking a reference result.',
      ],
      'The lab explicitly rounds operations to binary32 with Math.fround and compares JavaScript binary64 arithmetic. It does not emulate FP16/BF16 GPU kernels. A cache byte-size assumption is not an executed precision conversion, and numerical agreement does not establish model quality.',
    ),
    technical: [
      detail(
        'technical-error-budget',
        'A tolerance needs both scale and a reference',
        [
          'One common comparison allows an absolute tolerance plus a relative tolerance times the reference magnitude. Near zero the absolute term matters; at large scale the relative term can matter more. Specify which value is the reference and handle nonfinite values explicitly rather than subtracting infinities.',
          'This is a validation rule, not a universal error bound for every operation. Floating-point differences can arise from reduction order, intermediate precision, fused operations and output rounding. Compare the actual implementation contract before inferring a bug from non-bitwise agreement.',
        ],
        String.raw`|a-b|\le\mathrm{atol}+\mathrm{rtol}|b|`,
      ),
    ],
  },
}
