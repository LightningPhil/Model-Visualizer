export type WikiSection = {
  id: string
  title: string
  paragraphs: string[]
  equation?: string
  bullets?: string[]
}

export type WikiArticle = {
  id: string
  title: string
  summary: string
  category: string
  aliases: string[]
  sections: WikiSection[]
  prerequisites: string[]
  related: string[]
  sources: string[]
  /** Exact semantic kinds; never infer a model's mathematics from its display title. */
  kinds: string[]
  graphs: string[]
  experiment?: { title: string; steps: string[]; observation: string }
}

export type WikiSource = { title: string; url: string; note: string }

export const wikiSources: Record<string, WikiSource> = {
  topk: {
    title: 'Fan, Lewis & Dauphin · Hierarchical Neural Story Generation (2018)',
    url: 'https://arxiv.org/abs/1805.04833',
    note: 'Primary example of top-k truncation for generation; the guide explains the filtering rule without claiming that this workbench implements it.',
  },
  nucleus: {
    title: 'Holtzman et al. · The Curious Case of Neural Text Degeneration (2020)',
    url: 'https://arxiv.org/abs/1904.09751',
    note: 'Nucleus/top-p sampling: adaptive probability-mass truncation and renormalization. Sampling order and tie handling remain implementation choices.',
  },
  adam: {
    title: 'Kingma & Ba · Adam: A Method for Stochastic Optimization (2015)',
    url: 'https://arxiv.org/abs/1412.6980',
    note: 'Algorithm 1: first/second raw moments, initialization bias correction and the epsilon-outside-square-root update convention.',
  },
  adamw: {
    title: 'Loshchilov & Hutter · Decoupled Weight Decay Regularization (2019)',
    url: 'https://arxiv.org/abs/1711.05101',
    note: 'Why L2 penalty gradients and decoupled weight decay differ under adaptive optimization. Coefficient/schedule conventions must be stated.',
  },
  muon: {
    title: 'Keller Jordan · Muon: An optimizer for hidden layers (2024)',
    url: 'https://kellerjordan.github.io/posts/muon/',
    note: 'Author design explanation: matrix momentum, approximate Newton–Schulz transformation and auxiliary optimizer groups. No universal performance guarantee.',
  },
  'muon-code': {
    title: 'KellerJordan/Muon · pinned reference implementation',
    url: 'https://github.com/KellerJordan/Muon/blob/f98f1cacc0263b04290753e32be8d498c1efc806/muon.py',
    note: 'Immutable source for the EMA/lookahead convention, quintic coefficients, normalization, finite-iteration approximation and shape scaling. The lab uses a restricted square family and binary64.',
  },
  bfloat: {
    title: 'Kalamkar et al. · A Study of BFLOAT16 for Deep Learning Training (2019)',
    url: 'https://arxiv.org/abs/1905.12322',
    note: 'BFLOAT16 versus IEEE FP16/FP32 format fields, range and mixed-precision arithmetic. The wiki lab does not emulate BF16 or FP16 execution.',
  },
  backprop: {
    title:
      'Rumelhart, Hinton & Williams · Learning representations by back-propagating errors (1986)',
    url: 'https://www.nature.com/articles/323533a0',
    note: 'Backward chain-rule gradients and parameter learning. The lab uses an independently specified scalar regression example and SGD step.',
  },
  superposition: {
    title: 'Elhage et al. · Toy Models of Superposition (2022)',
    url: 'https://arxiv.org/abs/2209.10652',
    note: 'Sparse overlapping representations and polysemanticity in toy models. The wiki’s fixed dictionary illustration is not a trained SAE or feature extraction result.',
  },
  rome: {
    title: 'Meng et al. · Locating and Editing Factual Associations in GPT (2022)',
    url: 'https://rome.baulab.info/',
    note: 'Causal tracing and a distinct parameter-editing method in particular models. The app’s synthetic patching lab does not implement trained-model causal tracing.',
  },
  alibi: {
    title: 'Press, Smith & Lewis · Train Short, Test Long (2022)',
    url: 'https://arxiv.org/abs/2108.12409',
    note: 'ALiBi distance penalties added to query/key scores, with head-dependent slopes. The lab exposes one slope rather than claiming to reproduce the full architecture.',
  },
  instructgpt: {
    title:
      'Ouyang et al. · Training language models to follow instructions with human feedback (2022)',
    url: 'https://arxiv.org/abs/2203.02155',
    note: 'An explicit demonstration, reward-model and policy-optimization pipeline. These stages are not executed by the local base checkpoint.',
  },
  dpo: {
    title: 'Rafailov et al. · Direct Preference Optimization (2023)',
    url: 'https://arxiv.org/abs/2305.18290',
    note: 'Policy/reference response log-probability ratios and the pairwise DPO objective. The teaching lab restricts the universe to two completions.',
  },
  floating: {
    title:
      'Goldberg · What Every Computer Scientist Should Know About Floating-Point Arithmetic (1991)',
    url: 'https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html',
    note: 'Original numerical analysis tutorial, reprinted in Oracle’s guide: rounding, cancellation, exceptions and floating-point arithmetic. The lab explicitly uses binary32 rounding and JavaScript binary64.',
  },
  linear: {
    title: 'Katharopoulos et al. · Transformers are RNNs (2020)',
    url: 'https://proceedings.mlr.press/v119/katharopoulos20a.html',
    note: 'Normalized kernel-feature attention and its causal matrix/vector recurrence. This is a specific alternative to softmax, not every recurrent architecture.',
  },
  kivi: {
    title: 'Liu et al. · KIVI (2024)',
    url: 'https://arxiv.org/abs/2402.02750',
    note: 'Asymmetric KV quantization, group directions and a high-precision residual region. The lab’s fictional packed groups do not implement KIVI.',
  },
  attention: {
    title: 'Vaswani et al. · Attention Is All You Need (2017)',
    url: 'https://arxiv.org/abs/1706.03762',
    note: '§3.2: scaled dot-product and multi-head attention; §3.3–3.4: position-wise networks, embeddings and output. The original encoder–decoder uses different normalization and position encoding from this teaching profile.',
  },
  bpe: {
    title: 'OpenAI · tiktoken educational implementation',
    url: 'https://github.com/openai/tiktoken/blob/main/tiktoken/_educational.py',
    note: 'Primary reference implementation of ranked byte-pair merging. This app uses its own small demonstration vocabulary.',
  },
  gqa: {
    title: 'Ainslie et al. · GQA (2023)',
    url: 'https://arxiv.org/abs/2305.13245',
    note: '§2: groups of query heads share keys and values; MHA and MQA are endpoints.',
  },
  mqa: {
    title: 'Shazeer · Fast Transformer Decoding (2019)',
    url: 'https://arxiv.org/abs/1911.02150',
    note: 'Multi-query attention and the cost of repeatedly accessing keys and values during decoding.',
  },
  rms: {
    title: 'Zhang & Sennrich · Root Mean Square Layer Normalization (2019)',
    url: 'https://arxiv.org/abs/1910.07467',
    note: 'RMS normalization, learned gain and the distinction from mean-centered LayerNorm.',
  },
  rope: {
    title: 'Su et al. · RoFormer (2021)',
    url: 'https://arxiv.org/abs/2104.09864',
    note: 'Rotary position embeddings: position-dependent orthogonal rotations and relative-position dot products.',
  },
  swiglu: {
    title: 'Shazeer · GLU Variants Improve Transformer (2020)',
    url: 'https://arxiv.org/abs/2002.05202',
    note: '§2: gated feed-forward variants, including the SwiGLU activation and multiplicative branches.',
  },
  residual: {
    title: 'He et al. · Deep Residual Learning (2015)',
    url: 'https://arxiv.org/abs/1512.03385',
    note: 'Residual learning as an identity path plus a learned update. Transformer placement is documented separately by Vaswani et al.',
  },
  mistral: {
    title: 'Jiang et al. · Mistral 7B (2023)',
    url: 'https://arxiv.org/abs/2310.06825',
    note: '§2: sliding-window attention, rolling buffers and chunked prefill. This app explicitly defines its own window-count convention.',
  },
  sparse: {
    title: 'Child et al. · Generating Long Sequences with Sparse Transformers (2019)',
    url: 'https://arxiv.org/abs/1904.10509',
    note: 'Structured sparse attention background. The fixed-stride anchors in this app are an illustrative policy, not this paper’s factorization.',
  },
  flash: {
    title: 'Dao et al. · FlashAttention (2022)',
    url: 'https://arxiv.org/abs/2205.14135',
    note: 'IO-aware exact attention with tiled computation; logical attention need not materialize a full score matrix in device memory.',
  },
  paged: {
    title: 'Kwon et al. · PagedAttention / vLLM (2023)',
    url: 'https://arxiv.org/abs/2309.06180',
    note: 'Physical KV block allocation and sharing, distinct from logical tensor axes and attention head sharing.',
  },
  speculative: {
    title: 'Leviathan et al. · Speculative Decoding (2023)',
    url: 'https://proceedings.mlr.press/v202/leviathan23a.html',
    note: 'Drafting and target verification with distribution-preserving stochastic acceptance. The graphical simulator here uses a simpler scripted greedy match rule.',
  },
  mtp: {
    title: 'Gloeckle et al. · Better & Faster LLMs via Multi-token Prediction (2024)',
    url: 'https://arxiv.org/abs/2404.19737',
    note: 'Auxiliary future-token objectives with independent prediction heads; distinguish this from sequential MTP.',
  },
  deepseek: {
    title: 'DeepSeek-AI · DeepSeek-V3 Technical Report',
    url: 'https://arxiv.org/html/2412.19437v2',
    note: '§2: MLA and routed experts; §2.2: sequential MTP. These are architecture-specific designs, not generic properties of every decoder.',
  },
  engram: {
    title: 'Cheng et al. · Conditional Memory via Scalable Lookup (2026)',
    url: 'https://arxiv.org/html/2601.07372v1',
    note: 'Engram: n-gram lookup and context-dependent fusion. This app’s small hash demonstration does not reproduce a trained Engram module.',
  },
  stories: {
    title: 'Karpathy · llama2.c',
    url: 'https://github.com/karpathy/llama2.c',
    note: 'Original reference implementation and small Stories checkpoints. This repo pins its own checkpoint manifest, tokenizer and asset hashes.',
  },
  interpretability: {
    title: 'Jain & Wallace · Attention is not Explanation (2019)',
    url: 'https://aclanthology.org/N19-1357/',
    note: 'Evidence against treating attention weights alone as a sufficient explanation of a prediction; read in the scope of the studied models and experiments.',
  },
  layernorm: {
    title: 'Ba et al. · Layer Normalization (2016)',
    url: 'https://arxiv.org/abs/1607.06450',
    note: 'Feature-wise mean and variance normalization; a useful comparison with RMSNorm.',
  },
}
