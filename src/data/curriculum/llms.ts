import { CurriculumNode } from './types'

/**
 * Language Modeling Fundamentals
 *
 * Module 4.1: Language Modeling Fundamentals
 * 1. What is a Language Model?
 * 2. Tokenization
 * 3. Embeddings & Positional Encoding (planned)
 */
const languageModelingFundamentals: CurriculumNode = {
  slug: 'language-modeling-fundamentals',
  title: 'Language Modeling Fundamentals',
  children: [
    {
      slug: 'what-is-a-language-model',
      title: 'What is a Language Model?',
      description:
        'A language model predicts the next token given context\u2014the same supervised learning you already know, applied to text.',
      duration: '25 min',
      category: 'Language Modeling',
      objectives: [
        'Understand language modeling as next-token prediction',
        'Explain autoregressive generation: sample, append, repeat',
        'Describe how temperature reshapes the probability distribution',
        'Connect language modeling to the supervised learning frame you already know',
      ],
      skills: [
        'next-token-prediction',
        'autoregressive-generation',
        'probability-distribution',
        'temperature-sampling',
        'language-modeling',
      ],
      prerequisites: ['transfer-learning-project'],
      exercise: {
        constraints: [
          'What language models do, not how they work internally',
          'No attention, transformers, or architecture details',
          'No tokenization or embeddings\u2014those are the next two lessons',
          'No code\u2014conceptual only',
        ],
        steps: [
          'Connect phone autocomplete to language modeling',
          'Understand next-token prediction as a probability distribution over vocabulary',
          'Trace autoregressive generation step by step',
          'Experiment with temperature in the interactive widget',
          'See why "just predict the next token" is a powerful training signal',
        ],
      },
    },
    {
      slug: 'tokenization',
      title: 'Tokenization',
      description:
        'How text becomes the integer sequences your language model actually processes\u2014and why the method matters more than you\u2019d think.',
      duration: '30 min',
      category: 'Language Modeling',
      objectives: [
        'Explain why text must be converted to integers for a language model',
        'Compare character-level, word-level, and subword tokenization tradeoffs',
        'Describe the BPE algorithm: start from characters, iteratively merge the most frequent pair',
        'Implement BPE from scratch in a notebook',
        'Identify how tokenization affects model behavior (letter counting, arithmetic, multilingual)',
      ],
      skills: [
        'tokenization',
        'byte-pair-encoding',
        'subword-tokenization',
        'vocabulary-design',
      ],
      prerequisites: ['what-is-a-language-model'],
      exercise: {
        constraints: [
          'Subword tokenization concept and BPE algorithm only',
          'WordPiece and SentencePiece mentioned, not implemented',
          'No embeddings\u2014that\u2019s the next lesson',
          'Minimal BPE implementation, not production-grade',
        ],
        steps: [
          'See why character-level and word-level tokenization each fail',
          'Understand subword tokenization as the middle ground',
          'Trace BPE merges step by step with the interactive widget',
          'Implement BPE from scratch in the Colab notebook',
          'See how tokenization creates real model limitations (strawberry, arithmetic)',
        ],
      },
    },
    {
      slug: 'embeddings-and-position',
      title: 'Embeddings & Positional Encoding',
      description:
        'How integer token IDs become rich vectors the model can compute with\u2014and why the model needs to be told where each token is.',
      duration: '30 min',
      category: 'Language Modeling',
      objectives: [
        'Explain why one-hot encoding fails and how embeddings solve both the dimensionality and similarity problems',
        'Describe nn.Embedding as a learnable weight matrix indexed by integer',
        'Explain why positional encoding is needed (the bag-of-words problem)',
        'Describe sinusoidal positional encoding and its properties',
        'Implement embeddings and positional encoding in PyTorch',
      ],
      skills: [
        'token-embeddings',
        'positional-encoding',
        'nn-embedding',
        'one-hot-encoding',
        'sinusoidal-encoding',
      ],
      prerequisites: ['tokenization'],
      exercise: {
        constraints: [
          'Token embeddings and positional encoding only',
          'No attention or transformer internals\u2014that\u2019s Module 4.2',
          'No standalone embedding methods (Word2Vec, GloVe)',
          'No training from scratch\u2014Module 4.3',
        ],
        steps: [
          'See why integer IDs are arbitrary and one-hot encoding fails',
          'Understand embeddings as learned lookup tables',
          'Explore pretrained embedding space in the interactive widget',
          'See the bag-of-words problem that motivates positional encoding',
          'Visualize sinusoidal positional encoding patterns',
          'Implement embeddings and PE from scratch in the Colab notebook',
        ],
      },
    },
  ],
}

/**
 * Attention & the Transformer
 *
 * Module 4.2: Attention & the Transformer
 * 1. The Problem Attention Solves
 * 2. Queries, Keys, and the Relevance Function (planned)
 * 3. Values and Attention Output (planned)
 * 4. Multi-Head Attention (planned)
 * 5. The Transformer Block (planned)
 * 6. Decoder-Only Transformers (planned)
 */
const attentionAndTransformer: CurriculumNode = {
  slug: 'attention-and-transformer',
  title: 'Attention & the Transformer',
  children: [
    {
      slug: 'the-problem-attention-solves',
      title: 'The Problem Attention Solves',
      description:
        'How tokens create context-dependent representations using dot products and softmax\u2014and why one vector per token isn\u2019t enough.',
      duration: '30 min',
      category: 'Attention',
      objectives: [
        'Explain why static embeddings are insufficient for context-dependent meaning',
        'Describe dot-product attention: similarity scores, softmax normalization, weighted averaging',
        'Compute raw dot-product attention by hand on a small example',
        'Identify the dual-role limitation: one vector for both seeking and offering',
        'Explain why the raw attention matrix is symmetric and why that\u2019s a problem',
      ],
      skills: [
        'dot-product-attention',
        'attention-weights',
        'context-dependent-representations',
        'weighted-average',
        'attention-matrix',
      ],
      prerequisites: ['embeddings-and-position'],
      exercise: {
        constraints: [
          'Raw dot-product attention only (no Q, K, V projections)',
          'No multi-head attention or transformer architecture',
          'No scaled dot-product attention (scaling by \u221Ad)',
          'No causal masking',
        ],
        steps: [
          'See why static embeddings can\u2019t handle polysemy',
          'Understand weighted average as a mechanism for context',
          'Trace dot-product attention step by step on a tiny example',
          'Explore the interactive attention heatmap',
          'Feel the symmetry limitation and dual-role problem',
          'Implement raw attention from scratch in the Colab notebook',
        ],
      },
    },
    {
      slug: 'queries-and-keys',
      title: 'Queries, Keys, and the Relevance Function',
      description:
        'Two learned projection matrices break the symmetry limitation\u2014giving each token separate vectors for seeking and offering.',
      duration: '25 min',
      category: 'Attention',
      objectives: [
        'Explain how Q and K projections separate the seeking and offering roles that raw attention conflates',
        'Compute QK\u1D40 by hand and verify it produces an asymmetric relevance matrix',
        'Explain why scaling by \u221Ad_k prevents softmax saturation and keeps gradients alive',
        'Connect Q/K projections to nn.Linear layers you\u2019ve already used',
      ],
      skills: [
        'query-key-projections',
        'scaled-dot-product-attention',
        'asymmetric-attention',
        'softmax-saturation',
        'linear-projection',
      ],
      prerequisites: ['the-problem-attention-solves'],
      exercise: {
        constraints: [
          'Q and K projections and the scaling factor only',
          'No V projection or attention output\u2014that\u2019s the next lesson',
          'No multi-head attention, transformer block, or causal masking',
          'No cross-attention (self-attention only)',
        ],
        steps: [
          'Resolve the dual-role cliffhanger from The Problem Attention Solves',
          'Understand Q and K as learned linear projections via the job fair analogy',
          'Hand-trace Q, K, and QK\u1D40 on the same 4-token example from last lesson',
          'Compare raw XX\u1D40 weights (symmetric) to QK\u1D40 weights (asymmetric) side by side',
          'Understand why scaling by \u221Ad_k is essential at high dimensions',
          'Implement Q/K projections and verify asymmetry in the Colab notebook',
        ],
      },
    },
    {
      slug: 'values-and-attention-output',
      title: 'Values and the Attention Output',
      description:
        'The third projection separates "what makes me relevant" from "what I contribute"\u2014completing single-head attention and introducing the residual stream.',
      duration: '25 min',
      category: 'Attention',
      objectives: [
        'Explain how the V projection separates matching (K) from contributing (V)',
        'Compute the full single-head attention output: softmax(QK\u1D40/\u221Ad_k) V',
        'Explain why the attention output is added to the input via the residual stream',
        'Connect the residual stream to skip connections from ResNets',
      ],
      skills: [
        'value-projection',
        'single-head-attention',
        'attention-output',
        'residual-stream',
        'residual-connections',
      ],
      prerequisites: ['queries-and-keys'],
      exercise: {
        constraints: [
          'V projection and single-head attention output only',
          'Residual stream introduced, not fully developed',
          'No multi-head attention or output projection W_O\u2014that\u2019s next lesson',
          'No transformer block, layer normalization, or causal masking',
        ],
        steps: [
          'Recognize the matching-vs-contributing dual-role pattern from previous lessons',
          'Understand V as a third learned projection via the job fair resume analogy',
          'Hand-trace V, attention output, and residual output on the same 4-token example',
          'See the formula evolution across three lessons: softmax(XX\u1D40)X \u2192 softmax(QK\u1D40/\u221Ad_k) \u2192 softmax(QK\u1D40/\u221Ad_k)V',
          'Understand why the residual stream preserves token identity',
          'Implement single-head attention end-to-end in the Colab notebook',
        ],
      },
    },
    {
      slug: 'multi-head-attention',
      title: 'Multi-Head Attention',
      description:
        'One attention head captures one type of relationship. Multiple heads in parallel capture many\u2014without increasing compute.',
      duration: '30 min',
      category: 'Attention',
      objectives: [
        'Explain why a single attention head can only capture one notion of relevance',
        'Describe how multi-head attention runs h independent heads in parallel, each with its own W_Q, W_K, W_V',
        'Compute d_k = d_model / h and explain dimension splitting as budget allocation, not compute multiplication',
        'Explain how concatenation + W_O combines head outputs with learned cross-head mixing',
        'Implement multi-head attention from scratch in a notebook',
      ],
      skills: [
        'multi-head-attention',
        'dimension-splitting',
        'output-projection',
        'parallel-attention-heads',
        'cross-head-mixing',
      ],
      prerequisites: ['values-and-attention-output'],
      exercise: {
        constraints: [
          'Multi-head attention mechanism only',
          'No transformer block, layer normalization, or feed-forward network\u2014that\u2019s next lesson',
          'No causal masking\u2014Lesson 6',
          'No cross-attention (self-attention only)',
          'No nn.MultiheadAttention\u2014student builds from scratch',
        ],
        steps: [
          'Feel the single-head limitation: one Q can\u2019t attend for two reasons simultaneously',
          'Understand multiple heads as parallel attention with independent projections',
          'Trace dimension splitting: d_k = d_model / h as budget allocation',
          'Hand-trace a 2-head example with d_model=6, comparing attention patterns',
          'Understand W_O as learned cross-head mixing, not just reshaping',
          'See what real heads learn in trained models (messy, emergent)',
          'Implement multi-head attention from scratch in the Colab notebook',
        ],
      },
    },
    {
      slug: 'the-transformer-block',
      title: 'The Transformer Block',
      description:
        'Multi-head attention is only one-third of the story. Assemble MHA, FFN, residual connections, and layer norm into the repeating unit that stacks to form GPT.',
      duration: '25 min',
      category: 'Attention',
      objectives: [
        'Explain how multi-head attention, a feed-forward network, residual connections, and layer normalization compose into the transformer block',
        'Describe the "attention reads, FFN writes" mental model and why both sub-layers are needed',
        'Compare layer normalization to batch normalization and explain why transformers use layer norm',
        'Explain the FFN\u2019s 4x expansion factor and why it contains more parameters than attention',
        'Distinguish pre-norm from post-norm ordering and identify which is the modern standard',
        'Explain why the block\u2019s shape preservation enables stacking',
      ],
      skills: [
        'transformer-block',
        'layer-normalization',
        'feed-forward-network',
        'residual-stream',
        'pre-norm',
        'attention-reads-ffn-writes',
      ],
      prerequisites: ['multi-head-attention'],
      exercise: {
        constraints: [
          'Transformer block assembly and mental models only',
          'No PyTorch implementation\u2014that\u2019s Module 4.3',
          'No causal masking\u2014next lesson',
          'No full decoder-only architecture\u2014next lesson',
          'No training or scaling considerations',
        ],
        steps: [
          'See the parameter split that challenges "transformer = attention"',
          'Understand layer norm by contrast with batch norm',
          'Understand the FFN\u2019s 4x expansion and its "writes" role',
          'Trace data flow through the complete block diagram',
          'Distinguish pre-norm from post-norm and know the standard',
          'Apply the "attention reads, FFN writes" mental model to stacked blocks',
        ],
      },
    },
    {
      slug: 'decoder-only-transformers',
      title: 'Decoder-Only Transformers',
      description:
        'One new mechanism\u2014causal masking\u2014then assemble every piece into the complete GPT architecture, from tokens to probabilities.',
      duration: '30 min',
      category: 'Attention',
      objectives: [
        'Explain why causal masking exists and how it prevents data leakage during parallel training',
        'Describe the causal mask as a lower-triangular matrix applied before softmax',
        'Trace the complete GPT forward pass from text input to probability distribution',
        'Count GPT-2 parameters by component and verify ~124M total',
        'Distinguish encoder-only, encoder-decoder, and decoder-only architectures',
        'Explain why decoder-only won for LLMs: simplicity, scaling, generality',
      ],
      skills: [
        'causal-masking',
        'decoder-only-transformer',
        'gpt-architecture',
        'output-projection',
        'parameter-counting',
        'encoder-decoder-contrast',
      ],
      prerequisites: ['the-transformer-block'],
      exercise: {
        constraints: [
          'Causal masking concept and full architecture assembly only',
          'No PyTorch implementation\u2014that\u2019s Module 4.3',
          'No training, loss curves, or learning rate scheduling\u2014Module 4.3',
          'No KV caching, flash attention, or efficient inference\u2014Module 4.3',
          'No finetuning, instruction tuning, or RLHF\u2014Module 4.4',
        ],
        steps: [
          'See the cheating problem: unmasked attention leaks future tokens',
          'Understand causal masking as the lower-triangular constraint',
          'Trace a worked example of masked attention computation',
          'Understand the output projection as the reverse of embedding',
          'Walk through the complete GPT forward pass end-to-end',
          'Count all parameters in GPT-2 and verify ~124M',
          'Contrast encoder-only, encoder-decoder, and decoder-only architectures',
        ],
      },
    },
  ],
}

/**
 * Building & Training GPT
 *
 * Module 4.3: Building & Training GPT
 * 1. Building nanoGPT
 * 2. Pretraining on Real Text (planned)
 * 3. Scaling & Efficiency (planned)
 * 4. Loading Pretrained Weights (planned)
 */
const buildingAndTrainingGpt: CurriculumNode = {
  slug: 'building-and-training-gpt',
  title: 'Building & Training GPT',
  children: [
    {
      slug: 'building-nanogpt',
      title: 'Building nanoGPT',
      description:
        'Translate the complete GPT architecture into working PyTorch code\u2014every component, from token embedding to generated text.',
      duration: '35 min',
      category: 'Implementation',
      objectives: [
        'Implement the full GPT architecture in PyTorch (Head, MHA, FFN, Block, GPT)',
        'Verify tensor shapes at every layer boundary',
        'Apply transformer-specific weight initialization',
        'Count parameters programmatically and verify ~124M matches GPT-2',
        'Generate text from an untrained model using the autoregressive loop',
      ],
      skills: [
        'gpt-implementation',
        'nn-module-composition',
        'shape-verification',
        'weight-initialization',
        'autoregressive-generation-code',
        'parameter-counting',
      ],
      prerequisites: ['decoder-only-transformers'],
      exercise: {
        constraints: [
          'Implementing the architecture only\u2014no training',
          'GPT-2 small configuration (124M parameters)',
          'No nn.MultiheadAttention\u2014build from scratch',
          'No dataset preparation or data loading',
          'No GPU optimization, mixed precision, or flash attention',
        ],
        steps: [
          'Define GPTConfig dataclass with GPT-2 hyperparameters',
          'Build Head class with Q, K, V projections and causal masking',
          'Build CausalSelfAttention with batched multi-head computation',
          'Build FeedForward with 4x expansion and GELU',
          'Build Block with pre-norm ordering and residual connections',
          'Build GPT with embeddings, blocks, output projection, and weight tying',
          'Apply scaled weight initialization for deep transformers',
          'Verify parameter count matches ~124M',
          'Generate text from the untrained model',
        ],
      },
    },
    {
      slug: 'pretraining',
      title: 'Pretraining on Real Text',
      description:
        'Train your GPT model from scratch\u2014watch gibberish transform into recognizable English using the same training loop you already know.',
      duration: '35 min',
      category: 'Training',
      objectives: [
        'Prepare a text dataset for language model training (tokenize, chunk, input/target offset)',
        'Explain why every position predicts its next token simultaneously during training',
        'Build the complete training loop with AdamW, LR scheduling, and gradient clipping',
        'Implement warmup + cosine decay LR scheduling and explain why each phase exists',
        'Interpret loss curves and correlate loss values with generated text quality',
      ],
      skills: [
        'text-dataset-preparation',
        'lr-scheduling',
        'gradient-clipping',
        'adamw-optimizer',
        'training-loop-language-model',
        'loss-curve-interpretation',
      ],
      prerequisites: ['building-nanogpt'],
      exercise: {
        constraints: [
          'One model (GPT-2 small), one dataset (TinyShakespeare), one training run',
          'No GPU optimization, mixed precision, or flash attention\u2014Lesson 3',
          'No hyperparameter search\u2014use known-good nanoGPT values',
          'No evaluation beyond loss and qualitative text inspection',
          'No fine-tuning or transfer learning\u2014Module 4.4',
        ],
        steps: [
          'Load and tokenize a text corpus with tiktoken',
          'Create a TextDataset with the input/target offset pattern',
          'Predict the initial loss (~10.82) and verify',
          'Build the training loop with AdamW optimizer',
          'Implement warmup + cosine decay LR scheduling',
          'Add gradient clipping',
          'Run training and observe loss curve and generated text quality',
          'Experiment: try different peak LR values and removing gradient clipping',
        ],
      },
    },
    {
      slug: 'scaling-and-efficiency',
      title: 'Scaling & Efficiency',
      description:
        'Why transformer training and inference are slow, and the engineering solutions\u2014mixed precision, KV caching, flash attention, scaling laws\u2014that make modern LLMs practical.',
      duration: '30 min',
      category: 'Training',
      objectives: [
        'Distinguish compute-bound from memory-bound operations and explain arithmetic intensity',
        'Explain mixed precision with bfloat16 and why master weights must remain in float32',
        'Describe the KV caching mechanism and compute its speedup for autoregressive generation',
        'Explain how flash attention avoids materializing the full attention matrix using tiling',
        'State the Chinchilla scaling law result: scale model size and data together',
      ],
      skills: [
        'compute-bound-vs-memory-bound',
        'mixed-precision-bfloat16',
        'kv-caching',
        'flash-attention',
        'scaling-laws',
        'arithmetic-intensity',
      ],
      prerequisites: ['pretraining'],
      exercise: {
        constraints: [
          'Conceptual lesson\u2014no implementation or notebook',
          'Not multi-GPU or distributed training',
          'Not quantization\u2014deferred to Module 4.4',
          'Not specific GPU hardware details beyond the basic insight',
          'Not inference serving optimizations in depth',
        ],
        steps: [
          'Understand compute-bound vs memory-bound with concrete examples',
          'Trace the weight update example to see why mixed precision is "mixed"',
          'Walk through KV cache growth step by step during generation',
          'Compute the speedup from KV caching for a 100-token generation',
          'Understand flash attention as tiled computation, not a different algorithm',
          'Apply Chinchilla scaling insight to a hypothetical compute budget allocation',
        ],
      },
    },
    {
      slug: 'loading-real-weights',
      title: 'Loading Real Weights',
      description:
        'Load OpenAI\u2019s pretrained GPT-2 weights into the model you built from scratch\u2014and prove your implementation is correct.',
      duration: '30 min',
      category: 'Implementation',
      objectives: [
        'Load pretrained GPT-2 weights from HuggingFace into your own GPT architecture',
        'Build a weight name mapping handling Conv1D vs nn.Linear transposition',
        'Handle weight tying during loading (embedding shared with lm_head)',
        'Verify correctness by comparing logits against the HuggingFace reference model',
        'Generate coherent text from the correctly loaded model',
      ],
      skills: [
        'weight-loading',
        'weight-name-mapping',
        'conv1d-transpose',
        'logit-verification',
        'huggingface-basics',
        'pretrained-model-usage',
      ],
      prerequisites: ['scaling-and-efficiency'],
      exercise: {
        constraints: [
          'Loading GPT-2 small (124M) only\u2014other sizes as stretch exercise',
          'HuggingFace used only as a weight source and reference',
          'No training or fine-tuning the loaded model\u2014Module 4.4',
          'No quantization, deployment, or KV caching implementation',
        ],
        steps: [
          'Download GPT-2 weights via HuggingFace transformers',
          'Compare state dict keys between HuggingFace and your model',
          'Build the weight mapping function with Conv1D transposition handling',
          'Handle weight tying (lm_head shares with wte)',
          'See the negative example: loading without transposing produces garbage',
          'Verify with logit comparison using torch.allclose',
          'Generate coherent text from the correctly loaded model',
          'Stretch: load GPT-2 medium (345M) by updating the config',
        ],
      },
    },
  ],
}

/**
 * Beyond Pretraining
 *
 * Module 4.4: Beyond Pretraining
 * 1. Finetuning for Classification
 * 2. Instruction Tuning (planned)
 * 3. RLHF & Alignment (planned)
 * 4. Parameter-Efficient Finetuning (planned)
 * 5. Evaluation & Benchmarks (planned)
 */
const beyondPretraining: CurriculumNode = {
  slug: 'beyond-pretraining',
  title: 'Beyond Pretraining',
  children: [
    {
      slug: 'finetuning-for-classification',
      title: 'Finetuning for Classification',
      description:
        'Adapt your pretrained GPT-2 for text classification\u2014the same transfer learning pattern you used with CNNs, applied to a language model.',
      duration: '30 min',
      category: 'Fine-tuning',
      objectives: [
        'Add a classification head to pretrained GPT-2',
        'Explain why the last token\u2019s hidden state is the correct sequence representation for causal models',
        'Train a frozen-backbone classifier on a sentiment dataset',
        'Compare frozen vs unfrozen finetuning tradeoffs',
        'Connect transformer transfer learning to the CNN transfer learning pattern',
      ],
      skills: [
        'classification-head',
        'sequence-representation',
        'last-token-pooling',
        'frozen-backbone-training',
        'transformer-transfer-learning',
        'catastrophic-forgetting',
      ],
      prerequisites: ['loading-real-weights'],
      exercise: {
        constraints: [
          'Classification finetuning only\u2014no instruction tuning, LoRA, or RLHF',
          'SST-2 sentiment classification (binary)',
          'Manual training loop\u2014no HuggingFace Trainer API',
          'Frozen vs unfrozen comparison introduced, not deeply developed',
        ],
        steps: [
          'Load pretrained GPT-2 via HuggingFace',
          'Tokenize SST-2 examples with tiktoken',
          'Implement the classification head (nn.Linear on last token)',
          'Freeze backbone and train the head',
          'Evaluate accuracy on held-out data',
          'Unfreeze last N transformer blocks with differential LR',
          'Generate text before and after finetuning to observe catastrophic forgetting',
        ],
      },
    },
    {
      slug: 'instruction-tuning',
      title: 'Instruction Tuning (SFT)',
      description:
        'How a dataset of instruction-response pairs transforms a text completer into an instruction follower\u2014using the same training loop you already know.',
      duration: '35 min',
      category: 'Fine-tuning',
      objectives: [
        'Explain how supervised finetuning on instruction-response pairs transforms a base model into an instruction follower',
        'Describe instruction dataset format and why SFT teaches format, not knowledge',
        'Explain chat templates and special tokens as functional structural delimiters',
        'Implement loss masking to focus training signal on response tokens',
        'Perform SFT on a small instruction dataset in a notebook',
        'Contrast SFT with classification finetuning: same head vs new head, broad task vs narrow task',
      ],
      skills: [
        'supervised-finetuning',
        'instruction-datasets',
        'chat-templates',
        'special-tokens',
        'loss-masking',
        'sft-training',
      ],
      prerequisites: ['finetuning-for-classification'],
      exercise: {
        constraints: [
          'SFT on instruction-response pairs only',
          'No RLHF, DPO, or alignment\u2014that\u2019s the next lesson',
          'No LoRA or parameter-efficient finetuning\u2014Lesson 4',
          'No production-quality dataset curation',
          'No evaluation beyond qualitative observation',
        ],
        steps: [
          'Explore an instruction dataset (Alpaca format)',
          'Implement chat template formatting with special tokens',
          'Tokenize formatted examples and inspect token IDs',
          'Implement loss masking (labels = -100 for prompt tokens)',
          'Run SFT training for a small number of steps',
          'Compare base model vs SFT model on several prompts',
        ],
      },
    },
    {
      slug: 'rlhf-and-alignment',
      title: 'RLHF & Alignment',
      description:
        'Why instruction-following is not enough, and how human preferences become the training signal that gives language models judgment.',
      duration: '25 min',
      category: 'Fine-tuning',
      objectives: [
        'Explain why SFT alone produces models that can be harmful, sycophantic, or confidently wrong',
        'Describe human preference data format: comparison pairs, not absolute scores',
        'Explain how a reward model (pretrained LM + scalar head) is trained on preference data',
        'Describe the PPO training loop: generate, score, update with KL penalty',
        'Explain DPO as a simpler alternative that achieves similar results without a separate reward model',
        'Identify reward hacking as the failure mode that motivates the KL constraint',
      ],
      skills: [
        'alignment-problem',
        'human-preference-data',
        'reward-models',
        'ppo-intuition',
        'dpo',
        'reward-hacking',
        'kl-penalty',
      ],
      prerequisites: ['instruction-tuning'],
      exercise: {
        constraints: [
          'Conceptual lesson\u2014no implementation or notebook',
          'PPO at intuitive level only\u2014no algorithm details',
          'No RL formalism beyond minimum needed (policy = model behavior, reward = score)',
          'No constitutional AI or RLAIF\u2014Series 5',
          'No red teaming, adversarial evaluation, or safety benchmarks',
        ],
        steps: [
          'See three concrete SFT failure modes (harmful, sycophantic, confidently wrong)',
          'Understand why cross-entropy cannot express response quality',
          'Read a concrete preference comparison pair',
          'Predict the reward model architecture (callback to classification finetuning)',
          'Trace the PPO generate-score-update loop',
          'Understand why the KL penalty prevents reward hacking',
          'Compare PPO and DPO pipelines',
        ],
      },
    },
    {
      slug: 'lora-and-quantization',
      title: 'LoRA, Quantization & Inference',
      description:
        'The two techniques that take LLMs from \u201crequires a cluster\u201d to \u201cruns on your laptop\u201d\u2014efficient finetuning with LoRA and efficient inference with quantization.',
      duration: '35 min',
      category: 'Fine-tuning',
      objectives: [
        'Break down training memory into weights, gradients, and optimizer states with concrete arithmetic',
        'Explain low-rank decomposition: a large matrix factored into two smaller matrices',
        'Describe the LoRA architecture: frozen base weights + trainable low-rank bypass',
        'Trace absmax and zero-point quantization with specific numbers',
        'Explain why neural network weights tolerate quantization (redundancy, Gaussian distribution)',
        'Describe QLoRA as the combination of quantized base model + LoRA adapters',
      ],
      skills: [
        'training-memory-breakdown',
        'low-rank-decomposition',
        'lora-architecture',
        'lora-rank-hyperparameter',
        'absmax-quantization',
        'zero-point-quantization',
        'qlora',
        'parameter-efficient-finetuning',
      ],
      prerequisites: ['rlhf-and-alignment'],
      exercise: {
        constraints: [
          'LoRA and quantization mechanics only',
          'No SVD, eigenvalues, or formal linear algebra beyond rank intuition',
          'No quantization-aware training\u2014post-training quantization only',
          'No pruning, distillation, mixture of experts, or production deployment',
          'PEFT library for practical LoRA, not full from-scratch implementation',
        ],
        steps: [
          'Calculate memory requirements for inference and training at different precisions',
          'Implement a LoRALinear layer from scratch and verify frozen base weights',
          'Apply absmax quantization step by step and compute reconstruction error',
          'LoRA-finetune a model using the HuggingFace PEFT library',
          'Load a quantized model and compare memory, speed, and quality vs full precision',
        ],
      },
    },
    {
      slug: 'putting-it-all-together',
      title: 'Putting It All Together',
      description:
        'The complete LLM pipeline from raw text to aligned model on your laptop\u2014no new concepts, just synthesis of everything you have learned across Series 4.',
      duration: '20 min',
      category: 'Synthesis',
      objectives: [
        'Trace the complete LLM pipeline from raw text to deployed model',
        'Explain what each pipeline stage adds and why no stage can be skipped',
        'Map pipeline stages to real open-source model artifacts (base, instruct, quantized)',
        'Match practical scenarios to the appropriate adaptation method',
        'Articulate the adaptation spectrum from classification head to full finetuning',
      ],
      skills: [
        'llm-pipeline',
        'pipeline-dependencies',
        'model-ecosystem',
        'adaptation-spectrum',
        'series-synthesis',
      ],
      prerequisites: ['lora-and-quantization'],
    },
  ],
}

/**
 * LLMs & Transformers
 *
 * Series 4: Understanding transformer-based language models
 * from fundamentals through building GPT from scratch.
 */
export const llms: CurriculumNode = {
  slug: 'llms',
  title: 'LLMs & Transformers',
  icon: 'MessageSquare',
  description:
    'Language models, attention, and transformers\u2014from next-token prediction to building GPT from scratch',
  children: [
    languageModelingFundamentals,
    attentionAndTransformer,
    buildingAndTrainingGpt,
    beyondPretraining,
  ],
}
