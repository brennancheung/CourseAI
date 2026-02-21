import { CurriculumNode } from './types'

/**
 * Advanced Alignment
 *
 * Module 5.1: Advanced Alignment
 * 1. Constitutional AI
 * 2. Alignment Techniques Landscape
 * 3. Red Teaming & Adversarial Evaluation (planned)
 * 4. Alignment Evaluation & Benchmarks (planned)
 */
const advancedAlignment: CurriculumNode = {
  slug: 'advanced-alignment',
  title: 'Advanced Alignment',
  children: [
    {
      slug: 'constitutional-ai',
      title: 'Constitutional AI',
      description:
        'How explicit principles replace human annotators as the source of alignment data\u2014and why this changes the scaling equation for alignment.',
      duration: '25 min',
      category: 'Alignment',
      objectives: [
        'Explain how constitutional AI replaces human preference annotators with AI-generated feedback guided by explicit principles',
        'Describe the critique-and-revision mechanism that generates improved SFT training data',
        'Explain RLAIF: AI-generated preference labels replacing human labels in the RL training stage',
        'Identify that the constitution is used during training, not at inference time',
        'Articulate why principle design is the new alignment challenge (vague, conflicting, or missing principles)',
      ],
      skills: [
        'constitutional-ai',
        'critique-and-revision',
        'rlaif',
        'ai-feedback',
        'principle-based-alignment',
        'alignment-scaling',
      ],
      prerequisites: ['rlhf-and-alignment'],
      exercise: {
        constraints: [
          'Conceptual lesson\u2014no heavy implementation',
          'Constitutional AI mechanism and motivation only',
          'No DPO variations or other alignment techniques\u2014next lesson',
          'No red teaming, adversarial evaluation, or safety benchmarks',
          'No specific principles used by any particular company',
        ],
        steps: [
          'Reconnect RLHF mental models from Series 4 (editor analogy, preference pairs)',
          'See the human bottleneck problem: cost, consistency, and scale',
          'Understand constitutional principles as explicit alignment criteria',
          'Trace the critique-and-revision mechanism with a concrete example',
          'Understand RLAIF as replacing human annotators with AI applying principles',
          'Identify failure modes: vague, conflicting, and missing principles',
        ],
      },
    },
    {
      slug: 'alignment-techniques-landscape',
      title: 'The Alignment Techniques Landscape',
      description:
        'PPO and DPO are not the only options\u2014the design space of preference optimization is wider than you think, and the right choice depends on your constraints.',
      duration: '25 min',
      category: 'Alignment',
      objectives: [
        'Map the preference optimization design space along four axes: data format, reference model, online vs offline, reward model',
        'Explain what IPO, KTO, and ORPO each change relative to DPO and why',
        'Distinguish online from offline preference optimization and articulate the tradeoffs',
        'Use the design space framework to classify a new alignment technique by its tradeoffs',
        'Recognize that alignment methods are tradeoffs along axes, not linear improvements',
      ],
      skills: [
        'preference-optimization-landscape',
        'ipo',
        'kto',
        'orpo',
        'online-vs-offline-alignment',
        'iterative-alignment',
        'design-space-analysis',
      ],
      prerequisites: ['constitutional-ai'],
      exercise: {
        constraints: [
          'Conceptual lesson\u2014no loss function derivations or code',
          'Design space mapping and tradeoff reasoning only',
          'No benchmarking or performance comparisons between methods',
          'No constitutional AI (previous lesson) or red teaming (next lesson)',
          'No RL formalism beyond minimum needed',
        ],
        steps: [
          'Reconnect PPO and DPO mental models from RLHF & Alignment',
          'See three scenarios that break the PPO/DPO binary',
          'Learn the four design space axes as a classification framework',
          'Place IPO, KTO, and ORPO on the map relative to DPO',
          'Understand online vs offline preference optimization and iterative alignment',
          'Use the map to classify a novel method and choose methods for concrete scenarios',
        ],
      },
    },
    {
      slug: 'red-teaming-and-adversarial-evaluation',
      title: 'Red Teaming & Adversarial Evaluation',
      description:
        'How do you find where alignment breaks? Red teaming is the systematic search for failures\u2014and the reason alignment is never done.',
      duration: '25 min',
      category: 'Alignment',
      objectives: [
        'Explain red teaming as systematic adversarial probing (not just jailbreaks)',
        'Classify adversarial attacks into six categories by the mechanism they exploit',
        'Articulate three structural reasons why aligned models fail (surface patterns, distribution gaps, capability-safety tension)',
        'Explain why automated red teaming is necessary and how it parallels RLAIF',
        'Describe the attack-defense dynamic and why alignment is an ongoing process, not a one-time fix',
        'Explain defense-in-depth as a multi-layer approach to alignment robustness',
      ],
      skills: [
        'red-teaming',
        'adversarial-evaluation',
        'attack-taxonomy',
        'automated-red-teaming',
        'defense-in-depth',
        'alignment-surface',
        'capability-safety-tension',
      ],
      prerequisites: ['alignment-techniques-landscape'],
      exercise: {
        constraints: [
          'Conceptual lesson\u2014no implementing attacks in code',
          'Patterns and mechanisms, not specific current jailbreak recipes',
          'No benchmarks or evaluation metrics (next lesson)',
          'No constitutional AI or preference optimization details (previous lessons)',
        ],
        steps: [
          'Reconnect alignment concepts from Lessons 1-2 (blind spots, principle failures)',
          'See three failures of a "well-aligned" model that passed safety tests',
          'Learn red teaming as systematic adversarial probing (pen-testing analogy)',
          'Classify attacks into six categories by mechanism exploited',
          'Understand three structural reasons why aligned models fail',
          'See how automated red teaming scales adversarial testing (RLAIF parallel)',
          'Predict defenses and their costs for a concrete attack',
          'Trace the attack-defense cycle and understand defense-in-depth',
        ],
      },
    },
    {
      slug: 'evaluating-llms',
      title: 'Evaluating LLMs',
      description:
        'Benchmarks are not what they appear to be\u2014how contamination, Goodhart\u2019s law, and the proxy gap undermine LLM evaluation, and why measuring alignment may be harder than building it.',
      duration: '25 min',
      category: 'Alignment',
      objectives: [
        'Explain the proxy gap between what benchmarks measure and what they claim to measure',
        'Describe contamination as a structural property of internet-scale training, not just data leakage',
        'Apply Goodhart\u2019s law to evaluation: when a benchmark becomes an optimization target, it ceases to be a good measure',
        'Articulate why human evaluation is imperfect (cost, consistency, scale, bias) and how Chatbot Arena partially addresses these',
        'Explain LLM-as-judge as a scaling strategy and identify its systematic biases (verbosity, confidence, self-preference, format)',
        'Argue why evaluation may be fundamentally harder than training',
      ],
      skills: [
        'benchmark-evaluation',
        'contamination',
        'goodharts-law',
        'human-evaluation',
        'llm-as-judge',
        'evaluation-stack',
        'proxy-gap',
      ],
      prerequisites: ['red-teaming-and-adversarial-evaluation'],
      exercise: {
        constraints: [
          'Conceptual lesson\u2014no implementing evaluation systems',
          'Critical assessment of evaluation methods, not specific benchmark scores',
          'No constitutional AI, preference optimization, or red teaming details (previous lessons)',
          'No statistical methodology for evaluation',
        ],
        steps: [
          'Reconnect alignment concepts from Lessons 1-3 and reward hacking from Series 4',
          'See how benchmark rankings can contradict real-world user preferences',
          'Understand the evaluation stack: layers between capability and leaderboard number',
          'Learn contamination as a structural, not accidental, problem',
          'Apply Goodhart\u2019s law from reward hacking to evaluation metrics',
          'Assess human evaluation limitations (callback to annotation bottleneck)',
          'Evaluate LLM-as-judge approach and its systematic biases',
          'Design an evaluation strategy that combines multiple methods',
        ],
      },
    },
  ],
}

/**
 * Reasoning & In-Context Learning
 *
 * Module 5.2: Reasoning & In-Context Learning
 * 1. In-Context Learning
 * 2. Prompt Engineering (planned)
 * 3. Chain-of-Thought (planned)
 * 4. Reasoning Models (planned)
 */
const reasoningAndInContextLearning: CurriculumNode = {
  slug: 'reasoning-and-in-context-learning',
  title: 'Reasoning & In-Context Learning',
  children: [
    {
      slug: 'in-context-learning',
      title: 'In-Context Learning',
      description:
        'A model trained only on next-token prediction can learn new tasks from examples in the prompt\u2014without any weight update. The mechanism is attention, and it changes how you think about what "learning" means.',
      duration: '25 min',
      category: 'Reasoning & ICL',
      objectives: [
        'Explain why a transformer can learn new tasks from examples in the prompt without weight updates',
        'Identify attention as the specific mechanism that enables in-context learning',
        'Distinguish zero-shot from few-shot prompting and explain when each applies',
        'Articulate why ICL is a base model capability (not a product of SFT or RLHF)',
        'Describe the limitations of ICL: ordering sensitivity, context window constraints, and fragility',
        'Connect ICL to the "capability = vulnerability" pattern from red teaming',
      ],
      skills: [
        'in-context-learning',
        'few-shot-prompting',
        'zero-shot-prompting',
        'attention-as-icl-mechanism',
        'icl-limitations',
      ],
      prerequisites: ['evaluating-llms'],
      exercise: {
        constraints: [
          'Conceptual lesson with notebook exercises',
          'ICL phenomenon and attention-based mechanism only',
          'No systematic prompt engineering (next lesson)',
          'No chain-of-thought reasoning (Lesson 3)',
          'No retrieval-augmented generation',
          'No implementing ICL from scratch',
        ],
        steps: [
          'Reconnect attention as data-dependent computation and finetuning for classification',
          'See few-shot sentiment classification with no weight updates (the puzzle)',
          'Learn the GPT-3 discovery: ICL as a base model capability',
          'Trace the attention mechanism that enables ICL (QK matching, V retrieval)',
          'Verify "the prompt is a program" with same input, different examples, different outputs',
          'Explore ICL boundaries: novel mappings (works), ordering sensitivity (fragile)',
          'Connect ICL to few-shot jailbreaking (capability = vulnerability)',
          'Practice: zero-shot vs few-shot, novel tasks, ordering experiments, ICL vs finetuning',
        ],
      },
    },
    {
      slug: 'prompt-engineering',
      title: 'Prompt Engineering',
      description:
        'If the prompt is a program, then prompt engineering is programming\u2014selecting and combining format specification, role framing, example selection, and context augmentation to reliably control model behavior.',
      duration: '25 min',
      category: 'Reasoning & ICL',
      objectives: [
        'Treat prompt construction as structured programming with identifiable, composable components',
        'Use format specification to constrain the output distribution for consistent, parseable results',
        'Explain what role/system prompts do mechanistically (bias attention, not add knowledge)',
        'Apply few-shot example selection principles: diversity over quantity, format consistency',
        'Describe RAG as retrieval-augmented context grounded in the attention mechanism',
        'Reason about which prompt components to use for a given task based on the attention mechanism',
      ],
      skills: [
        'prompt-engineering',
        'format-specification',
        'role-prompting',
        'system-prompts',
        'few-shot-example-selection',
        'rag-overview',
        'structured-prompting',
      ],
      prerequisites: ['in-context-learning'],
      exercise: {
        constraints: [
          'Practical BUILD lesson with notebook exercises',
          'Prompt engineering techniques grounded in attention mechanism',
          'RAG conceptual overview only\u2014no pipeline implementation',
          'No chain-of-thought prompting (next lesson)',
          'No prompt optimization tools (DSPy, automated search)',
          'No agentic patterns or tool use',
        ],
        steps: [
          'See the before/after contrast: conversational vs structured prompting',
          'Learn prompt anatomy: six composable components with software engineering analogies',
          'Understand format specification as output distribution constraint',
          'Understand role prompts as attention bias (not knowledge addition)',
          'Apply few-shot example selection principles: diversity, format consistency, difficulty',
          'Predict which components matter most for a summarization task',
          'Learn RAG as retrieval-augmented context for attention',
          'See the context stuffing failure: more is not always better',
          'Design a complete prompt system for a customer support chatbot',
          'Practice: format specification, role effects, example selection, structured prompt design',
        ],
      },
    },
    {
      slug: 'chain-of-thought',
      title: 'Chain-of-Thought Reasoning',
      description:
        'Intermediate reasoning tokens give the model additional forward passes worth of computation\u2014expanding its effective capacity beyond what a single forward pass provides. The mechanism is the autoregressive loop you already understand.',
      duration: '25 min',
      category: 'Reasoning & ICL',
      objectives: [
        'Explain why generating intermediate reasoning tokens gives the model more computation (each token triggers another forward pass through N transformer blocks)',
        'Identify the fixed computation budget per forward pass as the architectural constraint CoT addresses',
        'Distinguish when CoT helps (multi-step problems exceeding single-pass capacity) from when it does not (factual recall, classification)',
        'Explain why CoT quality matters more than quantity (error propagation, noise diluting attention)',
        'Distinguish zero-shot CoT from few-shot CoT and connect each to prior lessons (prompt engineering and ICL)',
        'Describe the difference between process supervision and outcome supervision for evaluating reasoning chains',
      ],
      skills: [
        'chain-of-thought',
        'tokens-as-computation',
        'zero-shot-cot',
        'few-shot-cot',
        'process-supervision',
        'outcome-supervision',
        'cot-limitations',
      ],
      prerequisites: ['prompt-engineering'],
      exercise: {
        constraints: [
          'STRETCH lesson with notebook exercises',
          'CoT mechanism and when it helps\u2014grounded in autoregressive loop and fixed architecture',
          'Process vs outcome supervision introduced, not developed',
          'No reasoning models trained with RL (next lesson)',
          'No test-time compute scaling (next lesson)',
          'No search during inference, tree-of-thought, or beam search (next lesson)',
          'No self-consistency implementation details (mentioned only)',
        ],
        steps: [
          'See 17 \u00d7 24 with and without CoT\u2014same model, different answer (the puzzle)',
          'Understand the fixed computation budget: same N blocks for every token prediction',
          'Learn that intermediate tokens are additional forward passes (tokens as computation)',
          'Address misconceptions: not thinking, not showing internal work',
          'See when CoT helps (multi-step) and when it does not (factual recall)',
          'See error propagation: wrong intermediate step \u2192 wrong final answer',
          'Learn zero-shot CoT vs few-shot CoT and connect to prior lessons',
          'Understand process supervision vs outcome supervision for evaluating chains',
          'Practice: direct vs CoT comparison, token counting, error propagation, find the boundary',
        ],
      },
    },
    {
      slug: 'reasoning-models',
      title: 'Reasoning Models',
      description:
        'Base model chain-of-thought is unreliable\u2014sometimes the reasoning is correct, sometimes it is not. Reasoning models apply the same RL from RLHF with a different reward signal: answer correctness. Once a model can reason reliably, a new scaling axis opens: instead of bigger models, let the model think longer.',
      duration: '25 min',
      category: 'Reasoning & ICL',
      objectives: [
        'Explain how RL trains models to generate effective reasoning chains (same loop as RLHF, different reward signal)',
        'Distinguish outcome reward models (ORMs) from process reward models (PRMs) and explain why process supervision trains better reasoning',
        'Describe test-time compute scaling as a new scaling axis: trading inference compute for model size',
        'Explain self-consistency and best-of-N as search strategies during inference',
        'Articulate the paradigm shift from "scale the model" to "scale the inference compute"',
        'Identify when reasoning models help (multi-step problems) and when they waste compute (simple tasks)',
      ],
      skills: [
        'reasoning-models',
        'rl-for-reasoning',
        'process-reward-models',
        'outcome-reward-models',
        'test-time-compute-scaling',
        'self-consistency',
        'best-of-n',
        'inference-scaling',
      ],
      prerequisites: ['chain-of-thought'],
      exercise: {
        constraints: [
          'BUILD lesson with notebook exercises\u2014module capstone',
          'RL for reasoning mechanism (conceptual, not implementation)',
          'ORM vs PRM distinction developed from prior INTRODUCED depth',
          'Test-time compute scaling as paradigm shift',
          'Self-consistency developed from prior MENTIONED depth',
          'No implementing RL for reasoning in code',
          'No specific model architectures or training details (o1, DeepSeek-R1, etc.)',
          'No tree-of-thought or MCTS implementation details',
          'No distillation of reasoning models',
        ],
        steps: [
          'Reconnect tokens-as-computation (CoT) and RL training (RLHF)',
          'See before/after: base model CoT vs reasoning model on same problems',
          'Learn RL for reasoning: same loop as RLHF, reward is answer correctness',
          'Predict what goes wrong with outcome-only reward (reward hacking pattern)',
          'Develop process supervision from INTRODUCED: ORM vs PRM with concrete example',
          'Learn test-time compute scaling: two axes of scaling (model size vs inference compute)',
          'Develop self-consistency from MENTIONED: worked example with 5 chains and majority vote',
          'Apply concepts to a deployment scenario (hybrid compute allocation)',
          'Practice: base vs reasoning comparison, self-consistency experiment, process vs outcome evaluation, adaptive compute allocation',
        ],
      },
    },
  ],
}

/**
 * Scaling Architecture
 *
 * Module 5.3: Scaling Architecture
 * 1. Mixture of Experts
 * 2. Long Context & Efficient Attention (planned)
 * 3. Training & Serving at Scale (planned)
 */
const scalingArchitecture: CurriculumNode = {
  slug: 'scaling-architecture',
  title: 'Scaling Architecture',
  children: [
    {
      slug: 'mixture-of-experts',
      title: 'Mixture of Experts',
      description:
        'Every dense transformer activates all parameters for every token\u2014even when most FFN knowledge is irrelevant. MoE replaces the monolithic FFN with specialized sub-networks and a learned router, decoupling total parameters from per-token computation.',
      duration: '25 min',
      category: 'Scaling Architecture',
      objectives: [
        'Explain how MoE replaces the monolithic FFN with multiple expert FFNs and a learned router',
        'Describe the router mechanism as a single linear layer with softmax and top-k selection (same pattern as attention)',
        'Articulate the parameter-compute decoupling: total parameters vs active parameters per token',
        'Explain why expert specialization is emergent and per-token, not designed and per-topic',
        'Describe the load balancing problem (router collapse) and why auxiliary loss is necessary',
        'Use Mixtral 8\u00d77B as a concrete example of the MoE tradeoffs (47B total, ~13B active)',
      ],
      skills: [
        'mixture-of-experts',
        'conditional-computation',
        'moe-router',
        'expert-specialization',
        'load-balancing',
        'parameter-compute-decoupling',
        'sparse-models',
      ],
      prerequisites: ['reasoning-models'],
      exercise: {
        constraints: [
          'STRETCH lesson with notebook exercises',
          'Conditional computation concept and MoE architecture',
          'Router mechanism grounded in attention\u2019s dot-product-softmax pattern',
          'Load balancing introduced (problem + solution concept), not mathematically developed',
          'No implementing production MoE from scratch',
          'No communication overhead across GPUs (deferred to Lesson 3)',
          'No Switch Transformer or historical MoE variants in depth',
          'No specific training recipes or hyperparameters',
        ],
        steps: [
          'Reconnect FFN as knowledge store, parameter distribution, and key-value memories',
          'See the waste problem: all 117B FFN parameters active for every token in GPT-3',
          'Learn MoE architecture: N expert FFNs + router replacing the single FFN',
          'Demystify the router: single linear layer + softmax + top-k (same as attention)',
          'Understand parameter-compute decoupling with Mixtral 8\u00d77B concrete numbers',
          'Predict per-token compute for top-1 routing on a French translation task',
          'See per-token routing on "The mitochondria is the powerhouse of the cell"',
          'Learn load balancing: router collapse feedback loop and auxiliary loss solution',
          'Apply concepts to 13B dense vs 47B MoE training budget comparison',
          'Practice: implement router, MoE forward pass, routing visualization, collapse experiment',
        ],
      },
    },
    {
      slug: 'long-context-and-efficient-attention',
      title: 'Long Context & Efficient Attention',
      description:
        'The attention formula is quadratic in sequence length, positional encoding fails beyond training length, and KV cache memory explodes\u2014three distinct bottlenecks that require three targeted innovations: RoPE, sparse attention, and GQA.',
      duration: '25 min',
      category: 'Scaling Architecture',
      objectives: [
        'Explain how RoPE encodes relative position into the Q/K dot product via rotation in 2D subspaces',
        'Articulate why RoPE enables context extension beyond training length (relative position patterns transfer)',
        'Compute the quadratic attention cost at different sequence lengths and explain why flash attention does not fix it',
        'Describe sliding window and dilated attention patterns and explain the expressiveness-efficiency tradeoff',
        'Explain how stacked layers of sparse attention recover long-range information through the residual stream',
        'Describe GQA as sharing K/V across Q head groups, and place MHA, GQA, and MQA on a spectrum',
        'Compute KV cache savings for GQA vs MHA at production scale',
      ],
      skills: [
        'rope',
        'rotary-position-embeddings',
        'sparse-attention',
        'sliding-window-attention',
        'dilated-attention',
        'linear-attention',
        'grouped-query-attention',
        'gqa',
        'mqa',
        'kv-cache-optimization',
        'context-extension',
        'long-context',
      ],
      prerequisites: ['mixture-of-experts'],
      exercise: {
        constraints: [
          'BUILD lesson with notebook exercises',
          'Three barriers and three solutions\u2014conceptual with concrete calculations',
          'RoPE mechanism developed, context extension introduced',
          'Sparse attention patterns developed, linear attention introduced',
          'GQA developed with concrete KV cache calculations',
          'No implementing production RoPE, sparse attention, or GQA',
          'No NTK-aware scaling or YaRN formulas in detail',
          'No ring attention or sequence parallelism (Lesson 3)',
          'No state space models (SSMs) or Mamba',
        ],
        steps: [
          'Reconnect attention formula, positional encoding limits, and flash attention/KV caching',
          'See the three walls: position encoding failure, quadratic compute, KV cache explosion',
          'Learn RoPE: rotation in 2D subspaces, relative position in the dot product',
          'Verify RoPE understanding: learned PE vs RoPE on an 8K document',
          'Learn sparse attention: sliding window and dilated patterns for subquadratic compute',
          'See how stacked layers recover long-range information through the residual stream',
          'Learn GQA: sharing K/V across Q head groups, the MHA-GQA-MQA spectrum',
          'Apply all three to a 64K legal document analysis system',
          'Practice: RoPE rotation, sparse masks, GQA forward pass, cost calculator',
        ],
      },
    },
    {
      slug: 'training-and-serving-at-scale',
      title: 'Training & Serving at Scale',
      description:
        'A 70B model needs 840 GB of training memory\u2014the best GPU has 80 GB. This lesson closes the gap between architecture and deployment: parallelism strategies for training, speculative decoding and continuous batching for serving.',
      duration: '25 min',
      category: 'Scaling Architecture',
      objectives: [
        'Explain data parallelism (gradient all-reduce) and identify when it is sufficient vs when model parallelism is needed',
        'Distinguish tensor parallelism (split within layers) from pipeline parallelism (split across layers) by communication pattern and use case',
        'Articulate communication overhead as the central constraint shaping all parallelism decisions',
        'Describe ZeRO optimizer state sharding and explain why it targets the dominant memory cost',
        'Trace the speculative decoding draft-verify loop and explain why the speed comes from parallel verification, not the draft model',
        'Explain continuous batching slot management and why it improves GPU utilization over static batching',
      ],
      skills: [
        'data-parallelism',
        'tensor-parallelism',
        'pipeline-parallelism',
        'communication-overhead',
        'zero-optimizer',
        'speculative-decoding',
        'continuous-batching',
        'distributed-training',
        'inference-serving',
      ],
      prerequisites: ['long-context-and-efficient-attention'],
      exercise: {
        constraints: [
          'CONSOLIDATE lesson with notebook exercises\u2014module capstone',
          'Three parallelism strategies developed with concrete arithmetic',
          'ZeRO introduced (concept and motivation, not implementation)',
          'Speculative decoding developed from MENTIONED depth',
          'Continuous batching developed from MENTIONED depth',
          'No implementing parallelism in code (no PyTorch distributed, FSDP, DeepSpeed)',
          'No NCCL, MPI, or communication primitives',
          'No ring attention or sequence parallelism (mentioned only)',
          'No specific serving frameworks (vLLM, TGI mentioned only)',
        ],
        steps: [
          'Reconnect training memory breakdown, generate() loop, and compute-bound vs memory-bound',
          'See the scale wall: 70B model needs 840 GB, best GPU has 80 GB',
          'Learn data parallelism: replicate model, split data, all-reduce gradients',
          'Establish communication overhead as the central constraint (delivery truck at scale)',
          'Learn tensor parallelism: split weight matrices within layers, all-reduce per layer',
          'Learn pipeline parallelism: assign layers to GPUs, microbatching to fill the pipeline',
          'Compare all three strategies side by side (what is split, communication pattern, when to use)',
          'Predict parallelism strategy for a 30B model on 4 GPUs',
          'Learn ZeRO: shard optimizer states to cut dominant memory cost',
          'Learn speculative decoding: draft-verify loop modifying the generate() method',
          'Learn continuous batching: slot management for high-throughput serving',
          'Apply to Mixtral deployment scenario (MoE + parallelism + serving)',
          'Practice: memory calculator, speculative decoding simulation, batching simulation, strategy advisor',
        ],
      },
    },
  ],
}

/**
 * Recent LLM Advances
 *
 * Series 5: Extending the LLM foundation with modern techniques
 * including advanced alignment, reasoning, and multimodal capabilities.
 */
export const recentLlmAdvances: CurriculumNode = {
  slug: 'recent-llm-advances',
  title: 'Recent LLM Advances',
  icon: 'Sparkles',
  description:
    'Advanced alignment, reasoning models, and multimodal capabilities\u2014extending the LLM foundation into its modern form',
  children: [advancedAlignment, reasoningAndInContextLearning, scalingArchitecture],
}
