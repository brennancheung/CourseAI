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
  children: [advancedAlignment, reasoningAndInContextLearning],
}
