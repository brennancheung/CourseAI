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
  children: [advancedAlignment],
}
