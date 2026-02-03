/**
 * Learner State
 *
 * This file is updated by Claude Code after review conversations.
 * It captures the current state of skill development, active struggles,
 * and patterns discovered through learning sessions.
 *
 * Don't update this on every session — only when there's a meaningful shift.
 */

export type SkillLevel = 'unexplored' | 'learning' | 'practicing' | 'comfortable'

export type SkillAssessment = {
  level: SkillLevel
  notes: string
  lastPracticed?: string // ISO date
}

export type Struggle = {
  skill: string
  issue: string
  since: string // ISO date
  resolved?: string // ISO date if resolved
}

export type LearnerState = {
  lastUpdated: string // ISO date

  // Skill assessments from review conversations
  skills: Record<string, SkillAssessment>

  // Active struggles to address in upcoming lessons
  struggles: Struggle[]

  // Patterns noticed across sessions
  patterns: string[]

  // Preferences discovered
  preferences: {
    preferredTopics?: string[]
    sessionLength?: string
    notes?: string
  }
}

/**
 * Current learner state
 *
 * Updated by Claude Code based on review conversations.
 * Initial state reflects baseline from CLAUDE.md.
 */
export const learnerState: LearnerState = {
  lastUpdated: '2026-02-02',

  skills: {
    // Deep Learning Fundamentals
    'ml-intuition': {
      level: 'learning',
      notes: 'Previous DL course (pre-transformer era). Understands concepts but rusty on details.',
    },
    'linear-models': {
      level: 'learning',
      notes: 'Knows the basics. Needs refresher on implementation.',
    },
    'loss-functions': {
      level: 'unexplored',
      notes: '',
    },
    'gradient-descent': {
      level: 'unexplored',
      notes: '',
    },
    'backprop': {
      level: 'unexplored',
      notes: 'Learned before but forgotten. Needs animated walkthrough.',
    },
    'activation-functions': {
      level: 'unexplored',
      notes: '',
    },
    'regularization': {
      level: 'unexplored',
      notes: '',
    },

    // Math Foundations
    'calculus': {
      level: 'learning',
      notes: 'High school calculus. Familiar with more through self-study.',
    },
    'linear-algebra': {
      level: 'learning',
      notes: 'Basic matrix operations. Needs practice with transformations.',
    },

    // PyTorch / Implementation
    'python': {
      level: 'comfortable',
      notes: 'Strong programming background.',
    },
    'numpy': {
      level: 'learning',
      notes: 'Can use but not fluent.',
    },
    'pytorch': {
      level: 'unexplored',
      notes: 'Has used AI tools but not trained models.',
    },

    // Transformers & LLMs
    'attention': {
      level: 'unexplored',
      notes: 'Knows it exists. Needs deep dive on mechanics.',
    },
    'transformers': {
      level: 'unexplored',
      notes: 'Primary learning goal.',
    },
    'llm-training': {
      level: 'unexplored',
      notes: '',
    },
    'lora': {
      level: 'unexplored',
      notes: '',
    },

    // Diffusion Models
    'diffusion-basics': {
      level: 'unexplored',
      notes: '',
    },
    'stable-diffusion': {
      level: 'unexplored',
      notes: '',
    },
    'sd-finetuning': {
      level: 'unexplored',
      notes: '',
    },
  },

  struggles: [],

  patterns: [
    'ADHD: needs constrained lessons with clear objectives',
    'Previous DL course was pre-transformer era',
    'Learns best with interactive visualizations',
    'Strong programming background — can implement if shown the math',
  ],

  preferences: {
    preferredTopics: ['transformers', 'llms', 'stable-diffusion'],
    sessionLength: '15-25 min',
    notes: 'Goal is deep intuition, not just surface understanding.',
  },
}

// Helper functions
export const getSkillLevel = (skillId: string): SkillLevel => {
  return learnerState.skills[skillId]?.level ?? 'unexplored'
}

export const getActiveStruggles = (): Struggle[] => {
  return learnerState.struggles.filter((s) => !s.resolved)
}

export const getSkillsAtLevel = (level: SkillLevel): string[] => {
  return Object.entries(learnerState.skills)
    .filter(([, assessment]) => assessment.level === level)
    .map(([id]) => id)
}
