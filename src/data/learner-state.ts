/**
 * Learner State
 *
 * This file is updated by Claude Code after review conversations.
 * It captures the current state of skill development, active struggles,
 * and patterns discovered through practice.
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

  // Active struggles to address in upcoming exercises
  struggles: Struggle[]

  // Patterns noticed across sessions
  patterns: string[]

  // Preferences discovered
  preferences: {
    preferredKeys?: string[]
    sessionLength?: string
    notes?: string
  }
}

/**
 * Current learner state
 *
 * Updated by Claude Code based on review conversations.
 * Initial state reflects baseline from venture notes.
 */
export const learnerState: LearnerState = {
  lastUpdated: '2026-01-17',

  skills: {
    // Foundations
    'daw-navigation': {
      level: 'learning',
      notes: 'Basic Cubase navigation. Knows some shortcuts but still learning workflow.',
    },
    'midi-recording': {
      level: 'learning',
      notes: 'Can record MIDI. Limited experience with editing and quantizing.',
    },
    'layering-basics': {
      level: 'unexplored',
      notes: 'Mostly single-track playing so far.',
    },

    // Orchestration
    'string-voicing': {
      level: 'unexplored',
      notes: 'Has piano voicing intuition. Needs to learn orchestral distribution.',
    },
    'brass-basics': {
      level: 'unexplored',
      notes: '',
    },
    'woodwind-color': {
      level: 'unexplored',
      notes: '',
    },

    // Hybrid
    'braams': {
      level: 'unexplored',
      notes: 'Knows the sound, not how to create it.',
    },
    'risers-drops': {
      level: 'unexplored',
      notes: '',
    },
    'hybrid-blending': {
      level: 'unexplored',
      notes: '',
    },

    // Rhythm
    'percussion-layering': {
      level: 'unexplored',
      notes: '',
    },
    'rhythm-ostinatos': {
      level: 'unexplored',
      notes: '',
    },
    'rhythm-accents': {
      level: 'unexplored',
      notes: '',
    },

    // Emotion
    'tension-building': {
      level: 'unexplored',
      notes: 'Has intuitive sense from listening. Needs technical vocabulary.',
    },
    'release-resolution': {
      level: 'unexplored',
      notes: '',
    },
    'quiet-intensity': {
      level: 'unexplored',
      notes: '',
    },

    // Production
    'mixing-basics': {
      level: 'unexplored',
      notes: '',
    },
    'reverb-space': {
      level: 'unexplored',
      notes: '',
    },
    'final-polish': {
      level: 'unexplored',
      notes: '',
    },
  },

  struggles: [],

  patterns: [
    'ADHD: needs very constrained exercises to avoid paralysis',
    'Piano background: thinks in terms of keyboard voicings',
    'Prefers C# minor as comfort zone',
  ],

  preferences: {
    preferredKeys: ['C# minor', 'C minor'],
    sessionLength: '15-25 min',
    notes: 'Responds well to clear constraints and single-focus exercises.',
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
