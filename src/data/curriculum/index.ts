/**
 * Main curriculum export
 *
 * Composes all curriculum sections into a single tree.
 * Each section can be defined in its own file for organization.
 */

import { CurriculumNode } from './types'
import { foundations } from './foundations'
import { pytorch } from './pytorch'
import { cnns } from './cnns'
import { llms } from './llms'
import { stableDiffusion } from './stable-diffusion'
import { recentLlmAdvances } from './recent-llm-advances'
import { postSdAdvances } from './post-sd-advances'
import { specialTopics } from './special-topics'

// Re-export types and utilities
export * from './types'

/**
 * The complete curriculum tree
 *
 * Add new series here as they're built.
 * See course.md for the full planned series list.
 */
export const curriculum: CurriculumNode[] = [
  foundations,
  pytorch,
  cnns,
  llms,
  recentLlmAdvances,
  stableDiffusion,
  postSdAdvances,
  specialTopics,
]
