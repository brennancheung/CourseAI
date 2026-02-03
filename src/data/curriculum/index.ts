/**
 * Main curriculum export
 *
 * Composes all curriculum sections into a single tree.
 * Each section can be defined in its own file for organization.
 */

import { CurriculumNode } from './types'
import { foundations } from './foundations'

// Re-export types and utilities
export * from './types'

/**
 * The complete curriculum tree
 *
 * Add new top-level sections here as they're built:
 * - foundations (current)
 * - classicalML (future)
 * - neuralNetworks (future)
 * - etc.
 */
export const curriculum: CurriculumNode[] = [
  foundations,
  // Future sections:
  // classicalML,
  // neuralNetworks,
  // trainingDeepNetworks,
  // convolutionalNetworks,
  // sequenceModels,
  // generativeModels,
  // reinforcementLearning,
  // productionML,
  // frontiers,
]
