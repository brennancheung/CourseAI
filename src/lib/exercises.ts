export type ExerciseScore = {
  /** ABC notation string */
  abc: string
  /** Title for the score */
  title?: string
  /** Default tempo for playback */
  tempo?: number
}

export type Exercise = {
  slug: string
  title: string
  description: string
  category: string
  duration: string // e.g., "20 min"
  constraints: string[]
  steps: string[]
  // Why this exercise might be recommended
  recommendedWhen?: string
  // Skills this exercise builds
  skills?: string[]
  // Prerequisites (other exercise slugs)
  prerequisites?: string[]
  // Optional score notation for the exercise
  score?: ExerciseScore
  // Optional long-form content (markdown)
  content?: string
}

// Import exercises from lesson modules
import {
  whatIsLearningExercise,
  linearRegressionExercise,
  lossFunctionsExercise,
  gradientDescentExercise,
  learningRateExercise,
  implementingLinearRegressionExercise,
} from '@/components/lessons/module-1-1'

// Register all exercises
export const exercises: Record<string, Exercise> = {
  // Module 1.1: The Learning Problem
  [whatIsLearningExercise.slug]: whatIsLearningExercise,
  [linearRegressionExercise.slug]: linearRegressionExercise,
  [lossFunctionsExercise.slug]: lossFunctionsExercise,
  [gradientDescentExercise.slug]: gradientDescentExercise,
  [learningRateExercise.slug]: learningRateExercise,
  [implementingLinearRegressionExercise.slug]: implementingLinearRegressionExercise,
}

// Get all exercises as array
export const getAllExercises = (): Exercise[] => Object.values(exercises)

// Get exercise by slug
export const getExercise = (slug: string): Exercise | undefined => exercises[slug]

// Get exercises by category
export const getExercisesByCategory = (category: string): Exercise[] =>
  getAllExercises().filter((e) => e.category === category)

// Get all unique categories
export const getCategories = (): string[] =>
  [...new Set(getAllExercises().map((e) => e.category))]
