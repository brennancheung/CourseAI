import { curriculum, getAllLessons, type CurriculumNode } from '@/data/curriculum'

export type Exercise = {
  slug: string
  title: string
  description: string
  category: string
  duration: string
  constraints: string[]
  steps: string[]
  recommendedWhen?: string
  skills?: string[]
  prerequisites?: string[]
}

function nodeToExercise(node: CurriculumNode): Exercise | undefined {
  if (!node.exercise || !node.category || !node.duration) return undefined
  return {
    slug: node.slug,
    title: node.title,
    description: node.description ?? '',
    category: node.category,
    duration: node.duration,
    constraints: node.exercise.constraints,
    steps: node.exercise.steps,
    recommendedWhen: node.exercise.recommendedWhen,
    skills: node.skills,
    prerequisites: node.prerequisites,
  }
}

export const getAllExercises = (): Exercise[] =>
  getAllLessons(curriculum)
    .map(nodeToExercise)
    .filter((e): e is Exercise => e !== undefined)

export const getExercise = (slug: string): Exercise | undefined =>
  getAllExercises().find((e) => e.slug === slug)

export const getExercisesByCategory = (category: string): Exercise[] =>
  getAllExercises().filter((e) => e.category === category)

export const getCategories = (): string[] =>
  [...new Set(getAllExercises().map((e) => e.category))]
