'use client'

import { use } from 'react'
import { notFound } from 'next/navigation'
import { getExercise } from '@/lib/exercises'

// Import lesson components
import {
  WhatIsLearningLesson,
  LinearRegressionLesson,
  LossFunctionsLesson,
  GradientDescentLesson,
  LearningRateLesson,
  ImplementingLinearRegressionLesson,
} from '@/components/lessons/module-1-1'

// Map slugs to lesson components
const lessonComponents: Record<string, React.ComponentType> = {
  'what-is-learning': WhatIsLearningLesson,
  'linear-regression': LinearRegressionLesson,
  'loss-functions': LossFunctionsLesson,
  'gradient-descent': GradientDescentLesson,
  'learning-rate': LearningRateLesson,
  'implementing-linear-regression': ImplementingLinearRegressionLesson,
}

interface LessonPageProps {
  params: Promise<{ slug: string }>
}

export default function LessonPage({ params }: LessonPageProps) {
  const { slug } = use(params)

  // Check if exercise exists
  const exercise = getExercise(slug)
  if (!exercise) {
    notFound()
  }

  // Get the lesson component
  const LessonComponent = lessonComponents[slug]
  if (!LessonComponent) {
    // If no custom component, could render a default template
    // For now, show not found
    notFound()
  }

  return <LessonComponent />
}
