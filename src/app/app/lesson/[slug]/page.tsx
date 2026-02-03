'use client'

import { use } from 'react'
import { notFound } from 'next/navigation'
import { curriculum, findNodeBySlug, isLesson } from '@/data/curriculum'

// Import lesson components
// These are mapped by the lessonComponent field in curriculum data
import {
  WhatIsLearningLesson,
  LinearRegressionLesson,
  LossFunctionsLesson,
  GradientDescentLesson,
  LearningRateLesson,
  ImplementingLinearRegressionLesson,
} from '@/components/lessons/module-1-1'

// Map lessonComponent names to actual components
// This is the only place components need to be registered
const componentRegistry: Record<string, React.ComponentType> = {
  WhatIsLearningLesson,
  LinearRegressionLesson,
  LossFunctionsLesson,
  GradientDescentLesson,
  LearningRateLesson,
  ImplementingLinearRegressionLesson,
}

interface LessonPageProps {
  params: Promise<{ slug: string }>
}

export default function LessonPage({ params }: LessonPageProps) {
  const { slug } = use(params)

  // Find the lesson in curriculum
  const node = findNodeBySlug(curriculum, slug)

  // Must exist and be a lesson (leaf node)
  if (!node || !isLesson(node)) {
    notFound()
  }

  // Must have a component registered
  const componentName = node.lessonComponent
  if (!componentName) {
    notFound()
  }

  const LessonComponent = componentRegistry[componentName]
  if (!LessonComponent) {
    console.error(`Missing component registration for: ${componentName}`)
    notFound()
  }

  return <LessonComponent />
}
