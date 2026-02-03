import { CurriculumNode } from './types'

/**
 * The Learning Problem - introductory lessons
 */
const theLearningProblem: CurriculumNode = {
  slug: 'the-learning-problem',
  title: 'The Learning Problem',
  children: [
    {
      slug: 'core-concepts',
      title: 'Core Concepts',
      children: [
        {
          slug: 'what-is-learning',
          title: 'What is Learning?',
          lessonComponent: 'WhatIsLearningLesson',
          description: 'Understanding machine learning as function approximation',
        },
        {
          slug: 'linear-regression',
          title: 'Linear Regression',
          lessonComponent: 'LinearRegressionLesson',
          description: 'The simplest model: fitting a line to data',
        },
        {
          slug: 'loss-functions',
          title: 'Loss Functions',
          lessonComponent: 'LossFunctionsLesson',
          description: 'Measuring how wrong our predictions are',
        },
      ],
    },
    {
      slug: 'optimization',
      title: 'Optimization',
      children: [
        {
          slug: 'gradient-descent',
          title: 'Gradient Descent',
          lessonComponent: 'GradientDescentLesson',
          description: 'Following the slope to find the minimum',
        },
        {
          slug: 'learning-rate',
          title: 'Learning Rate',
          lessonComponent: 'LearningRateLesson',
          description: 'The most important hyperparameter',
        },
      ],
    },
    {
      slug: 'implementation',
      title: 'Implementation',
      children: [
        {
          slug: 'implementing-linear-regression',
          title: 'Linear Regression from Scratch',
          lessonComponent: 'ImplementingLinearRegressionLesson',
          description: 'Build it yourself in Python',
        },
      ],
    },
  ],
}

/**
 * Foundations - everything a beginner needs
 */
export const foundations: CurriculumNode = {
  slug: 'foundations',
  title: 'Foundations',
  icon: 'Layers',
  description: 'Core concepts every ML practitioner needs',
  children: [
    theLearningProblem,
    // Future: add more foundation topics
    // mathForML,
    // practicalSkills,
  ],
}
