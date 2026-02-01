export const siteConfig = {
  name: 'CourseAI',
  description: 'A Next.js webapp for learning AI and machine learning fundamentals',
  url: 'https://course-ai.com',
} as const

export type SiteConfig = typeof siteConfig
