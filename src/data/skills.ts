export type Skill = {
  id: string
  name: string
  description: string
  category: string
}

export type SkillCategory = {
  id: string
  name: string
  description: string
  skills: Skill[]
}

// TODO: Populate with AI/ML learning skills
export const skillCategories: SkillCategory[] = []

// Helper functions
export const getAllSkills = (): Skill[] =>
  skillCategories.flatMap((cat) => cat.skills)

export const getSkill = (id: string): Skill | undefined =>
  getAllSkills().find((s) => s.id === id)

export const getSkillCategory = (id: string): SkillCategory | undefined =>
  skillCategories.find((c) => c.id === id)
