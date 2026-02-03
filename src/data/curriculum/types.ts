/**
 * Curriculum data structure
 *
 * Uses a recursive tree - no confusing part/module/section terminology.
 * Just parent → child relationships. UI determines presentation based on depth.
 */

export type CurriculumNode = {
  slug: string
  title: string
  /** Optional icon for top-level nodes */
  icon?: string
  /** Child nodes - if present, this is a group. If absent, this is a leaf. */
  children?: CurriculumNode[]
  /** Only for leaf nodes: the lesson component to render */
  lessonComponent?: string
  /** Optional description shown on hover or in expanded view */
  description?: string
}

/**
 * Helper to check if a node is a leaf (lesson) vs a group
 */
export function isLesson(node: CurriculumNode): boolean {
  return !node.children || node.children.length === 0
}

/**
 * Find a node by slug (searches entire tree)
 */
export function findNodeBySlug(
  nodes: CurriculumNode[],
  slug: string
): CurriculumNode | undefined {
  for (const node of nodes) {
    if (node.slug === slug) return node
    if (node.children) {
      const found = findNodeBySlug(node.children, slug)
      if (found) return found
    }
  }
  return undefined
}

/**
 * Get the path to a node (array of slugs from root to node)
 */
export function getPathToNode(
  nodes: CurriculumNode[],
  slug: string,
  path: string[] = []
): string[] | undefined {
  for (const node of nodes) {
    if (node.slug === slug) return [...path, node.slug]
    if (node.children) {
      const found = getPathToNode(node.children, slug, [...path, node.slug])
      if (found) return found
    }
  }
  return undefined
}

/**
 * Get all lessons (leaf nodes) in order
 */
export function getAllLessons(nodes: CurriculumNode[]): CurriculumNode[] {
  const lessons: CurriculumNode[] = []

  function traverse(node: CurriculumNode) {
    if (isLesson(node)) {
      lessons.push(node)
    } else if (node.children) {
      node.children.forEach(traverse)
    }
  }

  nodes.forEach(traverse)
  return lessons
}

/**
 * Count total lessons in a node (including all descendants)
 */
export function countLessons(node: CurriculumNode): number {
  if (isLesson(node)) return 1
  if (!node.children) return 0
  return node.children.reduce((sum, child) => sum + countLessons(child), 0)
}
