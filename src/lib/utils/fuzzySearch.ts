/**
 * Fuzzy search utilities for filtering and matching
 */

/**
 * Fuzzy match a search string against a target string.
 * Returns true if all characters in search appear in target in order (but not necessarily consecutive).
 *
 * Examples:
 * - "pip" matches "Pipeline" (P, I, P)
 * - "vent" matches "Ventures" (V, E, N, T)
 * - "sett" matches "Settings" (S, E, T, T)
 */
export const fuzzyMatch = (search: string, target: string): boolean => {
  const searchLower = search.toLowerCase()
  const targetLower = target.toLowerCase()

  let searchIndex = 0
  let targetIndex = 0

  while (searchIndex < searchLower.length && targetIndex < targetLower.length) {
    if (searchLower[searchIndex] === targetLower[targetIndex]) {
      searchIndex++
    }
    targetIndex++
  }

  return searchIndex === searchLower.length
}

/**
 * Fuzzy match against multiple fields
 * Returns true if search matches any of the provided fields
 */
export const fuzzyMatchMultiple = (search: string, fields: string[]): boolean => {
  return fields.some(field => fuzzyMatch(search, field))
}

/**
 * Score a fuzzy match (higher score = better match)
 * Consecutive matches and early matches score higher
 */
export const fuzzyScore = (search: string, target: string): number => {
  const searchLower = search.toLowerCase()
  const targetLower = target.toLowerCase()

  let score = 0
  let searchIndex = 0
  let targetIndex = 0
  let consecutiveBonus = 1

  while (searchIndex < searchLower.length && targetIndex < targetLower.length) {
    if (searchLower[searchIndex] === targetLower[targetIndex]) {
      // Base point for match
      score += 1

      // Bonus for consecutive matches
      score += consecutiveBonus
      consecutiveBonus = Math.min(consecutiveBonus * 2, 8)

      // Bonus for early position match
      score += Math.max(0, 10 - targetIndex) / 10

      searchIndex++
    } else {
      consecutiveBonus = 1
    }
    targetIndex++
  }

  // Return 0 if not all characters matched
  if (searchIndex < searchLower.length) {
    return 0
  }

  // Bonus for shorter targets (more relevant matches)
  score += Math.max(0, 20 - target.length) / 2

  return score
}

/**
 * Filter and sort items by fuzzy search
 */
export const fuzzyFilter = <T>(
  items: T[],
  search: string,
  getSearchableFields: (item: T) => string[]
): T[] => {
  if (!search) return items

  // Score each item
  const scoredItems = items
    .map(item => {
      const fields = getSearchableFields(item)
      const scores = fields.map(field => fuzzyScore(search, field))
      const maxScore = Math.max(...scores, 0)
      return { item, score: maxScore }
    })
    .filter(({ score }) => score > 0)
    .sort((a, b) => b.score - a.score)

  return scoredItems.map(({ item }) => item)
}
