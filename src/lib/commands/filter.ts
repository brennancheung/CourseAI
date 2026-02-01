import { Command } from './types'
import { fuzzyMatch } from '@/lib/utils/fuzzySearch'

export const filterCommands = (commands: Command[], search: string): Command[] => {
  if (!search) return commands

  const searchLower = search.toLowerCase()

  return commands.filter(command => {
    if (command.visible === false) return false

    const titleMatch = fuzzyMatch(searchLower, command.title)
    const keywordMatch = command.keywords.some(keyword =>
      fuzzyMatch(searchLower, keyword)
    )
    const categoryMatch = fuzzyMatch(searchLower, command.category)

    return titleMatch || keywordMatch || categoryMatch
  })
}

export const groupBy = <T, K extends keyof T>(
  array: T[],
  key: K
): Record<string, T[]> => {
  return array.reduce((result, item) => {
    const group = String(item[key])
    if (!result[group]) {
      result[group] = []
    }
    result[group].push(item)
    return result
  }, {} as Record<string, T[]>)
}
