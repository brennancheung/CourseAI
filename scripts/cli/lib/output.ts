export type OutputFormat = 'pretty' | 'json'

let globalFormat: OutputFormat = 'pretty'

export const setOutputFormat = (format: OutputFormat) => {
  globalFormat = format
}

export const getOutputFormat = (): OutputFormat => globalFormat

export const output = (data: unknown) => {
  if (globalFormat === 'json') {
    console.log(JSON.stringify(data))
  } else {
    console.log(data)
  }
}

// Convex IDs are always 32 characters
const CONVEX_ID_LENGTH = 32

type TableOptions = {
  columnWidths?: number[]
  // Column indices that should never be truncated (e.g., ID columns)
  noTruncate?: number[]
}

export const outputTable = (
  headers: string[],
  rows: string[][],
  options?: TableOptions
) => {
  const { columnWidths, noTruncate = [] } = options ?? {}

  // Auto-detect ID columns by header name
  const idColumns = new Set([
    ...noTruncate,
    ...headers.map((h, i) => h.toLowerCase() === 'id' ? i : -1).filter(i => i >= 0)
  ])

  if (globalFormat === 'json') {
    // Convert rows to objects using headers as keys
    const objects = rows.map(row => {
      const obj: Record<string, string> = {}
      headers.forEach((h, i) => {
        obj[h.toLowerCase().replace(/\s+/g, '_')] = row[i]
      })
      return obj
    })
    console.log(JSON.stringify(objects))
    return
  }

  // Calculate column widths if not provided
  const widths = columnWidths ?? headers.map((h, i) => {
    const maxContent = Math.max(h.length, ...rows.map(r => (r[i] || '').length))
    // ID columns get exact width, others capped at 40
    if (idColumns.has(i)) {
      return Math.max(h.length, CONVEX_ID_LENGTH)
    }
    return Math.min(maxContent, 40)
  })

  // Print header
  const headerLine = headers.map((h, i) => h.padEnd(widths[i])).join('  ')
  console.log(headerLine)
  console.log('-'.repeat(headerLine.length))

  // Print rows
  rows.forEach(row => {
    const line = row.map((cell, i) => {
      const text = cell || ''
      // Never truncate ID columns
      if (idColumns.has(i)) {
        return text.padEnd(widths[i])
      }
      return text.length > widths[i] ? text.slice(0, widths[i] - 3) + '...' : text.padEnd(widths[i])
    }).join('  ')
    console.log(line)
  })

  console.log(`\n${rows.length} items.`)
}
