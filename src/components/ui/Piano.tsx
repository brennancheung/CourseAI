'use client'

import { cn } from '@/lib/utils'
import { isBlackKey, midiToNote } from '@/lib/midi'

/**
 * Highlighted key configuration.
 * Can specify a single color or layer multiple colors for range visualization.
 */
export type KeyHighlight = {
  color: string // Tailwind class, e.g., 'bg-blue-500' or 'bg-blue-500/60'
  label?: string // Optional label shown in tooltip
}

export interface PianoProps {
  /** Start of keyboard range (MIDI number). Default: 24 (C1) */
  startMidi?: number
  /** End of keyboard range (MIDI number). Default: 96 (C7) */
  endMidi?: number
  /** Map of MIDI note number to highlight configuration */
  highlightedKeys?: Map<number, KeyHighlight>
  /** Called when a key is clicked */
  onKeyClick?: (midi: number) => void
  /** Show octave labels on C keys. Default: true */
  showOctaveLabels?: boolean
  /** Height of white keys in pixels. Default: 100 */
  height?: number
  className?: string
}

/**
 * Piano keyboard visualization atom.
 *
 * A composable building block for keyboard-based visualizations.
 * Use this as the foundation for:
 * - Instrument range displays (PianoRoll)
 * - Chord visualizers
 * - Scale highlighters
 * - WebMIDI input displays
 *
 * @example Basic usage
 * ```tsx
 * <Piano startMidi={48} endMidi={72} />
 * ```
 *
 * @example With highlighted keys
 * ```tsx
 * const highlights = new Map([
 *   [60, { color: 'bg-blue-500', label: 'Root' }],
 *   [64, { color: 'bg-blue-400', label: 'Third' }],
 *   [67, { color: 'bg-blue-400', label: 'Fifth' }],
 * ])
 * <Piano highlightedKeys={highlights} />
 * ```
 *
 * @example With click handler
 * ```tsx
 * <Piano onKeyClick={(midi) => playNote(midi)} />
 * ```
 */
export function Piano({
  startMidi = 24,
  endMidi = 96,
  highlightedKeys,
  onKeyClick,
  showOctaveLabels = true,
  height = 100,
  className,
}: PianoProps) {
  // Build white and black key arrays with position info
  const whiteKeys: number[] = []
  const blackKeys: { midi: number; whiteKeyIndex: number }[] = []

  let whiteKeyIndex = 0
  for (let midi = startMidi; midi <= endMidi; midi++) {
    if (!isBlackKey(midi)) {
      whiteKeys.push(midi)
      whiteKeyIndex++
    } else {
      // Track which white key this black key sits after
      blackKeys.push({ midi, whiteKeyIndex: whiteKeyIndex - 1 })
    }
  }

  const whiteKeyCount = whiteKeys.length
  const blackKeyHeight = Math.round(height * 0.6)

  const getHighlightColor = (midi: number): string | null => {
    const highlight = highlightedKeys?.get(midi)
    return highlight?.color ?? null
  }

  const getTooltip = (midi: number): string => {
    const noteName = midiToNote(midi)
    const highlight = highlightedKeys?.get(midi)
    if (highlight?.label) {
      return `${noteName} (${highlight.label})`
    }
    return noteName
  }

  const handleClick = (midi: number) => {
    onKeyClick?.(midi)
  }

  const isInteractive = !!onKeyClick

  return (
    <div className={cn('relative', className)}>
      <div className="relative" style={{ height }}>
        {/* White keys - use flex to fill available width */}
        <div className="flex h-full">
          {whiteKeys.map((midi) => {
            const highlightColor = getHighlightColor(midi)
            const noteName = midiToNote(midi)
            const isC = midi % 12 === 0

            return (
              <button
                key={midi}
                type="button"
                onClick={() => handleClick(midi)}
                disabled={!isInteractive}
                className={cn(
                  'relative flex-1 min-w-0 h-full border-r border-zinc-300 bg-white transition-colors first:rounded-l last:rounded-r',
                  isInteractive && 'hover:bg-zinc-100 cursor-pointer',
                  !isInteractive && 'cursor-default'
                )}
                title={getTooltip(midi)}
              >
                {/* Highlight fill - inset from edges to show key structure */}
                {highlightColor && (
                  <div
                    className={cn('absolute inset-x-px top-px bottom-1 rounded-sm', highlightColor)}
                  />
                )}
                {showOctaveLabels && isC && (
                  <span className={cn(
                    'absolute bottom-1 left-1/2 -translate-x-1/2 text-[9px] pointer-events-none z-10',
                    highlightColor ? 'text-zinc-600 font-medium' : 'text-zinc-400'
                  )}>
                    {noteName}
                  </span>
                )}
              </button>
            )
          })}
        </div>

        {/* Black keys - positioned as percentage of white key positions */}
        {blackKeys.map(({ midi, whiteKeyIndex }) => {
          const highlightColor = getHighlightColor(midi)
          // Position: after the white key at whiteKeyIndex, offset by ~70% of a white key width
          const leftPercent = ((whiteKeyIndex + 0.65) / whiteKeyCount) * 100
          const widthPercent = (0.7 / whiteKeyCount) * 100

          return (
            <button
              key={midi}
              type="button"
              onClick={() => handleClick(midi)}
              disabled={!isInteractive}
              className={cn(
                'absolute top-0 rounded-b transition-colors z-10 bg-zinc-800',
                isInteractive && 'hover:bg-zinc-700 cursor-pointer',
                !isInteractive && 'cursor-default'
              )}
              style={{
                left: `${leftPercent}%`,
                width: `${widthPercent}%`,
                height: blackKeyHeight,
              }}
              title={getTooltip(midi)}
            >
              {/* Highlight fill - inset to show key structure */}
              {highlightColor && (
                <div
                  className={cn('absolute inset-x-px top-px bottom-1 rounded-sm', highlightColor)}
                />
              )}
            </button>
          )
        })}
      </div>
    </div>
  )
}

/**
 * Helper to create a highlight map from an array of notes with the same color.
 * @example highlightNotes([60, 64, 67], 'bg-blue-500')
 */
export function highlightNotes(
  notes: number[],
  color: string,
  label?: string
): Map<number, KeyHighlight> {
  const map = new Map<number, KeyHighlight>()
  for (const note of notes) {
    map.set(note, { color, label })
  }
  return map
}

/**
 * Helper to merge multiple highlight maps.
 * Later maps override earlier ones for the same key.
 */
export function mergeHighlights(
  ...maps: Map<number, KeyHighlight>[]
): Map<number, KeyHighlight> {
  const merged = new Map<number, KeyHighlight>()
  for (const map of maps) {
    for (const [key, value] of map) {
      merged.set(key, value)
    }
  }
  return merged
}

/**
 * Helper to create highlights for a range of notes.
 * @example highlightRange(60, 72, 'bg-blue-500/50')
 */
export function highlightRange(
  startMidi: number,
  endMidi: number,
  color: string,
  label?: string
): Map<number, KeyHighlight> {
  const map = new Map<number, KeyHighlight>()
  for (let midi = startMidi; midi <= endMidi; midi++) {
    map.set(midi, { color, label })
  }
  return map
}
