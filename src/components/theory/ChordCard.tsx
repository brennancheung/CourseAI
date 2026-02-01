'use client'

import { useCallback, useRef, useState } from 'react'
import type { Chord } from '@/lib/music-theory'
import { DEGREE_COLORS } from '@/lib/music-theory'
import { cn } from '@/lib/utils'
import abcjs from 'abcjs'

interface ChordCardProps {
  chord: Chord
  degree: number
  className?: string
}

type PlaybackState = 'stopped' | 'loading' | 'playing'

export function ChordCard({ chord, degree, className }: ChordCardProps) {
  const colors = DEGREE_COLORS[degree]
  const [playbackState, setPlaybackState] = useState<PlaybackState>('stopped')
  const synthRef = useRef<abcjs.MidiBuffer | null>(null)

  const handlePlayClick = useCallback(async () => {
    // If already playing, stop
    if (playbackState === 'playing' && synthRef.current) {
      synthRef.current.stop()
      synthRef.current = null
      setPlaybackState('stopped')
      return
    }

    // Don't start if loading
    if (playbackState === 'loading') return

    setPlaybackState('loading')

    try {
      // Generate ABC notation for the chord
      // Use uppercase note names, abcjs handles octave placement
      const notes = chord.notes.map(formatNoteForAbc).join('')
      const abc = `X:1
M:4/4
L:1/2
K:C clef=treble
[${notes}]2`

      // Create a hidden container for rendering
      const container = document.createElement('div')
      container.style.display = 'none'
      document.body.appendChild(container)

      const visualObjs = abcjs.renderAbc(container, abc, {})

      if (!visualObjs?.[0]) {
        throw new Error('Failed to render chord')
      }

      // Create synth and play
      const synth = new abcjs.synth.CreateSynth()
      synthRef.current = synth

      await synth.init({
        visualObj: visualObjs[0],
        options: { qpm: 60 },
      })

      await synth.prime()
      synth.start()
      setPlaybackState('playing')

      // Auto-stop after the chord duration (roughly 2 seconds at qpm=60 for a half note)
      setTimeout(() => {
        if (synthRef.current === synth) {
          synth.stop()
          synthRef.current = null
          setPlaybackState('stopped')
        }
      }, 2000)

      // Cleanup the hidden container
      document.body.removeChild(container)
    } catch (error) {
      console.error('Failed to play chord:', error)
      setPlaybackState('stopped')
    }
  }, [chord.notes, playbackState])

  return (
    <button
      type="button"
      onClick={handlePlayClick}
      disabled={playbackState === 'loading'}
      className={cn(
        'flex flex-col items-center rounded-lg border px-3 py-2 cursor-pointer transition-colors',
        colors.bg,
        colors.border,
        'hover:opacity-80',
        playbackState === 'playing' && 'ring-2 ring-primary',
        className
      )}
    >
      {/* Roman numeral */}
      <div className={cn('text-base font-bold', colors.text)}>{chord.roman}</div>

      {/* Chord name (includes root) */}
      <div className="text-sm font-semibold">{chord.name}</div>

      {/* 3rd and 5th only (root is in chord name) */}
      <div className="mt-1 flex flex-col items-center">
        <span className="text-xs text-muted-foreground leading-tight">{chord.notes[1]}</span>
        <span className="text-xs text-muted-foreground leading-tight">{chord.notes[2]}</span>
      </div>
    </button>
  )
}

/**
 * Format a note name for ABC notation
 * ABC uses uppercase for octave below middle C, lowercase for middle C and above
 * We want chords in a comfortable range, so we'll use the octave around middle C
 */
function formatNoteForAbc(note: string): string {
  // Convert sharp/flat notation to ABC format
  // ABC uses ^ for sharp, _ for flat
  const formatted = note
    .replace('#', '^')
    .replace('b', '_')

  // For ABC notation, we need to handle accidentals before the note
  // e.g., C# becomes ^C, Bb becomes _B
  if (formatted.includes('^') || formatted.includes('_')) {
    const accidental = formatted.charAt(formatted.length - 1) === '^' || formatted.charAt(formatted.length - 1) === '_'
      ? formatted.charAt(formatted.length - 1)
      : formatted.charAt(0)
    const noteLetter = formatted.replace(/[\^_]/g, '')
    return accidental + noteLetter
  }

  return formatted
}
