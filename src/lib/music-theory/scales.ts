/**
 * Scale utilities - generate scales from root and mode
 */

import type { Mode, NoteName, PitchClass, Scale } from './types'
import { getKeyPreference, noteToPitchClass, pitchClassToNote } from './notes'

// Scale intervals in semitones from root
const SCALE_INTERVALS: Record<Mode, number[]> = {
  major: [0, 2, 4, 5, 7, 9, 11],
  'natural-minor': [0, 2, 3, 5, 7, 8, 10],
  'harmonic-minor': [0, 2, 3, 5, 7, 8, 11],
}

/**
 * Get the notes of a scale
 */
export function getScaleNotes(root: NoteName, mode: Mode): NoteName[] {
  const rootPc = noteToPitchClass(root)
  const intervals = SCALE_INTERVALS[mode]
  const preferFlats = getKeyPreference(root) === 'flats'

  return intervals.map((interval) => {
    const pc = ((rootPc + interval) % 12) as PitchClass
    return pitchClassToNote(pc, preferFlats)
  })
}

/**
 * Get a full scale object
 */
export function getScale(root: NoteName, mode: Mode): Scale {
  return {
    root,
    mode,
    notes: getScaleNotes(root, mode),
  }
}

/**
 * Get the display name for a mode
 */
export function getModeDisplayName(mode: Mode): string {
  const names: Record<Mode, string> = {
    major: 'Major',
    'natural-minor': 'Minor',
    'harmonic-minor': 'Harmonic Minor',
  }
  return names[mode]
}
