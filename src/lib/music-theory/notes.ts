/**
 * Note utilities - pitch class conversion, enharmonic handling
 */

import type { NoteName, PitchClass } from './types'

// Chromatic scale with sharps (for sharp-side keys)
export const CHROMATIC_SHARPS: NoteName[] = [
  'C',
  'C#',
  'D',
  'D#',
  'E',
  'F',
  'F#',
  'G',
  'G#',
  'A',
  'A#',
  'B',
]

// Chromatic scale with flats (for flat-side keys)
export const CHROMATIC_FLATS: NoteName[] = [
  'C',
  'Db',
  'D',
  'Eb',
  'E',
  'F',
  'Gb',
  'G',
  'Ab',
  'A',
  'Bb',
  'B',
]

// Map note names to pitch class (0-11)
const NOTE_TO_PITCH_CLASS: Record<NoteName, PitchClass> = {
  C: 0,
  'C#': 1,
  Db: 1,
  D: 2,
  'D#': 3,
  Eb: 3,
  E: 4,
  F: 5,
  'F#': 6,
  Gb: 6,
  G: 7,
  'G#': 8,
  Ab: 8,
  A: 9,
  'A#': 10,
  Bb: 10,
  B: 11,
}

// Keys that prefer flat spellings
const FLAT_KEYS: NoteName[] = ['F', 'Bb', 'Eb', 'Ab', 'Db', 'Gb']

// Keys that prefer sharp spellings
const SHARP_KEYS: NoteName[] = ['G', 'D', 'A', 'E', 'B', 'F#']

/**
 * Convert a note name to its pitch class (0-11)
 */
export function noteToPitchClass(note: NoteName): PitchClass {
  return NOTE_TO_PITCH_CLASS[note]
}

/**
 * Convert a pitch class to a note name
 */
export function pitchClassToNote(pc: PitchClass, preferFlats: boolean): NoteName {
  const chromatic = preferFlats ? CHROMATIC_FLATS : CHROMATIC_SHARPS
  return chromatic[pc]
}

/**
 * Get whether a key prefers flat or sharp spellings
 */
export function getKeyPreference(key: NoteName): 'flats' | 'sharps' {
  if (FLAT_KEYS.includes(key)) return 'flats'
  if (SHARP_KEYS.includes(key)) return 'sharps'
  // C is neutral, default to sharps
  return 'sharps'
}

/**
 * Transpose a note by a number of semitones
 */
export function transposeNote(note: NoteName, semitones: number, preferFlats: boolean): NoteName {
  const pc = noteToPitchClass(note)
  const newPc = ((pc + semitones) % 12 + 12) % 12 as PitchClass
  return pitchClassToNote(newPc, preferFlats)
}

/**
 * Get the interval in semitones between two notes
 */
export function getInterval(from: NoteName, to: NoteName): number {
  const fromPc = noteToPitchClass(from)
  const toPc = noteToPitchClass(to)
  return ((toPc - fromPc) % 12 + 12) % 12
}

/**
 * Format a note for display (no changes needed, but could add unicode symbols later)
 */
export function formatNote(note: NoteName): string {
  return note
}
