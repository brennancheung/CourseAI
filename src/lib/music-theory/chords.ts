/**
 * Chord utilities - generate chords and diatonic chord progressions
 */

import type { Chord, ChordQuality, Mode, NoteName, PitchClass } from './types'
import { getKeyPreference, noteToPitchClass, pitchClassToNote } from './notes'
import { getScaleNotes } from './scales'

// Chord intervals in semitones from root (triads)
const CHORD_INTERVALS: Record<ChordQuality, number[]> = {
  major: [0, 4, 7],
  minor: [0, 3, 7],
  diminished: [0, 3, 6],
  augmented: [0, 4, 8],
}

// Diatonic chord qualities for each scale degree by mode
const DIATONIC_QUALITIES: Record<Mode, ChordQuality[]> = {
  major: ['major', 'minor', 'minor', 'major', 'major', 'minor', 'diminished'],
  'natural-minor': ['minor', 'diminished', 'major', 'minor', 'minor', 'major', 'major'],
  'harmonic-minor': ['minor', 'diminished', 'augmented', 'minor', 'major', 'major', 'diminished'],
}

// Roman numerals for major chords (uppercase)
const MAJOR_NUMERALS = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII']
// Roman numerals for minor/diminished chords (lowercase)
const MINOR_NUMERALS = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii']

/**
 * Get the notes of a chord
 */
export function getChordNotes(root: NoteName, quality: ChordQuality): NoteName[] {
  const rootPc = noteToPitchClass(root)
  const intervals = CHORD_INTERVALS[quality]
  const preferFlats = getKeyPreference(root) === 'flats'

  return intervals.map((interval) => {
    const pc = ((rootPc + interval) % 12) as PitchClass
    return pitchClassToNote(pc, preferFlats)
  })
}

/**
 * Get the roman numeral for a chord
 */
function getRomanNumeral(degree: number, quality: ChordQuality): string {
  const isUppercase = quality === 'major' || quality === 'augmented'
  const numeral = isUppercase ? MAJOR_NUMERALS[degree] : MINOR_NUMERALS[degree]

  if (quality === 'diminished') return `${numeral}\u00B0` // degree symbol
  if (quality === 'augmented') return `${numeral}+`
  return numeral
}

/**
 * Get the display name for a chord (e.g., "C major", "Dm", "B°")
 */
function getChordName(root: NoteName, quality: ChordQuality): string {
  const qualityNames: Record<ChordQuality, string> = {
    major: '',
    minor: 'm',
    diminished: '\u00B0',
    augmented: '+',
  }
  return `${root}${qualityNames[quality]}`
}

/**
 * Build a chord object
 */
export function getChord(root: NoteName, quality: ChordQuality, degree: number = 0): Chord {
  return {
    root,
    quality,
    notes: getChordNotes(root, quality),
    roman: getRomanNumeral(degree, quality),
    name: getChordName(root, quality),
  }
}

/**
 * Get the 7 diatonic chords for a key
 * @param useHarmonicV - In minor modes, use major V chord (from harmonic minor)
 */
export function getDiatonicChords(
  root: NoteName,
  mode: Mode,
  useHarmonicV: boolean = false
): Chord[] {
  const scaleNotes = getScaleNotes(root, mode)
  let qualities = [...DIATONIC_QUALITIES[mode]]

  // In natural minor, optionally use harmonic minor's V chord (major instead of minor)
  if (mode === 'natural-minor' && useHarmonicV) {
    qualities[4] = 'major' // V becomes major
  }

  return scaleNotes.map((note, degree) => getChord(note, qualities[degree], degree))
}

/**
 * Check if a chord is a primary chord (I/i, IV/iv, V)
 */
export function isPrimaryChord(degree: number): boolean {
  return degree === 0 || degree === 3 || degree === 4
}

/**
 * Harmonic function types
 */
export type HarmonicFunction = 'tonic' | 'subdominant' | 'dominant'

/**
 * Get the harmonic function of a scale degree
 * - Tonic (I, iii, vi): stability, resolution
 * - Subdominant (ii, IV): departure, movement
 * - Dominant (V, vii°): tension, wants to resolve
 */
export function getHarmonicFunction(degree: number): HarmonicFunction {
  // Tonic family: I (0), iii (2), vi (5)
  if (degree === 0 || degree === 2 || degree === 5) return 'tonic'
  // Subdominant family: ii (1), IV (3)
  if (degree === 1 || degree === 3) return 'subdominant'
  // Dominant family: V (4), vii° (6)
  return 'dominant'
}

/**
 * Color classes for each scale degree based on harmonic function
 * Cool colors (blue/green/violet) = tonic/stable
 * Warm colors (red/pink) = dominant/tension
 * Neutral (orange/yellow) = subdominant/transition
 */
export const DEGREE_COLORS: Record<number, { bg: string; border: string; text: string }> = {
  0: { bg: 'bg-blue-500/15', border: 'border-blue-500/50', text: 'text-blue-700 dark:text-blue-300' }, // I - tonic
  1: { bg: 'bg-orange-500/15', border: 'border-orange-500/50', text: 'text-orange-700 dark:text-orange-300' }, // ii - subdominant
  2: { bg: 'bg-emerald-500/15', border: 'border-emerald-500/50', text: 'text-emerald-700 dark:text-emerald-300' }, // iii - tonic family
  3: { bg: 'bg-yellow-500/15', border: 'border-yellow-500/50', text: 'text-yellow-700 dark:text-yellow-300' }, // IV - subdominant
  4: { bg: 'bg-red-500/15', border: 'border-red-500/50', text: 'text-red-700 dark:text-red-300' }, // V - dominant
  5: { bg: 'bg-violet-500/15', border: 'border-violet-500/50', text: 'text-violet-700 dark:text-violet-300' }, // vi - tonic family
  6: { bg: 'bg-pink-500/15', border: 'border-pink-500/50', text: 'text-pink-700 dark:text-pink-300' }, // vii° - dominant family
}
