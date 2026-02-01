/**
 * Music theory type definitions
 */

// All possible note names (including enharmonic equivalents)
export type NoteName =
  | 'C'
  | 'C#'
  | 'Db'
  | 'D'
  | 'D#'
  | 'Eb'
  | 'E'
  | 'F'
  | 'F#'
  | 'Gb'
  | 'G'
  | 'G#'
  | 'Ab'
  | 'A'
  | 'A#'
  | 'Bb'
  | 'B'

// Pitch class (0-11) for internal calculations
export type PitchClass = 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11

// Supported scale modes
export type Mode = 'major' | 'natural-minor' | 'harmonic-minor'

// Chord qualities
export type ChordQuality = 'major' | 'minor' | 'diminished' | 'augmented'

// A chord with all its properties
export interface Chord {
  root: NoteName
  quality: ChordQuality
  notes: NoteName[]
  roman: string
  name: string
}

// A scale with its notes
export interface Scale {
  root: NoteName
  mode: Mode
  notes: NoteName[]
}

// Circle of fifths position (0 = C, clockwise)
export type CirclePosition = 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11
