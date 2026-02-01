/**
 * Music theory utilities
 */

// Types
export type {
  NoteName,
  PitchClass,
  Mode,
  ChordQuality,
  Chord,
  Scale,
  CirclePosition,
} from './types'

// Note utilities
export {
  CHROMATIC_SHARPS,
  CHROMATIC_FLATS,
  noteToPitchClass,
  pitchClassToNote,
  getKeyPreference,
  transposeNote,
  getInterval,
  formatNote,
} from './notes'

// Scale utilities
export { getScaleNotes, getScale, getModeDisplayName } from './scales'

// Chord utilities
export {
  getChordNotes,
  getChord,
  getDiatonicChords,
  isPrimaryChord,
  getHarmonicFunction,
  DEGREE_COLORS,
  type HarmonicFunction,
} from './chords'

// Circle of fifths
export {
  CIRCLE_OF_FIFTHS_MAJOR,
  CIRCLE_OF_FIFTHS_MINOR,
  getCirclePosition,
  getRelativeMinor,
  getRelativeMajor,
  getAdjacentKeys,
  getMajorKeyAtPosition,
  getMinorKeyAtPosition,
} from './circle-of-fifths'
