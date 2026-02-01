/**
 * Circle of fifths utilities
 */

import type { CirclePosition, NoteName } from './types'
import { noteToPitchClass } from './notes'

// Circle of fifths order for major keys (starting at C, going clockwise)
export const CIRCLE_OF_FIFTHS_MAJOR: NoteName[] = [
  'C', // 12 o'clock
  'G', // 1 o'clock
  'D', // 2 o'clock
  'A', // 3 o'clock
  'E', // 4 o'clock
  'B', // 5 o'clock
  'Gb', // 6 o'clock (enharmonic with F#)
  'Db', // 7 o'clock
  'Ab', // 8 o'clock
  'Eb', // 9 o'clock
  'Bb', // 10 o'clock
  'F', // 11 o'clock
]

// Relative minor keys (same positions as their relative majors)
export const CIRCLE_OF_FIFTHS_MINOR: NoteName[] = [
  'A', // relative to C
  'E', // relative to G
  'B', // relative to D
  'F#', // relative to A
  'C#', // relative to E
  'G#', // relative to B
  'Eb', // relative to Gb
  'Bb', // relative to Db
  'F', // relative to Ab
  'C', // relative to Eb
  'G', // relative to Bb
  'D', // relative to F
]

/**
 * Get the position on the circle for a major key (0-11, where 0 is C at 12 o'clock)
 */
export function getCirclePosition(key: NoteName): CirclePosition {
  // Find by pitch class since we might have enharmonic equivalents
  const targetPc = noteToPitchClass(key)
  const index = CIRCLE_OF_FIFTHS_MAJOR.findIndex((k) => noteToPitchClass(k) === targetPc)
  if (index !== -1) return index as CirclePosition

  // Check minor keys
  const minorIndex = CIRCLE_OF_FIFTHS_MINOR.findIndex((k) => noteToPitchClass(k) === targetPc)
  if (minorIndex !== -1) return minorIndex as CirclePosition

  return 0 as CirclePosition
}

/**
 * Get the relative minor of a major key
 */
export function getRelativeMinor(majorKey: NoteName): NoteName {
  const position = getCirclePosition(majorKey)
  return CIRCLE_OF_FIFTHS_MINOR[position]
}

/**
 * Get the relative major of a minor key
 */
export function getRelativeMajor(minorKey: NoteName): NoteName {
  const position = getCirclePosition(minorKey)
  return CIRCLE_OF_FIFTHS_MAJOR[position]
}

/**
 * Get adjacent keys on the circle (share 6 out of 7 notes)
 */
export function getAdjacentKeys(key: NoteName): { clockwise: NoteName; counterClockwise: NoteName } {
  const position = getCirclePosition(key)
  const clockwise = CIRCLE_OF_FIFTHS_MAJOR[(position + 1) % 12]
  const counterClockwise = CIRCLE_OF_FIFTHS_MAJOR[(position + 11) % 12]
  return { clockwise, counterClockwise }
}

/**
 * Get the major key at a circle position
 */
export function getMajorKeyAtPosition(position: CirclePosition): NoteName {
  return CIRCLE_OF_FIFTHS_MAJOR[position]
}

/**
 * Get the minor key at a circle position
 */
export function getMinorKeyAtPosition(position: CirclePosition): NoteName {
  return CIRCLE_OF_FIFTHS_MINOR[position]
}
