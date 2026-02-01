/**
 * MIDI utilities for note conversion and keyboard layout.
 *
 * MIDI note numbers: C-1=0, C0=12, C1=24, C2=36, C3=48, C4=60 (middle C), C5=72, C6=84, C7=96
 *
 * Use MIDI internally for:
 * - WebMIDI integration
 * - Mathematical operations (transposition, ranges)
 * - Consistent representation
 *
 * Use note names in data/UI for human readability.
 */

// Note name to MIDI number mapping
const NOTE_TO_MIDI: Record<string, number> = {
  'C-1': 0, 'C#-1': 1, 'Db-1': 1, 'D-1': 2, 'D#-1': 3, 'Eb-1': 3, 'E-1': 4, 'F-1': 5,
  'F#-1': 6, 'Gb-1': 6, 'G-1': 7, 'G#-1': 8, 'Ab-1': 8, 'A-1': 9, 'A#-1': 10, 'Bb-1': 10, 'B-1': 11,
  'C0': 12, 'C#0': 13, 'Db0': 13, 'D0': 14, 'D#0': 15, 'Eb0': 15, 'E0': 16, 'F0': 17,
  'F#0': 18, 'Gb0': 18, 'G0': 19, 'G#0': 20, 'Ab0': 20, 'A0': 21, 'A#0': 22, 'Bb0': 22, 'B0': 23,
  'C1': 24, 'C#1': 25, 'Db1': 25, 'D1': 26, 'D#1': 27, 'Eb1': 27, 'E1': 28, 'F1': 29,
  'F#1': 30, 'Gb1': 30, 'G1': 31, 'G#1': 32, 'Ab1': 32, 'A1': 33, 'A#1': 34, 'Bb1': 34, 'B1': 35,
  'C2': 36, 'C#2': 37, 'Db2': 37, 'D2': 38, 'D#2': 39, 'Eb2': 39, 'E2': 40, 'F2': 41,
  'F#2': 42, 'Gb2': 42, 'G2': 43, 'G#2': 44, 'Ab2': 44, 'A2': 45, 'A#2': 46, 'Bb2': 46, 'B2': 47,
  'C3': 48, 'C#3': 49, 'Db3': 49, 'D3': 50, 'D#3': 51, 'Eb3': 51, 'E3': 52, 'F3': 53,
  'F#3': 54, 'Gb3': 54, 'G3': 55, 'G#3': 56, 'Ab3': 56, 'A3': 57, 'A#3': 58, 'Bb3': 58, 'B3': 59,
  'C4': 60, 'C#4': 61, 'Db4': 61, 'D4': 62, 'D#4': 63, 'Eb4': 63, 'E4': 64, 'F4': 65,
  'F#4': 66, 'Gb4': 66, 'G4': 67, 'G#4': 68, 'Ab4': 68, 'A4': 69, 'A#4': 70, 'Bb4': 70, 'B4': 71,
  'C5': 72, 'C#5': 73, 'Db5': 73, 'D5': 74, 'D#5': 75, 'Eb5': 75, 'E5': 76, 'F5': 77,
  'F#5': 78, 'Gb5': 78, 'G5': 79, 'G#5': 80, 'Ab5': 80, 'A5': 81, 'A#5': 82, 'Bb5': 82, 'B5': 83,
  'C6': 84, 'C#6': 85, 'Db6': 85, 'D6': 86, 'D#6': 87, 'Eb6': 87, 'E6': 88, 'F6': 89,
  'F#6': 90, 'Gb6': 90, 'G6': 91, 'G#6': 92, 'Ab6': 92, 'A6': 93, 'A#6': 94, 'Bb6': 94, 'B6': 95,
  'C7': 96, 'C#7': 97, 'Db7': 97, 'D7': 98, 'D#7': 99, 'Eb7': 99, 'E7': 100, 'F7': 101,
  'F#7': 102, 'Gb7': 102, 'G7': 103, 'G#7': 104, 'Ab7': 104, 'A7': 105, 'A#7': 106, 'Bb7': 106, 'B7': 107,
  'C8': 108, 'C#8': 109, 'Db8': 109, 'D8': 110, 'D#8': 111, 'Eb8': 111, 'E8': 112, 'F8': 113,
  'F#8': 114, 'Gb8': 114, 'G8': 115, 'G#8': 116, 'Ab8': 116, 'A8': 117, 'A#8': 118, 'Bb8': 118, 'B8': 119,
  'C9': 120, 'C#9': 121, 'Db9': 121, 'D9': 122, 'D#9': 123, 'Eb9': 123, 'E9': 124, 'F9': 125,
  'F#9': 126, 'Gb9': 126, 'G9': 127,
}

const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

/**
 * Convert a note name to MIDI number.
 * @example noteToMidi('C4') // 60 (middle C)
 * @example noteToMidi('G3') // 55
 */
export function noteToMidi(note: string): number {
  const midi = NOTE_TO_MIDI[note]
  if (midi === undefined) {
    console.warn(`Unknown note: ${note}, defaulting to C4 (60)`)
    return 60
  }
  return midi
}

/**
 * Shorthand alias for noteToMidi - convenient for inline use.
 * @example n('C4') // 60
 * @example [n('C4'), n('E4'), n('G4')] // C major chord
 */
export const n = noteToMidi

/**
 * Convert a MIDI number to note name.
 * @example midiToNote(60) // 'C4'
 * @example midiToNote(69) // 'A4'
 */
export function midiToNote(midi: number): string {
  const octave = Math.floor(midi / 12) - 1
  const noteIndex = midi % 12
  return `${NOTE_NAMES[noteIndex]}${octave}`
}

/**
 * Check if a MIDI note is a black key.
 */
export function isBlackKey(midi: number): boolean {
  const note = midi % 12
  return [1, 3, 6, 8, 10].includes(note)
}

/**
 * Check if a MIDI note is a white key.
 */
export function isWhiteKey(midi: number): boolean {
  return !isBlackKey(midi)
}

/**
 * Get the note name without octave (e.g., 'C#' from 'C#4' or MIDI 61).
 */
export function getNoteClass(midi: number): string {
  return NOTE_NAMES[midi % 12]
}

/**
 * Generate an array of MIDI notes in a range.
 * @example midiRange(60, 72) // C4 to C5
 */
export function midiRange(start: number, end: number): number[] {
  const notes: number[] = []
  for (let i = start; i <= end; i++) {
    notes.push(i)
  }
  return notes
}

/**
 * Generate an array of MIDI notes from note names.
 * @example notesToMidi(['C4', 'E4', 'G4']) // [60, 64, 67]
 */
export function notesToMidi(notes: string[]): number[] {
  return notes.map(noteToMidi)
}

/**
 * Transpose a MIDI note by semitones.
 * @example transpose(60, 7) // 67 (C4 + perfect fifth = G4)
 */
export function transpose(midi: number, semitones: number): number {
  return midi + semitones
}

/**
 * Get all notes in a chord given root and intervals.
 * @example chord(60, [0, 4, 7]) // C major: [60, 64, 67]
 * @example chord(n('A3'), [0, 3, 7]) // A minor: [57, 60, 64]
 */
export function chord(root: number, intervals: number[]): number[] {
  return intervals.map(interval => root + interval)
}

// Common chord intervals
export const CHORD_INTERVALS = {
  major: [0, 4, 7],
  minor: [0, 3, 7],
  diminished: [0, 3, 6],
  augmented: [0, 4, 8],
  major7: [0, 4, 7, 11],
  minor7: [0, 3, 7, 10],
  dominant7: [0, 4, 7, 10],
  sus2: [0, 2, 7],
  sus4: [0, 5, 7],
}

// Common scale intervals
export const SCALE_INTERVALS = {
  major: [0, 2, 4, 5, 7, 9, 11],
  minor: [0, 2, 3, 5, 7, 8, 10],
  harmonicMinor: [0, 2, 3, 5, 7, 8, 11],
  melodicMinor: [0, 2, 3, 5, 7, 9, 11],
  pentatonicMajor: [0, 2, 4, 7, 9],
  pentatonicMinor: [0, 3, 5, 7, 10],
  blues: [0, 3, 5, 6, 7, 10],
  chromatic: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
}

/**
 * Get all notes in a scale across octaves.
 * @example scale(60, SCALE_INTERVALS.major, 60, 84) // C major from C4 to C6
 */
export function scale(root: number, intervals: number[], startMidi: number, endMidi: number): number[] {
  const rootClass = root % 12
  const notes: number[] = []

  for (let midi = startMidi; midi <= endMidi; midi++) {
    const noteClass = midi % 12
    const intervalFromRoot = (noteClass - rootClass + 12) % 12
    if (intervals.includes(intervalFromRoot)) {
      notes.push(midi)
    }
  }

  return notes
}
