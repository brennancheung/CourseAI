/**
 * Maps ABCJS instrument names to Sonatina Symphonic Orchestra preset numbers.
 *
 * IMPORTANT: This mapping is specific to the converted Sonatina SF2 soundfont.
 * Different articulations have different preset numbers.
 *
 * ## Choosing Articulations
 *
 * - **Sustain**: Long notes, held chords, slow melodies
 * - **Legato**: Connected melodic lines, smooth passages
 * - **Staccato**: Short detached notes, 8th/16th note patterns, rhythmic figures
 * - **Marcato**: Accented notes, powerful statements
 * - **Pizzicato**: Plucked strings (strings only)
 *
 * ## Instrument Ranges (written pitch)
 *
 * **Strings:**
 * - Violin: G3 - A7 (lowest open string G below middle C)
 * - Viola: C3 - E6 (lowest open string C below middle C)
 * - Cello: C2 - A5 (lowest open string C, two octaves below middle C)
 * - Double Bass: E1 - G4 (written; sounds octave lower)
 *
 * **Brass:**
 * - French Horn: F2 - C6 (written; sounds fifth lower)
 * - Trumpet: F#3 - D6
 * - Trombone: E2 - Bb4 (tenor)
 * - Tuba: D1 - F4
 *
 * **Woodwinds:**
 * - Flute: C4 - D7
 * - Oboe: Bb3 - A6
 * - Clarinet: D3 - Bb6 (written; sounds whole step lower for Bb clarinet)
 * - Bassoon: Bb1 - Eb5
 *
 * ## Sonatina Preset Numbers
 *
 * Strings (section):
 * - All Strings: 96 (Marcato), 97 (Pizzicato), 98 (Staccato), 99 (Sustain)
 * - Basses: 100 (Legato), 101 (Marcato), 102 (Pizzicato), 103 (Staccato), 104 (Sustain)
 * - Celli: 105 (Legato), 106 (Marcato), 107 (Pizzicato), 108 (Staccato), 109 (Sustain)
 * - Violas: 114 (Legato), 115 (Marcato), 116 (Pizzicato), 117 (Staccato), 118 (Sustain)
 * - Violin Solo: 119 (Legato), 122 (Staccato), 123 (Sustain)
 *
 * Brass:
 * - Horns: 11 (Legato), 12 (Marcato), 13 (Staccato), 14 (Sustain)
 * - Trombones: 15 (Legato), 17 (Marcato), 18 (Staccato), 20 (Sustain)
 * - Trumpets: 29 (Legato), 31 (Marcato), 32 (Staccato), 34 (Sustain)
 * - Tuba: 35 (Legato), 37 (Marcato), 38 (Staccato), 40 (Sustain)
 *
 * Woodwinds:
 * - Flutes: 191 (Sustain), 190 (Staccato)
 * - Oboes: 200 (Sustain), 199 (Staccato)
 * - Clarinets: 173 (Sustain), 172 (Staccato)
 * - Bassoons: 164 (Sustain), 163 (Staccato)
 *
 * Other:
 * - Harp: 86
 * - Timpani: 93
 * - Glockenspiel: 92
 * - Xylophone: 95
 * - Chimes: 89
 * - Chorus: 84
 */

type Articulation = 'sustain' | 'legato' | 'staccato' | 'marcato' | 'pizzicato'

interface InstrumentPresets {
  sustain: number
  legato?: number
  staccato?: number
  marcato?: number
  pizzicato?: number
}

/**
 * Sonatina preset numbers organized by instrument and articulation.
 */
const sonatina: Record<string, InstrumentPresets> = {
  // Strings - Section
  strings: { sustain: 99, staccato: 98, marcato: 96, pizzicato: 97 },
  basses: { sustain: 104, legato: 100, staccato: 103, marcato: 101, pizzicato: 102 },
  celli: { sustain: 109, legato: 105, staccato: 108, marcato: 106, pizzicato: 107 },
  violas: { sustain: 118, legato: 114, staccato: 117, marcato: 115, pizzicato: 116 },
  violin: { sustain: 123, legato: 119, staccato: 122 },

  // Brass
  horns: { sustain: 14, legato: 11, staccato: 13, marcato: 12 },
  trumpets: { sustain: 34, legato: 29, staccato: 32, marcato: 31 },
  trombones: { sustain: 20, legato: 15, staccato: 18, marcato: 17 },
  tuba: { sustain: 40, legato: 35, staccato: 38, marcato: 37 },

  // Woodwinds
  flutes: { sustain: 191, staccato: 190 },
  oboes: { sustain: 200, staccato: 199 },
  clarinets: { sustain: 173, staccato: 172 },
  bassoons: { sustain: 164, staccato: 163 },

  // Percussion & Other (no articulation variants)
  harp: { sustain: 86 },
  timpani: { sustain: 93 },
  glockenspiel: { sustain: 92 },
  xylophone: { sustain: 95 },
  chimes: { sustain: 89 },
  chorus: { sustain: 84 },
}

/**
 * Aliases mapping various instrument names to canonical Sonatina keys.
 */
const instrumentAliases: Record<string, string> = {
  // Strings
  violin: 'violin',
  violins: 'violin',
  violin_1: 'violin',
  violin_2: 'violin',
  '1st_violins': 'violin',
  '2nd_violins': 'violin',
  vln: 'violin',
  vln1: 'violin',
  vln2: 'violin',

  viola: 'violas',
  violas: 'violas',
  vla: 'violas',

  cello: 'celli',
  celli: 'celli',
  cellos: 'celli',
  vc: 'celli',

  contrabass: 'basses',
  bass: 'basses',
  basses: 'basses',
  double_bass: 'basses',
  cb: 'basses',

  strings: 'strings',
  string_ensemble: 'strings',
  string_ensemble_1: 'strings',
  string_ensemble_2: 'strings',

  // Brass
  french_horn: 'horns',
  horn: 'horns',
  horns: 'horns',
  hn: 'horns',

  trumpet: 'trumpets',
  trumpets: 'trumpets',
  tpt: 'trumpets',

  trombone: 'trombones',
  trombones: 'trombones',
  tbn: 'trombones',

  tuba: 'tuba',
  tba: 'tuba',

  brass: 'horns', // Default brass to horns
  brass_section: 'horns',

  // Woodwinds
  flute: 'flutes',
  flutes: 'flutes',
  fl: 'flutes',
  piccolo: 'flutes', // Use flute samples for piccolo

  oboe: 'oboes',
  oboes: 'oboes',
  ob: 'oboes',
  english_horn: 'oboes', // Use oboe samples

  clarinet: 'clarinets',
  clarinets: 'clarinets',
  cl: 'clarinets',

  bassoon: 'bassoons',
  bassoons: 'bassoons',
  bn: 'bassoons',

  // Percussion & Other
  harp: 'harp',
  orchestral_harp: 'harp',

  timpani: 'timpani',
  timp: 'timpani',

  glockenspiel: 'glockenspiel',
  celesta: 'glockenspiel',
  vibraphone: 'glockenspiel',

  xylophone: 'xylophone',
  marimba: 'xylophone',

  tubular_bells: 'chimes',

  choir: 'chorus',
  choir_aahs: 'chorus',
  voice_oohs: 'chorus',
  chorus: 'chorus',
  synth_choir: 'chorus',

  // GM defaults (when no instrument specified in ABC)
  acoustic_grand_piano: 'strings', // Default to strings, not piano
  bright_acoustic_piano: 'strings',
  electric_grand_piano: 'strings',
  piano: 'strings',
}

/**
 * Parse articulation suffix from instrument name.
 * Supports formats like "cello_staccato", "VcStac", "celli_pizz"
 */
function parseArticulation(name: string): { instrument: string; articulation: Articulation } {
  const lower = name.toLowerCase()

  // Check for articulation suffixes
  const articulationPatterns: Array<{ pattern: RegExp; articulation: Articulation }> = [
    { pattern: /_?stac(cato)?$/i, articulation: 'staccato' },
    { pattern: /_?pizz(icato)?$/i, articulation: 'pizzicato' },
    { pattern: /_?marc(ato)?$/i, articulation: 'marcato' },
    { pattern: /_?leg(ato)?$/i, articulation: 'legato' },
    { pattern: /_?sus(tain)?$/i, articulation: 'sustain' },
  ]

  for (const { pattern, articulation } of articulationPatterns) {
    if (pattern.test(lower)) {
      const instrument = lower.replace(pattern, '')
      return { instrument, articulation }
    }
  }

  return { instrument: lower, articulation: 'sustain' }
}

/**
 * Normalize an instrument name for lookup.
 */
function normalizeInstrumentName(name: string): string {
  return name.toLowerCase().replace(/\s+/g, '_')
}

/**
 * Get the Sonatina preset number for an instrument name.
 *
 * Supports articulation suffixes:
 * - "cello" → Celli Sustain (109)
 * - "cello_staccato" or "cello_stac" → Celli Staccato (108)
 * - "VcStac" → Celli Staccato (108)
 * - "violas_pizz" → Violas Pizzicato (116)
 *
 * @param instrumentName - The instrument name, optionally with articulation suffix
 * @returns The Sonatina preset number
 */
export function getProgram(instrumentName: string): number {
  const normalized = normalizeInstrumentName(instrumentName)
  const { instrument, articulation } = parseArticulation(normalized)

  // Try to find the canonical instrument name
  let canonicalName = instrumentAliases[instrument]

  // If not found, try removing trailing numbers
  if (!canonicalName) {
    const withoutNumbers = instrument.replace(/_?\d+$/, '')
    canonicalName = instrumentAliases[withoutNumbers]
  }

  // If still not found, default to strings
  if (!canonicalName) {
    console.warn(`[instrument-map] Unknown instrument "${instrumentName}", defaulting to strings`)
    canonicalName = 'strings'
  }

  const presets = sonatina[canonicalName]
  if (!presets) {
    console.warn(`[instrument-map] No presets for "${canonicalName}", defaulting to strings`)
    return sonatina.strings.sustain
  }

  // Get the requested articulation, falling back to sustain
  const preset = presets[articulation] ?? presets.sustain
  return preset
}

/**
 * Get preset for a specific instrument and articulation.
 * More explicit than getProgram() - doesn't parse suffixes.
 */
export function getPreset(instrument: string, articulation: Articulation = 'sustain'): number {
  const normalized = normalizeInstrumentName(instrument)
  const canonicalName = instrumentAliases[normalized] ?? 'strings'
  const presets = sonatina[canonicalName]

  if (!presets) {
    return sonatina.strings.sustain
  }

  return presets[articulation] ?? presets.sustain
}

/**
 * Map from ABCJS voice ID abbreviations to Sonatina preset numbers.
 * These match the V: voice IDs used in ABC notation (e.g., V:Vln1, V:Hn).
 */
const voiceIdToProgram: Record<string, number> = {
  // Strings - default to sustain for voice IDs
  vln: sonatina.violin.sustain,
  vln1: sonatina.violin.sustain,
  vln2: sonatina.violin.sustain,
  vla: sonatina.violas.sustain,
  vc: sonatina.celli.sustain,
  cb: sonatina.basses.sustain,

  // Strings - staccato variants
  vlnstac: sonatina.violin.staccato!,
  vlastac: sonatina.violas.staccato!,
  vcstac: sonatina.celli.staccato!,
  cbstac: sonatina.basses.staccato!,

  // Brass
  hn: sonatina.horns.sustain,
  tpt: sonatina.trumpets.sustain,
  tbn: sonatina.trombones.sustain,
  tba: sonatina.tuba.sustain,

  // Woodwinds
  fl: sonatina.flutes.sustain,
  ob: sonatina.oboes.sustain,
  cl: sonatina.clarinets.sustain,
  bn: sonatina.bassoons.sustain,

  // Percussion
  timp: sonatina.timpani.sustain,
}

/**
 * Get the Sonatina preset number for a voice ID.
 */
export function getProgramFromVoiceId(voiceId: string): number {
  const normalized = voiceId.toLowerCase()

  if (voiceIdToProgram[normalized] !== undefined) {
    return voiceIdToProgram[normalized]
  }

  // Try without trailing numbers
  const withoutNumbers = normalized.replace(/\d+$/, '')
  if (voiceIdToProgram[withoutNumbers] !== undefined) {
    return voiceIdToProgram[withoutNumbers]
  }

  // Default to All Strings Sustain
  return sonatina.strings.sustain
}

/**
 * Get the Sonatina preset for a track based on voice info.
 */
export function getProgramForTrack(
  trackIndex: number,
  voiceInfo: Array<{ id: string; name?: string }>
): number {
  const voice = voiceInfo[trackIndex]
  if (!voice) {
    return sonatina.strings.sustain
  }

  // Try voice name first (more descriptive)
  if (voice.name) {
    return getProgram(voice.name)
  }

  // Fall back to voice ID
  return getProgramFromVoiceId(voice.id)
}
