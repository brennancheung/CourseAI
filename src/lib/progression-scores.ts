/**
 * ABC notation scores for cinematic chord progressions
 *
 * Each progression uses multi-voice orchestral notation with:
 * - Appropriate instrumentation for the emotional context
 * - Rhythmic drive (pulsing bass/low instruments)
 * - Melodic movement and interest
 * - Realistic orchestration patterns
 *
 * IMPORTANT: ScorePlayer must use chordsOff: true to prevent ABCJS from
 * auto-generating accompaniment from chord symbols. See docs/abcjs-synth-behavior.md
 */

export type ProgressionFamily =
  | 'epic-triumphant'
  | 'emotional-melancholic'
  | 'dark-tension'
  | 'uplifting-major'

export interface ProgressionScore {
  id: string
  name: string
  numerals: string
  family: ProgressionFamily
  /** Short description of the feel */
  description: string
  /** When to use this progression */
  mood: string
  /** Example chord names in a specific key */
  example: string
  /** Explanation of why this progression works emotionally */
  whyItWorks: string
  /** Optional tempo notes */
  tempoNotes?: string
  abc: string
  tempo: number
}

export const progressionScores: ProgressionScore[] = [
  // ============================================
  // EPIC/TRIUMPHANT FAMILY
  // ============================================
  {
    id: 'epic',
    name: 'The Epic',
    numerals: 'i - VI - III - VII',
    family: 'epic-triumphant',
    description: 'Powerful, triumphant, forward momentum',
    mood: 'Hero moments, triumph over adversity, epic reveals',
    example: 'Am → F → C → G',
    whyItWorks:
      'The VI creates lift, III provides brightness, VII builds tension that wants to resolve back to i. The constant upward motion (i→VI is up, III→VII is up) creates a sense of rising/ascending.',
    tempoNotes: 'Works at any tempo. Slow = majestic. Fast = action/chase.',
    tempo: 130,
    abc: `X:1
T:The Epic (i-VI-III-VII)
M:4/4
L:1/8
Q:1/4=130
K:Am
%%staves {(Hn) (Vln1) (Vla) (Vc)}
V:Hn clef=treble name="Horns"
V:Vln1 clef=treble name="Violins"
V:Vla clef=alto name="Violas"
V:Vc clef=bass name="Celli"
%
[V:Hn] "Am (i)"A4 c4 | "F (VI)"A4 c4 | "C (III)"G4 c4 | "G (VII)"G4 B4 |
[V:Vln1] e2 e2 a2 a2 | c2 c2 f2 f2 | e2 e2 g2 g2 | d2 d2 g2 g2 |
[V:Vla] C2 E2 A2 E2 | F,2 A,2 C2 A,2 | C2 E2 G2 E2 | B,2 D2 G2 D2 |
[V:Vc] A,2 A,2 A,2 A,2 | F,2 F,2 F,2 F,2 | C,2 C,2 C,2 C,2 | G,2 G,2 G,2 G,2 |`,
  },
  {
    id: 'building-hero',
    name: 'The Building Hero',
    numerals: 'i - VII - VI - VII',
    family: 'epic-triumphant',
    description: 'Building anticipation, gathering strength',
    mood: 'Preparing for battle, montage sequences, building to a climax',
    example: 'Am → G → F → G',
    whyItWorks:
      'The oscillation between VI and VII creates tension without resolution. You\'re stuck in "almost there" territory, perfect for building momentum.',
    tempo: 108,
    abc: `X:1
T:Building Hero (i-VII-VI-VII)
M:4/4
L:1/8
Q:1/4=108
K:Am
%%staves {(Tpt) (Vln1) (Vla) (Vc)}
V:Tpt clef=treble name="Trumpets"
V:Vln1 clef=treble name="Violins"
V:Vla clef=alto name="Violas"
V:Vc clef=bass name="Celli"
%
[V:Tpt] "Am (i)"E4 A4 | "G (VII)"D4 G4 | "F (VI)"C4 F4 | "G (VII)"D4 B4 |
[V:Vln1] A,2 C2 E2 A2 | G,2 B,2 D2 G2 | F,2 A,2 C2 F2 | G,2 B,2 D2 G2 |
[V:Vla] C2 C2 E2 E2 | B,2 B,2 D2 D2 | A,2 A,2 C2 C2 | B,2 B,2 D2 D2 |
[V:Vc] A,2 A,2 A,2 A,2 | G,2 G,2 G,2 G,2 | F,2 F,2 F,2 F,2 | G,2 G,2 G,2 G,2 |`,
  },
  {
    id: 'arrival',
    name: 'The Arrival',
    numerals: 'VI - VII - i',
    family: 'epic-triumphant',
    description: 'Triumphant arrival, resolution, "we made it"',
    mood: 'Victory moment, returning home, achieving the goal',
    example: 'F → G → Am',
    whyItWorks:
      'This is the payoff. VII→i is the strongest resolution in minor keys. Starting on VI gives you runway to build into the final resolution.',
    tempo: 88,
    abc: `X:1
T:The Arrival (VI-VII-i)
M:4/4
L:1/8
Q:1/4=88
K:Am
%%staves {(Hn) (Tpt) (Vln1) (Vc)}
V:Hn clef=treble name="Horns"
V:Tpt clef=treble name="Trumpets"
V:Vln1 clef=treble name="Violins"
V:Vc clef=bass name="Celli"
%
[V:Hn] "F (VI)"F4 A4 | "G (VII)"G4 B4 | "Am (i)"A8 |
[V:Tpt] c2 c2 f2 f2 | d2 d2 g2 g2 | e2 a2 e2 a2 |
[V:Vln1] A,2 C2 F2 A2 | B,2 D2 G2 B2 | C2 E2 A2 c2 |
[V:Vc] F,2 F,2 F,2 F,2 | G,2 G,2 G,2 G,2 | A,2 A,2 A,4 |`,
  },

  // ============================================
  // EMOTIONAL/MELANCHOLIC FAMILY
  // ============================================
  {
    id: 'sad-epic',
    name: 'The Sad Epic',
    numerals: 'i - iv - VII - III',
    family: 'emotional-melancholic',
    description: 'Bittersweet, emotional, nostalgic triumph',
    mood: 'Sacrifice scenes, pyrrhic victories, emotional flashbacks',
    example: 'Am → Dm → G → C',
    whyItWorks:
      'The iv (minor subdominant) adds weight and sadness. The move to III at the end opens up rather than resolves—leaves you hanging emotionally.',
    tempo: 76,
    abc: `X:1
T:The Sad Epic (i-iv-VII-III)
M:4/4
L:1/8
Q:1/4=76
K:Am
%%staves {(Vln1) (Vln2) (Vla) (Vc)}
V:Vln1 clef=treble name="Violin 1"
V:Vln2 clef=treble name="Violin 2"
V:Vla clef=alto name="Viola"
V:Vc clef=bass name="Cello"
%
[V:Vln1] "Am (i)"e4 a4 | "Dm (iv)"f4 a4 | "G (VII)"g4 b4 | "C (III)"g4 c'4 |
[V:Vln2] c2 e2 e2 a2 | d2 f2 f2 a2 | d2 g2 g2 b2 | e2 g2 g2 c'2 |
[V:Vla] A,2 C2 E2 C2 | D,2 F,2 A,2 F,2 | G,2 B,2 D2 B,2 | C2 E2 G2 E2 |
[V:Vc] A,2 A,2 A,2 A,2 | D,2 D,2 D,2 D,2 | G,2 G,2 G,2 G,2 | C,2 C,2 C,2 C,2 |`,
  },
  {
    id: 'emotional-build',
    name: 'The Emotional Build',
    numerals: 'i - VI - iv - VII',
    family: 'emotional-melancholic',
    description: 'Deeply emotional, swelling, cinematic crying moment',
    mood: 'Death scenes, reunions, emotional peaks',
    example: 'Am → F → Dm → G',
    whyItWorks:
      'Combines the lift of VI with the weight of iv. The VII at the end keeps it unresolved—perfect for sustaining emotion without closure.',
    tempo: 69,
    abc: `X:1
T:Emotional Build (i-VI-iv-VII)
M:4/4
L:1/8
Q:1/4=69
K:Am
%%staves {(Vln1) (Vln2) (Vla) (Vc)}
V:Vln1 clef=treble name="Violin 1"
V:Vln2 clef=treble name="Violin 2"
V:Vla clef=alto name="Viola"
V:Vc clef=bass name="Cello"
%
[V:Vln1] "Am (i)"e3 f e2 a2 | "F (VI)"c3 d c2 f2 | "Dm (iv)"d3 e d2 a2 | "G (VII)"d3 e d2 g2 |
[V:Vln2] c2 c2 e2 e2 | A2 A2 c2 c2 | A2 A2 d2 d2 | B2 B2 d2 d2 |
[V:Vla] A,2 C2 E2 C2 | F,2 A,2 C2 A,2 | D,2 F,2 A,2 F,2 | G,2 B,2 D2 B,2 |
[V:Vc] A,2 A,2 A,2 A,2 | F,2 F,2 F,2 F,2 | D,2 D,2 D,2 D,2 | G,2 G,2 G,2 G,2 |`,
  },
  {
    id: 'hopeful-sad',
    name: 'The Hopeful Sad',
    numerals: 'i - v - VI - VII',
    family: 'emotional-melancholic',
    description: 'Sad but with hope, light at the end of the tunnel',
    mood: 'Character finding hope, dawn after darkness',
    example: 'Am → Em → F → G',
    whyItWorks:
      'The minor v keeps it melancholic, but the VI-VII ending lifts upward. Creates a sense of hope emerging from sadness.',
    tempo: 80,
    abc: `X:1
T:The Hopeful Sad (i-v-VI-VII)
M:4/4
L:1/8
Q:1/4=80
K:Am
%%staves {(Vln1) (Vln2) (Vla) (Vc)}
V:Vln1 clef=treble name="Violin 1"
V:Vln2 clef=treble name="Violin 2"
V:Vla clef=alto name="Viola"
V:Vc clef=bass name="Cello"
%
[V:Vln1] "Am (i)"a4 e4 | "Em (v)"g4 e4 | "F (VI)"a4 f4 | "G (VII)"b4 g4 |
[V:Vln2] c2 e2 a2 e2 | B2 e2 g2 e2 | c2 f2 a2 f2 | d2 g2 b2 g2 |
[V:Vla] A,2 C2 E2 A2 | E,2 G,2 B,2 E2 | F,2 A,2 C2 F2 | G,2 B,2 D2 G2 |
[V:Vc] A,2 A,2 A,2 A,2 | E,2 E,2 E,2 E,2 | F,2 F,2 F,2 F,2 | G,2 G,2 G,2 G,2 |`,
  },

  // ============================================
  // DARK/TENSION FAMILY
  // ============================================
  {
    id: 'descending-doom',
    name: 'The Descending Doom',
    numerals: 'i - VII - VI - V',
    family: 'dark-tension',
    description: 'Inevitable doom, descending into darkness, fate closing in',
    mood: 'Villain reveals, approaching danger, tragic realization',
    example: 'Am → G → F → E',
    whyItWorks:
      'Stepwise descending bass (A→G→F→E) creates a sense of falling. The V (major) at the end wants to resolve to i, creating unresolved tension.',
    tempo: 66,
    abc: `X:1
T:Descending Doom (i-VII-VI-V)
M:4/4
L:1/8
Q:1/4=66
K:Am
%%staves {(Tbn) (Vla) (Vc) (Cb)}
V:Tbn clef=bass name="Trombones"
V:Vla clef=alto name="Violas"
V:Vc clef=bass name="Celli"
V:Cb clef=bass name="Basses"
%
[V:Tbn] "Am (i)"A,4 E4 | "G (VII)"G,4 D4 | "F (VI)"F,4 C4 | "E (V)"E,4 B,4 |
[V:Vla] C2 E2 A2 E2 | B,2 D2 G2 D2 | A,2 C2 F2 C2 | ^G,2 B,2 E2 B,2 |
[V:Vc] A,2 A,2 A,2 A,2 | G,2 G,2 G,2 G,2 | F,2 F,2 F,2 F,2 | E,2 E,2 E,2 E,2 |
[V:Cb] A,,2 A,,2 A,,2 A,,2 | G,,2 G,,2 G,,2 G,,2 | F,,2 F,,2 F,,2 F,,2 | E,,2 E,,2 E,,2 E,,2 |`,
  },
  {
    id: 'classical-tension',
    name: 'The Classical Tension',
    numerals: 'i - ii° - V - i',
    family: 'dark-tension',
    description: 'Classical drama, sophisticated tension and release',
    mood: 'Period pieces, formal dramatic moments',
    example: 'Am → Bdim → E → Am',
    whyItWorks:
      'The diminished chord creates maximum instability, V→i is the strongest resolution. A classical, formal sound that feels inevitable.',
    tempo: 76,
    abc: `X:1
T:Classical Tension (i-ii°-V-i)
M:4/4
L:1/8
Q:1/4=76
K:Am
%%staves {(Vln1) (Vln2) (Vla) (Vc)}
V:Vln1 clef=treble name="Violin 1"
V:Vln2 clef=treble name="Violin 2"
V:Vla clef=alto name="Viola"
V:Vc clef=bass name="Cello"
%
[V:Vln1] "Am (i)"e2 c2 e2 a2 | "Bdim (ii°)"f2 d2 f2 b2 | "E (V)"^g2 e2 ^g2 b2 | "Am (i)"a8 |
[V:Vln2] A2 E2 A2 c2 | B2 F2 B2 d2 | B2 ^G2 B2 e2 | c2 A2 e2 a2 |
[V:Vla] C2 A,2 C2 E2 | D2 B,2 D2 F2 | E2 B,2 E2 ^G2 | E2 C2 A2 c2 |
[V:Vc] A,2 A,2 A,2 A,2 | B,2 B,2 B,2 B,2 | E,2 E,2 E,2 E,2 | A,2 A,2 A,4 |`,
  },
  {
    id: 'ominous',
    name: 'The Ominous',
    numerals: 'i - iv - i - V',
    family: 'dark-tension',
    description: 'Brooding, ominous, something bad is coming',
    mood: 'Suspense building, villain scheming, dark atmosphere',
    example: 'Am → Dm → Am → E',
    whyItWorks:
      'Staying close to home (i and iv) with the V creating unresolved tension. The return to i before V makes the tension feel cyclical and inescapable.',
    tempo: 72,
    abc: `X:1
T:The Ominous (i-iv-i-V)
M:4/4
L:1/8
Q:1/4=72
K:Am
%%staves {(Vln1) (Vln2) (Vla) (Vc)}
V:Vln1 clef=treble name="Violin 1"
V:Vln2 clef=treble name="Violin 2"
V:Vla clef=alto name="Viola"
V:Vc clef=bass name="Cello"
%
[V:Vln1] "Am (i)"E4 A4 | "Dm (iv)"F4 A4 | "Am (i)"E4 A4 | "E (V)"^G4 B4 |
[V:Vln2] C4 E4 | D4 F4 | C4 E4 | E4 ^G4 |
[V:Vla] A,2 A,2 C2 E2 | D,2 D,2 F2 A2 | A,2 A,2 C2 E2 | E,2 E,2 ^G2 B2 |
[V:Vc] A,,2 A,,2 A,,2 A,,2 | D,,2 D,,2 D,,2 D,,2 | A,,2 A,,2 A,,2 A,,2 | E,,2 E,,2 E,,2 E,,2 |`,
  },

  // ============================================
  // UPLIFTING/MAJOR FAMILY
  // ============================================
  {
    id: 'pop-epic',
    name: 'The Pop Epic',
    numerals: 'I - V - vi - IV',
    family: 'uplifting-major',
    description: 'Uplifting, anthemic, universally emotional',
    mood: 'Inspirational moments, sports victories, overcoming odds',
    example: 'C → G → Am → F',
    whyItWorks:
      'The vi (minor) provides emotional depth in an otherwise major context. Extremely versatile—the most used progression in popular music for a reason.',
    tempo: 120,
    abc: `X:1
T:The Pop Epic (I-V-vi-IV)
M:4/4
L:1/8
Q:1/4=120
K:C
%%staves {(Hn) (Vln1) (Vla) (Vc)}
V:Hn clef=treble name="Horns"
V:Vln1 clef=treble name="Violins"
V:Vla clef=alto name="Violas"
V:Vc clef=bass name="Celli"
%
[V:Hn] "C (I)"G4 c4 | "G (V)"G4 B4 | "Am (vi)"A4 c4 | "F (IV)"A4 c4 |
[V:Vln1] e2 e2 g2 g2 | d2 d2 g2 g2 | c2 c2 e2 e2 | c2 c2 f2 f2 |
[V:Vla] C2 E2 G2 E2 | B,2 D2 G2 D2 | A,2 C2 E2 C2 | F,2 A,2 C2 A,2 |
[V:Vc] C,2 C,2 C,2 C,2 | G,2 G,2 G,2 G,2 | A,2 A,2 A,2 A,2 | F,2 F,2 F,2 F,2 |`,
  },
  {
    id: 'journey',
    name: 'The Journey',
    numerals: 'I - IV - vi - V',
    family: 'uplifting-major',
    description: 'Adventure, setting out, optimistic journey',
    mood: 'Opening scenes, adventure beginnings, road trips',
    example: 'C → F → Am → G',
    whyItWorks:
      'IV creates forward motion, vi adds depth, V keeps it moving. The perfect "setting out on an adventure" feel.',
    tempo: 116,
    abc: `X:1
T:The Journey (I-IV-vi-V)
M:4/4
L:1/8
Q:1/4=116
K:C
%%staves {(Hn) (Vln1) (Vla) (Vc)}
V:Hn clef=treble name="Horns"
V:Vln1 clef=treble name="Violins"
V:Vla clef=alto name="Violas"
V:Vc clef=bass name="Celli"
%
[V:Hn] "C (I)"E4 G4 | "F (IV)"F4 A4 | "Am (vi)"E4 A4 | "G (V)"D4 G4 |
[V:Vln1] c2 e2 g2 c'2 | c2 f2 a2 c'2 | c2 e2 a2 c'2 | B2 d2 g2 b2 |
[V:Vla] G,2 C2 E2 G2 | A,2 C2 F2 A2 | A,2 C2 E2 A2 | G,2 B,2 D2 G2 |
[V:Vc] C,2 C,2 C,2 C,2 | F,2 F,2 F,2 F,2 | A,2 A,2 A,2 A,2 | G,2 G,2 G,2 G,2 |`,
  },
  {
    id: 'sensitive',
    name: 'The Sensitive',
    numerals: 'vi - IV - I - V',
    family: 'uplifting-major',
    description: 'Vulnerable, emotional, intimate epic',
    mood: 'Love themes, personal sacrifice, quiet heroism',
    example: 'Am → F → C → G',
    whyItWorks:
      'Starting on vi (minor) immediately establishes emotional vulnerability, but the major chords provide hope. The same chords as Pop Epic, but the minor start changes everything.',
    tempo: 84,
    abc: `X:1
T:The Sensitive (vi-IV-I-V)
M:4/4
L:1/8
Q:1/4=84
K:C
%%staves {(Vln1) (Vln2) (Vla) (Vc)}
V:Vln1 clef=treble name="Violin 1"
V:Vln2 clef=treble name="Violin 2"
V:Vla clef=alto name="Viola"
V:Vc clef=bass name="Cello"
%
[V:Vln1] "Am (vi)"a3 b a2 e2 | "F (IV)"a3 b a2 f2 | "C (I)"g3 a g2 e2 | "G (V)"g3 a g2 d2 |
[V:Vln2] e2 c2 e2 a2 | f2 c2 f2 a2 | e2 c2 e2 g2 | d2 B2 d2 g2 |
[V:Vla] A,2 C2 E2 A2 | F,2 A,2 C2 F2 | G,2 C2 E2 G2 | G,2 B,2 D2 G2 |
[V:Vc] A,2 A,2 A,2 A,2 | F,2 F,2 F,2 F,2 | C,2 C,2 C,2 C,2 | G,2 G,2 G,2 G,2 |`,
  },
]

export const getProgressionScore = (id: string): ProgressionScore | undefined =>
  progressionScores.find((p) => p.id === id)

export const getProgressionsByFamily = (family: ProgressionFamily): ProgressionScore[] =>
  progressionScores.filter((p) => p.family === family)

export const progressionFamilies: {
  id: ProgressionFamily
  name: string
  description: string
}[] = [
  {
    id: 'epic-triumphant',
    name: 'Epic/Triumphant',
    description: 'Powerful, heroic, forward momentum',
  },
  {
    id: 'emotional-melancholic',
    name: 'Emotional/Melancholic',
    description: 'Bittersweet, nostalgic, deeply felt',
  },
  {
    id: 'dark-tension',
    name: 'Dark/Tension',
    description: 'Ominous, suspenseful, inevitable',
  },
  {
    id: 'uplifting-major',
    name: 'Uplifting/Major',
    description: 'Hopeful, inspiring, anthemic',
  },
]
