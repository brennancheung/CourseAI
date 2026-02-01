'use client'

import { ScorePlayer } from '@/components/score/ScorePlayer'
import { DebugScorePlayer } from '@/components/score/DebugScorePlayer'

/**
 * Test page to verify ABC notation rendering behavior
 * Each example tests a specific edge case
 */

const tests = [
  {
    name: 'Test 1: Whole notes (L:1/1)',
    description: 'Base length = whole note, chord duration = 1',
    abc: `X:1
T:Whole Notes (L:1/1)
M:4/4
L:1/1
Q:1/4=60
K:Am
"Am"[A,CE]1 | "Dm"[D,FA]1 | "Am"[A,CE]1 | "E"[E,^GB]1 |`,
  },
  {
    name: 'Test 2: Quarter notes as whole bars (L:1/4, duration 4)',
    description: 'Base length = quarter, chord duration = 4 quarters = whole note',
    abc: `X:1
T:Quarter Base, Duration 4 (L:1/4)
M:4/4
L:1/4
Q:1/4=60
K:Am
"Am"[A,CE]4 | "Dm"[D,FA]4 | "Am"[A,CE]4 | "E"[E,^GB]4 |`,
  },
  {
    name: 'Test 3: Actual quarter notes (L:1/4, duration 1)',
    description: 'Base length = quarter, 4 separate quarter note chords per bar',
    abc: `X:1
T:Quarter Notes (L:1/4)
M:4/4
L:1/4
Q:1/4=60
K:Am
"Am"[A,CE] [A,CE] [A,CE] [A,CE] | "Dm"[D,FA] [D,FA] [D,FA] [D,FA] | "Am"[A,CE] [A,CE] [A,CE] [A,CE] | "E"[E,^GB] [E,^GB] [E,^GB] [E,^GB] |`,
  },
  {
    name: 'Test 4: Eighth notes (L:1/8)',
    description: '8 eighth note chords per bar',
    abc: `X:1
T:Eighth Notes (L:1/8)
M:4/4
L:1/8
Q:1/4=60
K:Am
"Am"[A,CE] [A,CE] [A,CE] [A,CE] [A,CE] [A,CE] [A,CE] [A,CE] | "Dm"[D,FA] [D,FA] [D,FA] [D,FA] [D,FA] [D,FA] [D,FA] [D,FA] |`,
  },
  {
    name: 'Test 5: Multi-voice (2 staves)',
    description: 'Separate treble and bass voices',
    abc: `X:1
T:Multi-Voice (2 Staves)
M:4/4
L:1/4
Q:1/4=60
K:Am
%%staves {(Treble) (Bass)}
V:Treble clef=treble name="Treble"
V:Bass clef=bass name="Bass"
%
[V:Treble] "Am"[CE]4 | "Dm"[FA]4 | "Am"[CE]4 | "E"[^GB]4 |
[V:Bass] A,4 | D,4 | A,4 | E,4 |`,
  },
  {
    name: 'Test 6: Multi-voice with rhythm',
    description: 'Treble holds, bass plays quarter notes',
    abc: `X:1
T:Multi-Voice with Rhythm
M:4/4
L:1/4
Q:1/4=60
K:Am
%%staves {(Treble) (Bass)}
V:Treble clef=treble name="Treble"
V:Bass clef=bass name="Bass"
%
[V:Treble] "Am"[CEA]4 | "Dm"[DFA]4 |
[V:Bass] A, A, A, A, | D, D, D, D, |`,
  },
  {
    name: 'Test 7: String quartet voices',
    description: '4 separate instrument voices',
    abc: `X:1
T:String Quartet Voices
M:4/4
L:1/4
Q:1/4=60
K:Am
%%staves {(V1) (V2) (Va) (Vc)}
V:V1 clef=treble name="Violin 1"
V:V2 clef=treble name="Violin 2"
V:Va clef=alto name="Viola"
V:Vc clef=bass name="Cello"
%
[V:V1] e4 | f4 | e4 | e4 |
[V:V2] c4 | d4 | c4 | B4 |
[V:Va] A4 | A4 | A4 | ^G4 |
[V:Vc] A,4 | D,4 | A,4 | E,4 |`,
  },
  {
    name: 'Test 8: The Ominous (current)',
    description: 'Current notation from progression-scores.ts',
    abc: `X:1
T:The Ominous (i-iv-i-V)
M:4/4
L:1/4
Q:1/4=54
K:Am
"Am (i)"[A,CEA]4 | "Dm (iv)"[D,DFA]4 | "Am (i)"[A,CEA]4 | "E (V)"[E,E^GB]4 |
"Am (i)"[A,CEA]4 | "Dm (iv)"[D,DFA]4 | "Am (i)"[A,CEA]4 | "E (V)"[E,E^GBe]4 |`,
  },
]

// Test WITHOUT chord symbols
const debugNoChords = `X:1
T:Debug: Whole Notes WITHOUT Chord Symbols
M:4/4
L:1/1
Q:1/4=60
K:Am
[A,CE]1 | [D,FA]1 | [A,CE]1 | [E,^GB]1 |`

// Test WITH chord symbols (same notes)
const debugWithChords = `X:1
T:Debug: Whole Notes WITH Chord Symbols
M:4/4
L:1/1
Q:1/4=60
K:Am
"Am"[A,CE]1 | "Dm"[D,FA]1 | "Am"[A,CE]1 | "E"[E,^GB]1 |`

export default function TestNotationPage() {
  return (
    <div className="container max-w-4xl py-8 space-y-8">
      <div>
        <h1 className="text-2xl font-bold">ABC Notation Test Page</h1>
        <p className="text-muted-foreground mt-2">
          Testing different ABC notation patterns to verify rendering and playback behavior.
        </p>
      </div>

      {/* Debug section - comparing WITH vs WITHOUT chord symbols */}
      <div className="border-2 border-yellow-500 rounded-lg p-4 space-y-6 bg-yellow-500/10">
        <h2 className="font-semibold text-yellow-600 text-lg">DEBUG: Chord Symbol Hypothesis</h2>
        <p className="text-sm text-muted-foreground">
          Testing if chord symbols (&quot;Am&quot;, &quot;Dm&quot;) cause ABCJS to auto-generate accompaniment.
        </p>

        <div className="space-y-4">
          <div className="border rounded p-3 bg-background">
            <h3 className="font-medium text-green-600">WITHOUT Chord Symbols (should be correct)</h3>
            <pre className="mt-2 p-2 bg-muted rounded text-xs overflow-x-auto">{debugNoChords}</pre>
            <DebugScorePlayer abc={debugNoChords} initialTempo={60} />
          </div>

          <div className="border rounded p-3 bg-background">
            <h3 className="font-medium text-red-600">WITH Chord Symbols (might add accompaniment)</h3>
            <pre className="mt-2 p-2 bg-muted rounded text-xs overflow-x-auto">{debugWithChords}</pre>
            <DebugScorePlayer abc={debugWithChords} initialTempo={60} />
          </div>
        </div>
      </div>

      {tests.map((test, i) => (
        <div key={i} className="border rounded-lg p-4 space-y-3">
          <div>
            <h2 className="font-semibold">{test.name}</h2>
            <p className="text-sm text-muted-foreground">{test.description}</p>
          </div>
          <details className="text-xs">
            <summary className="cursor-pointer text-muted-foreground hover:text-foreground">
              Show ABC source
            </summary>
            <pre className="mt-2 p-2 bg-muted rounded text-xs overflow-x-auto">
              {test.abc}
            </pre>
          </details>
          <ScorePlayer abc={test.abc} initialTempo={60} />
        </div>
      ))}
    </div>
  )
}
