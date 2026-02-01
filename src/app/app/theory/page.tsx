import { CircleOfFifths } from '@/components/theory/CircleOfFifths'
import { ModeToggle } from '@/components/theory/ModeToggle'
import { ScaleHeader } from '@/components/theory/ScaleHeader'
import { ChordViewer } from '@/components/theory/ChordViewer'

export default function TheoryPage() {
  return (
    <div className="space-y-8">
      <div className="space-y-2">
        <h1 className="text-2xl font-bold">Theory Reference</h1>
        <p className="text-muted-foreground">
          Select a key to see its scale and diatonic chords
        </p>
      </div>

      {/* Circle of Fifths and Mode Toggle */}
      <div className="flex flex-col md:flex-row gap-8 items-start">
        <CircleOfFifths />
        <div className="space-y-6">
          <ModeToggle />
          <ScaleHeader />
        </div>
      </div>

      {/* Chord Viewer */}
      <ChordViewer />
    </div>
  )
}
