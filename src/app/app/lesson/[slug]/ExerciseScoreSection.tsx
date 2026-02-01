'use client'

import { ScorePlayer } from '@/components/score'
import type { ExerciseScore } from '@/lib/exercises'

type ExerciseScoreSectionProps = {
  score: ExerciseScore
}

export function ExerciseScoreSection({ score }: ExerciseScoreSectionProps) {
  return (
    <div className="rounded-lg border bg-card p-4 space-y-3">
      <h2 className="font-semibold">Reference Score</h2>
      <p className="text-sm text-muted-foreground">
        Listen to hear what you&apos;re aiming for. Use the controls to adjust tempo.
      </p>
      <ScorePlayer
        abc={score.abc}
        title={score.title}
        initialTempo={score.tempo ?? 120}
      />
    </div>
  )
}
