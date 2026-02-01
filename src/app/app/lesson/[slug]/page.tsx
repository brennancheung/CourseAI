import { getExercise } from '@/lib/exercises'
import { ExerciseScoreSection } from './ExerciseScoreSection'
import { ExerciseHeader } from '@/components/exercises/ExerciseHeader'
import { ExerciseContent } from '@/components/exercises/ExerciseContent'
import { ProgressionPlayer } from '@/components/exercises/ProgressionPlayer'
import { OstinatoLesson } from '@/components/exercises/OstinatoLesson'
import { InstrumentCharacterLesson } from '@/components/exercises/InstrumentCharacterLesson'
import { ZimmerStudyLesson } from '@/components/exercises/ZimmerStudyLesson'
import { DjawadiStudyLesson } from '@/components/exercises/DjawadiStudyLesson'
import { TrailerSoundLesson } from '@/components/exercises/TrailerSoundLesson'
import { DevastatorLesson } from '@/components/exercises/DevastatorLesson'
import { TheoryLesson } from '@/components/exercises/TheoryLesson'
import {
  LessonLayout,
  ObjectiveBlock,
  ConstraintBlock,
  StepList,
  TipBlock,
  NextStepBlock,
} from '@/components/lessons'
import { Row, RowContent, RowAside } from '@/components/layout/Row'

type ExercisePageProps = {
  params: Promise<{ slug: string }>
}

export default async function ExercisePage({ params }: ExercisePageProps) {
  const { slug } = await params
  const exercise = getExercise(slug)

  if (!exercise) {
    return (
      <div className="space-y-4">
        <h1 className="text-2xl font-bold">Exercise not found</h1>
      </div>
    )
  }

  // Custom layouts for lessons with rich interactive content
  if (slug === 'theory-reference') {
    return <TheoryLesson exercise={exercise} />
  }

  if (slug === 'ostinato-patterns') {
    return <OstinatoLesson exercise={exercise} />
  }

  if (slug === 'instrument-character') {
    return <InstrumentCharacterLesson exercise={exercise} />
  }

  if (slug === 'zimmer-study') {
    return <ZimmerStudyLesson exercise={exercise} />
  }

  if (slug === 'djawadi-study') {
    return <DjawadiStudyLesson exercise={exercise} />
  }

  if (slug === 'trailer-sound') {
    return <TrailerSoundLesson exercise={exercise} />
  }

  // Devastator Breakout Pro curriculum
  const devastatorSlugs = [
    'devastator-orientation',
    'anatomy-of-braam',
    'pulse-foundations',
    'tick-tock-tension',
    'hit-stack',
    'custom-risers',
    'sequencer-to-midi',
    'hybrid-hits',
    'building-a-drop',
  ]
  if (devastatorSlugs.includes(slug)) {
    return <DevastatorLesson exercise={exercise} />
  }

  return (
    <LessonLayout>
      {/* Header + Constraints */}
      <Row>
        <RowContent>
          <ExerciseHeader exercise={exercise} />
          {exercise.description && (
            <ObjectiveBlock>{exercise.description}</ObjectiveBlock>
          )}
        </RowContent>
        <RowAside>
          <ConstraintBlock items={exercise.constraints} />
        </RowAside>
      </Row>

      {/* Progression player for cinematic-progressions exercise */}
      {slug === 'cinematic-progressions' && (
        <Row>
          <RowContent>
            <ProgressionPlayer />
          </RowContent>
        </Row>
      )}

      {/* Score visualization if available */}
      {exercise.score && (
        <Row>
          <RowContent>
            <ExerciseScoreSection score={exercise.score} />
          </RowContent>
        </Row>
      )}

      {/* Instructions + Recommended When */}
      <Row>
        <RowContent>
          <StepList steps={exercise.steps} />
        </RowContent>
        {exercise.recommendedWhen && (
          <RowAside>
            <TipBlock title="Best when">
              {exercise.recommendedWhen}
            </TipBlock>
          </RowAside>
        )}
      </Row>

      {/* Long-form content if available */}
      {exercise.content && (
        <Row>
          <RowContent>
            <ExerciseContent content={exercise.content} />
          </RowContent>
        </Row>
      )}

      {/* Complete session */}
      <Row>
        <RowContent>
          <NextStepBlock
            href={`/app/lesson/${slug}/log`}
            description="Review your session with Claude Code to capture what you learned and what was challenging."
          />
        </RowContent>
      </Row>
    </LessonLayout>
  )
}
