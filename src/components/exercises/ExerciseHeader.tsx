import { type Exercise } from '@/lib/exercises'

interface ExerciseHeaderProps {
  exercise: Exercise
}

/**
 * Consistent header for exercise pages.
 * Displays: title, category tag + duration, description
 */
export function ExerciseHeader({ exercise }: ExerciseHeaderProps) {
  return (
    <div>
      <h1 className="text-2xl font-bold">{exercise.title}</h1>
      <div className="flex items-center gap-2 mt-2">
        <span className="text-xs px-2 py-1 rounded-full bg-muted">
          {exercise.category}
        </span>
        <span className="text-xs text-muted-foreground">~{exercise.duration}</span>
      </div>
      <p className="text-muted-foreground mt-3">{exercise.description}</p>
    </div>
  )
}
