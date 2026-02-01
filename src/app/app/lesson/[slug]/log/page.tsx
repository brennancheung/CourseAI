import Link from 'next/link'
import { getExercise } from '@/lib/exercises'

export default async function LogSessionPage({
  params,
}: {
  params: Promise<{ slug: string }>
}) {
  const { slug } = await params
  const exercise = getExercise(slug)
  const title = exercise?.title ?? slug

  return (
    <div className="space-y-6 max-w-2xl">
      <Link
        href={`/app/lesson/${slug}`}
        className="text-sm text-muted-foreground hover:text-foreground"
      >
        ← Back to exercise
      </Link>

      <div>
        <h1 className="text-2xl font-bold">Session Complete</h1>
        <p className="text-muted-foreground mt-1">
          Nice work on <span className="text-foreground">{title}</span>.
        </p>
      </div>

      <div className="rounded-lg border bg-card p-6 space-y-4">
        <h2 className="font-semibold">Review with Claude Code</h2>
        <p className="text-sm text-muted-foreground">
          For the best reflection, have a conversation about your session. Open Claude Code and run:
        </p>
        <code className="block bg-muted px-4 py-3 rounded-md text-sm font-mono">/review</code>
        <p className="text-sm text-muted-foreground">
          Talk through what happened, what clicked, and what was hard. The conversation captures
          nuance that forms don&apos;t.
        </p>
      </div>

      <div className="flex gap-3">
        <Link
          href="/app"
          className="inline-flex items-center justify-center rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90"
        >
          Back to Lessons
        </Link>
        <Link
          href="/app/journal"
          className="inline-flex items-center justify-center rounded-md border px-4 py-2 text-sm font-medium hover:bg-accent"
        >
          View Journal
        </Link>
      </div>
    </div>
  )
}
