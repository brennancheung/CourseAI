import Markdown from 'react-markdown'

interface ExerciseContentProps {
  content: string
  className?: string
}

/**
 * Renders markdown content with consistent exercise styling.
 * Used for long-form educational content in exercises.
 */
export function ExerciseContent({ content, className = '' }: ExerciseContentProps) {
  return (
    <div className={`prose prose-sm dark:prose-invert max-w-none ${className}`}>
      <Markdown
        components={{
          h2: ({ children }) => (
            <h2 className="text-lg font-bold mt-8 mb-4 border-b pb-2">{children}</h2>
          ),
          h3: ({ children }) => (
            <h3 className="text-base font-semibold mt-6 mb-2">{children}</h3>
          ),
          p: ({ children }) => (
            <p className="text-sm text-muted-foreground mb-3">{children}</p>
          ),
          strong: ({ children }) => (
            <strong className="text-foreground font-semibold">{children}</strong>
          ),
          ul: ({ children }) => (
            <ul className="text-sm text-muted-foreground space-y-1 mb-4 pl-5 list-disc">
              {children}
            </ul>
          ),
          li: ({ children }) => <li>{children}</li>,
          hr: () => <hr className="my-6 border-border" />,
        }}
      >
        {content}
      </Markdown>
    </div>
  )
}
