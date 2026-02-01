interface PageHeaderProps {
  title: string
  description: string
}

/**
 * Consistent page header for app pages.
 * Displays title and description with standard styling.
 */
export function PageHeader({ title, description }: PageHeaderProps) {
  return (
    <div>
      <h1 className="text-2xl font-bold">{title}</h1>
      <p className="text-muted-foreground">{description}</p>
    </div>
  )
}
