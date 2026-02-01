import { formatShortcut } from '@/lib/platform'

interface CommandShortcutProps {
  shortcut?: {
    mac?: string
    windows?: string
  }
}

export const CommandShortcut = ({ shortcut }: CommandShortcutProps) => {
  if (!shortcut) return null

  const formatted = formatShortcut(shortcut)
  if (!formatted) return null

  return (
    <span className="text-xs text-muted-foreground ml-auto">
      {formatted}
    </span>
  )
}
