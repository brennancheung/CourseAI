export interface Command {
  id: string
  title: string
  subtitle?: string
  category: string
  icon?: string // Icon name from lucide-react
  keywords: string[]
  shortcut?: {
    mac?: string
    windows?: string
  }
  action: () => void | Promise<void>
  enabled?: boolean
  visible?: boolean
}

export interface CommandContext {
  router: {
    push: (path: string) => void
  }
  actions?: {
    setTheme?: (theme: string) => void
  }
}
