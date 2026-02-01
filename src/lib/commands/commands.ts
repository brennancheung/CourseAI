import { Command, CommandContext } from './types'

export const getCommands = (context: CommandContext): Command[] => {
  const { router, actions } = context

  const commands: Command[] = [
    // Navigation Commands
    {
      id: 'go-dashboard',
      title: 'Go to Dashboard',
      category: 'Navigation',
      icon: 'home',
      keywords: ['dashboard', 'home', 'main', 'overview'],
      action: () => router.push('/app'),
    },
    {
      id: 'go-upload',
      title: 'Go to Upload',
      category: 'Navigation',
      icon: 'upload',
      keywords: ['upload', 'add', 'new'],
      action: () => router.push('/app/upload'),
    },
    {
      id: 'go-library',
      title: 'Go to Library',
      category: 'Navigation',
      icon: 'library',
      keywords: ['library', 'files', 'content', 'media'],
      action: () => router.push('/app/library'),
    },
    {
      id: 'go-usage',
      title: 'Go to Usage',
      category: 'Navigation',
      icon: 'bar-chart',
      keywords: ['usage', 'storage', 'stats', 'analytics'],
      action: () => router.push('/app/usage'),
    },
    {
      id: 'go-settings',
      title: 'Go to Settings',
      category: 'Navigation',
      icon: 'settings',
      keywords: ['settings', 'preferences', 'configuration', 'options'],
      action: () => router.push('/app/settings'),
    },

    // View Commands
    {
      id: 'set-theme-light',
      title: 'Set Light Theme',
      category: 'View',
      icon: 'sun',
      keywords: ['light', 'theme', 'mode', 'bright'],
      action: () => {
        actions?.setTheme?.('light')
      },
    },
    {
      id: 'set-theme-dark',
      title: 'Set Dark Theme',
      category: 'View',
      icon: 'moon',
      keywords: ['dark', 'theme', 'mode', 'night'],
      action: () => {
        actions?.setTheme?.('dark')
      },
    },
    {
      id: 'set-theme-system',
      title: 'Use System Theme',
      category: 'View',
      icon: 'laptop',
      keywords: ['system', 'theme', 'mode', 'auto', 'automatic'],
      action: () => {
        actions?.setTheme?.('system')
      },
    },

    // Help Commands
    {
      id: 'show-shortcuts',
      title: 'Show Keyboard Shortcuts',
      category: 'Help',
      icon: 'keyboard',
      keywords: ['keyboard', 'shortcuts', 'keys', 'help', 'hotkeys'],
      shortcut: { mac: 'cmd+/', windows: 'ctrl+/' },
      action: () => {
        // TODO: Implement shortcuts modal
        console.log('Show shortcuts')
      },
    },
  ]

  // Filter out commands with visible: false
  return commands.filter(cmd => cmd.visible !== false)
}
