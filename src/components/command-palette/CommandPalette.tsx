'use client'

import { useState, useCallback, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { useHotkey } from '@/hooks/useHotkey'
import {
  Home,
  Settings,
  Moon,
  Sun,
  Keyboard,
  Laptop,
  LucideIcon,
  Upload,
  Library,
  BarChart3,
} from 'lucide-react'

// Icon mapping - add icons here as needed for your commands
const iconMap: Record<string, LucideIcon> = {
  home: Home,
  settings: Settings,
  moon: Moon,
  sun: Sun,
  keyboard: Keyboard,
  laptop: Laptop,
  upload: Upload,
  library: Library,
  'bar-chart': BarChart3,
}

import {
  Command,
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from '@/components/ui/command'
import { getCommands } from '@/lib/commands/commands'
import { filterCommands, groupBy } from '@/lib/commands/filter'
import { CommandShortcut } from './CommandShortcut'
import { useCommandPaletteActions } from './CommandPaletteProvider'
import type { Command as CommandType } from '@/lib/commands/types'

export const CommandPalette = () => {
  const [open, setOpen] = useState(false)
  const [search, setSearch] = useState('')
  const router = useRouter()
  const actions = useCommandPaletteActions()

  const commands = getCommands({
    router,
    actions,
  })

  const filteredCommands = filterCommands(commands, search)

  // Group by category
  const groupedCommands = groupBy(filteredCommands, 'category')

  // Handle command execution
  const executeCommand = useCallback((command: CommandType) => {
    if (command.enabled === false) return

    setOpen(false)
    setSearch('')

    // Execute after closing to prevent UI glitches
    setTimeout(() => {
      command.action()
    }, 100)
  }, [])

  // Reset search when dialog closes
  useEffect(() => {
    if (!open) {
      setSearch('')
    }
  }, [open])

  // Open command palette with Cmd+K
  useHotkey('k', () => {
    setOpen(prev => !prev)
  })

  return (
    <CommandDialog open={open} onOpenChange={setOpen}>
      <Command className="rounded-lg border shadow-md">
        <CommandInput
          placeholder="Search commands..."
          value={search}
          onValueChange={setSearch}
        />
        <CommandList>
          <CommandEmpty>No commands found.</CommandEmpty>

          {Object.entries(groupedCommands).map(([category, categoryCommands]) => (
            <CommandGroup key={category} heading={category}>
              {categoryCommands.map((command) => {
                const Icon = command.icon ? iconMap[command.icon] : null

                return (
                  <CommandItem
                    key={command.id}
                    value={command.title}
                    keywords={command.keywords}
                    onSelect={() => executeCommand(command)}
                    disabled={command.enabled === false}
                    className="flex items-center gap-2"
                  >
                    {Icon && <Icon className="h-4 w-4" />}
                    <span className="flex-1">{command.title}</span>
                    {command.subtitle && (
                      <span className="text-xs text-muted-foreground">
                        {command.subtitle}
                      </span>
                    )}
                    <CommandShortcut shortcut={command.shortcut} />
                  </CommandItem>
                )
              })}
            </CommandGroup>
          ))}
        </CommandList>
      </Command>
    </CommandDialog>
  )
}
