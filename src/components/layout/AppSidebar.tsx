'use client'

import Link from 'next/link'
import { BookOpen, ChevronRight } from 'lucide-react'
import {
  Sidebar,
  SidebarContent,
  SidebarHeader,
  SidebarFooter,
} from '@/components/ui/sidebar'
import { cn } from '@/lib/utils'
import { ThemeToggle } from '@/components/common/ThemeToggle'

type AppSidebarProps = {
  pathname: string
}

export const AppSidebar = ({ pathname }: AppSidebarProps) => {
  return (
    <Sidebar className="w-72 border-r border-border bg-card flex flex-col flex-shrink-0">
      <SidebarHeader className="h-14 border-b border-border">
        <div className="h-14 flex items-center justify-between px-4">
          <div className="flex items-center gap-2 font-semibold">
            <BookOpen className="w-5 h-5 text-primary" />
            <span>Lessons</span>
          </div>
          <ThemeToggle className="h-8 w-8" />
        </div>
      </SidebarHeader>

      <SidebarContent>
        <nav className="flex-1 p-3 overflow-y-auto">
          {/* Theory */}
          <div className="mb-6">
            <h3 className="text-sm font-semibold text-foreground uppercase tracking-wider px-2 mb-1">
              Theory
            </h3>
            <div className="space-y-0.5">
              <Link
                href="/app/theory"
                className={cn(
                  'flex items-center justify-between px-2 py-1 rounded-md text-sm transition-colors',
                  pathname === '/app/theory'
                    ? 'bg-primary/10 text-foreground'
                    : 'text-muted-foreground hover:bg-muted hover:text-foreground'
                )}
              >
                <span className="truncate">Theory Reference</span>
                <ChevronRight
                  className={cn(
                    'w-4 h-4 flex-shrink-0',
                    pathname === '/app/theory' ? 'opacity-100' : 'opacity-0'
                  )}
                />
              </Link>
            </div>
          </div>

          {/* Foundations */}
          <div className="mb-6">
            <h3 className="text-sm font-semibold text-foreground uppercase tracking-wider px-2 mb-1">
              Foundations
            </h3>
            <div className="space-y-0.5">
              <Link
                href="/app/lesson/instrument-character"
                className={cn(
                  'flex items-center justify-between px-2 py-1 rounded-md text-sm transition-colors',
                  pathname === '/app/lesson/instrument-character'
                    ? 'bg-primary/10 text-foreground'
                    : 'text-muted-foreground hover:bg-muted hover:text-foreground'
                )}
              >
                <span className="truncate">Instrument Character</span>
                <ChevronRight
                  className={cn(
                    'w-4 h-4 flex-shrink-0',
                    pathname === '/app/lesson/instrument-character' ? 'opacity-100' : 'opacity-0'
                  )}
                />
              </Link>
              <Link
                href="/app/lesson/cinematic-progressions"
                className={cn(
                  'flex items-center justify-between px-2 py-1 rounded-md text-sm transition-colors',
                  pathname === '/app/lesson/cinematic-progressions'
                    ? 'bg-primary/10 text-foreground'
                    : 'text-muted-foreground hover:bg-muted hover:text-foreground'
                )}
              >
                <span className="truncate">Cinematic Chord Progressions</span>
                <ChevronRight
                  className={cn(
                    'w-4 h-4 flex-shrink-0',
                    pathname === '/app/lesson/cinematic-progressions' ? 'opacity-100' : 'opacity-0'
                  )}
                />
              </Link>
              <Link
                href="/app/lesson/ostinato-patterns"
                className={cn(
                  'flex items-center justify-between px-2 py-1 rounded-md text-sm transition-colors',
                  pathname === '/app/lesson/ostinato-patterns'
                    ? 'bg-primary/10 text-foreground'
                    : 'text-muted-foreground hover:bg-muted hover:text-foreground'
                )}
              >
                <span className="truncate">Cinematic Ostinato Patterns</span>
                <ChevronRight
                  className={cn(
                    'w-4 h-4 flex-shrink-0',
                    pathname === '/app/lesson/ostinato-patterns' ? 'opacity-100' : 'opacity-0'
                  )}
                />
              </Link>
              <Link
                href="/app/lesson/string-sustains"
                className={cn(
                  'flex items-center justify-between px-2 py-1 rounded-md text-sm transition-colors',
                  pathname === '/app/lesson/string-sustains'
                    ? 'bg-primary/10 text-foreground'
                    : 'text-muted-foreground hover:bg-muted hover:text-foreground'
                )}
              >
                <span className="truncate">String Sustains with C#m Progressions</span>
                <ChevronRight
                  className={cn(
                    'w-4 h-4 flex-shrink-0',
                    pathname === '/app/lesson/string-sustains' ? 'opacity-100' : 'opacity-0'
                  )}
                />
              </Link>
            </div>
          </div>

          {/* Composer Studies */}
          <div className="mb-6">
            <h3 className="text-sm font-semibold text-foreground uppercase tracking-wider px-2 mb-1">
              Composer Studies
            </h3>
            <div className="space-y-0.5">
              <Link
                href="/app/lesson/zimmer-study"
                className={cn(
                  'flex items-center justify-between px-2 py-1 rounded-md text-sm transition-colors',
                  pathname === '/app/lesson/zimmer-study'
                    ? 'bg-primary/10 text-foreground'
                    : 'text-muted-foreground hover:bg-muted hover:text-foreground'
                )}
              >
                <span className="truncate">Composer Study: Hans Zimmer</span>
                <ChevronRight
                  className={cn(
                    'w-4 h-4 flex-shrink-0',
                    pathname === '/app/lesson/zimmer-study' ? 'opacity-100' : 'opacity-0'
                  )}
                />
              </Link>
              <Link
                href="/app/lesson/djawadi-study"
                className={cn(
                  'flex items-center justify-between px-2 py-1 rounded-md text-sm transition-colors',
                  pathname === '/app/lesson/djawadi-study'
                    ? 'bg-primary/10 text-foreground'
                    : 'text-muted-foreground hover:bg-muted hover:text-foreground'
                )}
              >
                <span className="truncate">Composer Study: Ramin Djawadi</span>
                <ChevronRight
                  className={cn(
                    'w-4 h-4 flex-shrink-0',
                    pathname === '/app/lesson/djawadi-study' ? 'opacity-100' : 'opacity-0'
                  )}
                />
              </Link>
            </div>
          </div>

          {/* Production */}
          <div className="mb-6">
            <h3 className="text-sm font-semibold text-foreground uppercase tracking-wider px-2 mb-1">
              Production
            </h3>
            <div className="space-y-0.5">
              <Link
                href="/app/lesson/trailer-sound"
                className={cn(
                  'flex items-center justify-between px-2 py-1 rounded-md text-sm transition-colors',
                  pathname === '/app/lesson/trailer-sound'
                    ? 'bg-primary/10 text-foreground'
                    : 'text-muted-foreground hover:bg-muted hover:text-foreground'
                )}
              >
                <span className="truncate">The Epic Trailer Sound</span>
                <ChevronRight
                  className={cn(
                    'w-4 h-4 flex-shrink-0',
                    pathname === '/app/lesson/trailer-sound' ? 'opacity-100' : 'opacity-0'
                  )}
                />
              </Link>
            </div>
          </div>

          {/* Hybrid Elements */}
          <div className="mb-6">
            <h3 className="text-sm font-semibold text-foreground uppercase tracking-wider px-2 mb-1">
              Hybrid Elements
            </h3>
            <div className="space-y-0.5">
              <Link
                href="/app/lesson/brass-stabs"
                className={cn(
                  'flex items-center justify-between px-2 py-1 rounded-md text-sm transition-colors',
                  pathname === '/app/lesson/brass-stabs'
                    ? 'bg-primary/10 text-foreground'
                    : 'text-muted-foreground hover:bg-muted hover:text-foreground'
                )}
              >
                <span className="truncate">Basic Brass Stabs</span>
                <ChevronRight
                  className={cn(
                    'w-4 h-4 flex-shrink-0',
                    pathname === '/app/lesson/brass-stabs' ? 'opacity-100' : 'opacity-0'
                  )}
                />
              </Link>
              <Link
                href="/app/lesson/synth-layers"
                className={cn(
                  'flex items-center justify-between px-2 py-1 rounded-md text-sm transition-colors',
                  pathname === '/app/lesson/synth-layers'
                    ? 'bg-primary/10 text-foreground'
                    : 'text-muted-foreground hover:bg-muted hover:text-foreground'
                )}
              >
                <span className="truncate">Layering Synth Pads</span>
                <ChevronRight
                  className={cn(
                    'w-4 h-4 flex-shrink-0',
                    pathname === '/app/lesson/synth-layers' ? 'opacity-100' : 'opacity-0'
                  )}
                />
              </Link>
            </div>
          </div>

          {/* Rhythm & Percussion */}
          <div className="mb-6">
            <h3 className="text-sm font-semibold text-foreground uppercase tracking-wider px-2 mb-1">
              Rhythm &amp; Percussion
            </h3>
            <div className="space-y-0.5">
              <Link
                href="/app/lesson/epic-percussion"
                className={cn(
                  'flex items-center justify-between px-2 py-1 rounded-md text-sm transition-colors',
                  pathname === '/app/lesson/epic-percussion'
                    ? 'bg-primary/10 text-foreground'
                    : 'text-muted-foreground hover:bg-muted hover:text-foreground'
                )}
              >
                <span className="truncate">Epic Percussion Pattern</span>
                <ChevronRight
                  className={cn(
                    'w-4 h-4 flex-shrink-0',
                    pathname === '/app/lesson/epic-percussion' ? 'opacity-100' : 'opacity-0'
                  )}
                />
              </Link>
            </div>
          </div>

          {/* Devastator Breakout Pro */}
          <div className="mb-6">
            <h3 className="text-sm font-semibold text-foreground uppercase tracking-wider px-2 mb-1">
              Devastator Breakout Pro
            </h3>
            <div className="space-y-0.5">
              <Link
                href="/app/lesson/devastator-orientation"
                className={cn(
                  'flex items-center justify-between px-2 py-1 rounded-md text-sm transition-colors',
                  pathname === '/app/lesson/devastator-orientation'
                    ? 'bg-primary/10 text-foreground'
                    : 'text-muted-foreground hover:bg-muted hover:text-foreground'
                )}
              >
                <span className="truncate">Devastator Orientation</span>
                <ChevronRight
                  className={cn(
                    'w-4 h-4 flex-shrink-0',
                    pathname === '/app/lesson/devastator-orientation' ? 'opacity-100' : 'opacity-0'
                  )}
                />
              </Link>
              <Link
                href="/app/lesson/anatomy-of-braam"
                className={cn(
                  'flex items-center justify-between px-2 py-1 rounded-md text-sm transition-colors',
                  pathname === '/app/lesson/anatomy-of-braam'
                    ? 'bg-primary/10 text-foreground'
                    : 'text-muted-foreground hover:bg-muted hover:text-foreground'
                )}
              >
                <span className="truncate">Anatomy of a Braam</span>
                <ChevronRight
                  className={cn(
                    'w-4 h-4 flex-shrink-0',
                    pathname === '/app/lesson/anatomy-of-braam' ? 'opacity-100' : 'opacity-0'
                  )}
                />
              </Link>
              <Link
                href="/app/lesson/pulse-foundations"
                className={cn(
                  'flex items-center justify-between px-2 py-1 rounded-md text-sm transition-colors',
                  pathname === '/app/lesson/pulse-foundations'
                    ? 'bg-primary/10 text-foreground'
                    : 'text-muted-foreground hover:bg-muted hover:text-foreground'
                )}
              >
                <span className="truncate">Pulse Foundations</span>
                <ChevronRight
                  className={cn(
                    'w-4 h-4 flex-shrink-0',
                    pathname === '/app/lesson/pulse-foundations' ? 'opacity-100' : 'opacity-0'
                  )}
                />
              </Link>
              <Link
                href="/app/lesson/tick-tock-tension"
                className={cn(
                  'flex items-center justify-between px-2 py-1 rounded-md text-sm transition-colors',
                  pathname === '/app/lesson/tick-tock-tension'
                    ? 'bg-primary/10 text-foreground'
                    : 'text-muted-foreground hover:bg-muted hover:text-foreground'
                )}
              >
                <span className="truncate">Tick-Tock Tension</span>
                <ChevronRight
                  className={cn(
                    'w-4 h-4 flex-shrink-0',
                    pathname === '/app/lesson/tick-tock-tension' ? 'opacity-100' : 'opacity-0'
                  )}
                />
              </Link>
              <Link
                href="/app/lesson/hit-stack"
                className={cn(
                  'flex items-center justify-between px-2 py-1 rounded-md text-sm transition-colors',
                  pathname === '/app/lesson/hit-stack'
                    ? 'bg-primary/10 text-foreground'
                    : 'text-muted-foreground hover:bg-muted hover:text-foreground'
                )}
              >
                <span className="truncate">The Hit Stack</span>
                <ChevronRight
                  className={cn(
                    'w-4 h-4 flex-shrink-0',
                    pathname === '/app/lesson/hit-stack' ? 'opacity-100' : 'opacity-0'
                  )}
                />
              </Link>
              <Link
                href="/app/lesson/custom-risers"
                className={cn(
                  'flex items-center justify-between px-2 py-1 rounded-md text-sm transition-colors',
                  pathname === '/app/lesson/custom-risers'
                    ? 'bg-primary/10 text-foreground'
                    : 'text-muted-foreground hover:bg-muted hover:text-foreground'
                )}
              >
                <span className="truncate">Custom Risers (XY Engine)</span>
                <ChevronRight
                  className={cn(
                    'w-4 h-4 flex-shrink-0',
                    pathname === '/app/lesson/custom-risers' ? 'opacity-100' : 'opacity-0'
                  )}
                />
              </Link>
              <Link
                href="/app/lesson/sequencer-to-midi"
                className={cn(
                  'flex items-center justify-between px-2 py-1 rounded-md text-sm transition-colors',
                  pathname === '/app/lesson/sequencer-to-midi'
                    ? 'bg-primary/10 text-foreground'
                    : 'text-muted-foreground hover:bg-muted hover:text-foreground'
                )}
              >
                <span className="truncate">Sequencer → MIDI Export</span>
                <ChevronRight
                  className={cn(
                    'w-4 h-4 flex-shrink-0',
                    pathname === '/app/lesson/sequencer-to-midi' ? 'opacity-100' : 'opacity-0'
                  )}
                />
              </Link>
              <Link
                href="/app/lesson/hybrid-hits"
                className={cn(
                  'flex items-center justify-between px-2 py-1 rounded-md text-sm transition-colors',
                  pathname === '/app/lesson/hybrid-hits'
                    ? 'bg-primary/10 text-foreground'
                    : 'text-muted-foreground hover:bg-muted hover:text-foreground'
                )}
              >
                <span className="truncate">Hybrid Hits</span>
                <ChevronRight
                  className={cn(
                    'w-4 h-4 flex-shrink-0',
                    pathname === '/app/lesson/hybrid-hits' ? 'opacity-100' : 'opacity-0'
                  )}
                />
              </Link>
              <Link
                href="/app/lesson/building-a-drop"
                className={cn(
                  'flex items-center justify-between px-2 py-1 rounded-md text-sm transition-colors',
                  pathname === '/app/lesson/building-a-drop'
                    ? 'bg-primary/10 text-foreground'
                    : 'text-muted-foreground hover:bg-muted hover:text-foreground'
                )}
              >
                <span className="truncate">Building a Drop (Capstone)</span>
                <ChevronRight
                  className={cn(
                    'w-4 h-4 flex-shrink-0',
                    pathname === '/app/lesson/building-a-drop' ? 'opacity-100' : 'opacity-0'
                  )}
                />
              </Link>
            </div>
          </div>
        </nav>
      </SidebarContent>

      <SidebarFooter className="border-t border-border p-4">
        <div className="text-xs text-muted-foreground">19 lessons available</div>
      </SidebarFooter>
    </Sidebar>
  )
}
