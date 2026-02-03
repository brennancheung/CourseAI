'use client'

import { BookOpen } from 'lucide-react'
import {
  Sidebar,
  SidebarContent,
  SidebarHeader,
  SidebarFooter,
} from '@/components/ui/sidebar'
import { ThemeToggle } from '@/components/common/ThemeToggle'
import { CurriculumTree } from './CurriculumTree'
import { curriculum, getAllLessons } from '@/data/curriculum'

type AppSidebarProps = {
  pathname: string
}

/**
 * Extract lesson slug from pathname
 * /app/lesson/gradient-descent → gradient-descent
 */
function getLessonSlug(pathname: string): string | undefined {
  const match = pathname.match(/\/app\/lesson\/([^/]+)/)
  return match ? match[1] : undefined
}

export function AppSidebar({ pathname }: AppSidebarProps) {
  const currentSlug = getLessonSlug(pathname)
  const totalLessons = getAllLessons(curriculum).length

  return (
    <Sidebar className="w-72 border-r border-border bg-card flex flex-col flex-shrink-0">
      <SidebarHeader className="h-14 border-b border-border">
        <div className="h-14 flex items-center justify-between px-4">
          <div className="flex items-center gap-2 font-semibold">
            <BookOpen className="w-5 h-5 text-primary" />
            <span>CourseAI</span>
          </div>
          <ThemeToggle className="h-8 w-8" />
        </div>
      </SidebarHeader>

      <SidebarContent>
        <nav className="flex-1 p-3 overflow-y-auto">
          <CurriculumTree nodes={curriculum} currentSlug={currentSlug} />
        </nav>
      </SidebarContent>

      <SidebarFooter className="border-t border-border p-4">
        <div className="text-xs text-muted-foreground">
          {totalLessons} lessons available
        </div>
      </SidebarFooter>
    </Sidebar>
  )
}
