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

type LessonLink = {
  href: string
  label: string
}

type LessonSection = {
  title: string
  lessons: LessonLink[]
}

// Define the curriculum structure
const curriculum: LessonSection[] = [
  {
    title: 'Module 1.1: The Learning Problem',
    lessons: [
      { href: '/app/lesson/what-is-learning', label: 'What is Learning?' },
      { href: '/app/lesson/linear-regression', label: 'Linear Regression' },
      { href: '/app/lesson/loss-functions', label: 'Loss Functions' },
      { href: '/app/lesson/gradient-descent', label: 'Gradient Descent' },
      { href: '/app/lesson/learning-rate', label: 'Learning Rate Deep Dive' },
      { href: '/app/lesson/implementing-linear-regression', label: 'Implementation (Colab)' },
    ],
  },
  // Future modules - add as they're built
  // {
  //   title: 'Module 1.2: From Linear to Neural',
  //   lessons: [
  //     { href: '/app/lesson/limits-of-linearity', label: 'Limits of Linearity' },
  //     ...
  //   ],
  // },
]

function LessonItem({ lesson, pathname }: { lesson: LessonLink; pathname: string }) {
  const isActive = pathname === lesson.href

  return (
    <Link
      href={lesson.href}
      className={cn(
        'flex items-center justify-between px-2 py-1.5 rounded-md text-sm transition-colors',
        isActive
          ? 'bg-primary/10 text-foreground'
          : 'text-muted-foreground hover:bg-muted hover:text-foreground'
      )}
    >
      <span className="truncate">{lesson.label}</span>
      <ChevronRight
        className={cn(
          'w-4 h-4 flex-shrink-0',
          isActive ? 'opacity-100' : 'opacity-0'
        )}
      />
    </Link>
  )
}

function SectionGroup({ section, pathname }: { section: LessonSection; pathname: string }) {
  return (
    <div className="mb-6">
      <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider px-2 mb-2">
        {section.title}
      </h3>
      <div className="space-y-0.5">
        {section.lessons.map((lesson) => (
          <LessonItem key={lesson.href} lesson={lesson} pathname={pathname} />
        ))}
      </div>
    </div>
  )
}

export const AppSidebar = ({ pathname }: AppSidebarProps) => {
  const totalLessons = curriculum.reduce((sum, section) => sum + section.lessons.length, 0)

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
          {curriculum.map((section) => (
            <SectionGroup
              key={section.title}
              section={section}
              pathname={pathname}
            />
          ))}
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
