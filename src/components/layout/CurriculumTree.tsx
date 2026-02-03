'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { ChevronRight, ChevronDown } from 'lucide-react'
import { CurriculumNode, isLesson, getPathToNode } from '@/data/curriculum'
import { cn } from '@/lib/utils'

type CurriculumTreeProps = {
  nodes: CurriculumNode[]
  currentSlug?: string
}

type TreeNodeProps = {
  node: CurriculumNode
  depth: number
  currentSlug?: string
  expandedSlugs: Set<string>
  onToggle: (slug: string) => void
}

/**
 * Single node in the curriculum tree
 * Renders differently based on depth and whether it's a leaf
 */
function TreeNode({ node, depth, currentSlug, expandedSlugs, onToggle }: TreeNodeProps) {
  const isCurrentLesson = node.slug === currentSlug
  const isExpanded = expandedSlugs.has(node.slug)

  // Style based on depth
  const depthStyles = getDepthStyles(depth)

  // Leaf node = lesson link
  if (isLesson(node)) {
    return (
      <Link
        href={`/app/lesson/${node.slug}`}
        className={cn(
          'flex items-center gap-2 py-1.5 px-2 rounded-md text-sm transition-colors',
          depthStyles.text,
          isCurrentLesson
            ? 'bg-primary/10 text-foreground font-medium'
            : 'text-muted-foreground hover:bg-muted hover:text-foreground'
        )}
        style={{ paddingLeft: `${depth * 12 + 8}px` }}
      >
        <span className={cn(
          'w-1.5 h-1.5 rounded-full flex-shrink-0',
          isCurrentLesson ? 'bg-primary' : 'bg-muted-foreground/30'
        )} />
        <span className="truncate">{node.title}</span>
      </Link>
    )
  }

  // Group node = collapsible header
  return (
    <div>
      <button
        onClick={() => onToggle(node.slug)}
        className={cn(
          'w-full flex items-center gap-2 py-1.5 px-2 rounded-md transition-colors',
          depthStyles.header,
          'hover:bg-muted'
        )}
        style={{ paddingLeft: `${depth * 12 + 8}px` }}
      >
        {isExpanded ? (
          <ChevronDown className="w-4 h-4 flex-shrink-0 text-muted-foreground" />
        ) : (
          <ChevronRight className="w-4 h-4 flex-shrink-0 text-muted-foreground" />
        )}
        <span className={cn('truncate', depthStyles.text)}>{node.title}</span>
      </button>

      {/* Children */}
      {isExpanded && node.children && (
        <div className="mt-0.5">
          {node.children.map((child) => (
            <TreeNode
              key={child.slug}
              node={child}
              depth={depth + 1}
              currentSlug={currentSlug}
              expandedSlugs={expandedSlugs}
              onToggle={onToggle}
            />
          ))}
        </div>
      )}
    </div>
  )
}

/**
 * Get styles based on depth level
 */
function getDepthStyles(depth: number): { header: string; text: string } {
  // Depth 0: Top-level (bold, larger)
  if (depth === 0) {
    return {
      header: 'font-semibold text-sm',
      text: 'font-semibold',
    }
  }
  // Depth 1: Major section
  if (depth === 1) {
    return {
      header: 'font-medium text-sm',
      text: 'font-medium',
    }
  }
  // Depth 2+: Subsections and lessons
  return {
    header: 'text-sm',
    text: '',
  }
}

/**
 * Main curriculum tree component
 */
export function CurriculumTree({ nodes, currentSlug }: CurriculumTreeProps) {
  // Track which nodes are expanded
  const [expandedSlugs, setExpandedSlugs] = useState<Set<string>>(new Set())

  // Auto-expand path to current lesson
  useEffect(() => {
    if (!currentSlug) return

    const path = getPathToNode(nodes, currentSlug)
    if (path) {
      setExpandedSlugs((prev) => {
        const next = new Set(prev)
        // Add all ancestors (except the lesson itself)
        path.slice(0, -1).forEach((slug) => next.add(slug))
        return next
      })
    }
  }, [currentSlug, nodes])

  const handleToggle = (slug: string) => {
    setExpandedSlugs((prev) => {
      const next = new Set(prev)
      if (next.has(slug)) {
        next.delete(slug)
      } else {
        next.add(slug)
      }
      return next
    })
  }

  return (
    <div className="space-y-1">
      {nodes.map((node) => (
        <TreeNode
          key={node.slug}
          node={node}
          depth={0}
          currentSlug={currentSlug}
          expandedSlugs={expandedSlugs}
          onToggle={handleToggle}
        />
      ))}
    </div>
  )
}
