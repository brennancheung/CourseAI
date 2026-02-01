import Link from 'next/link'
import type { ReactNode } from 'react'

export type NavItem = {
  name: string
  href: string
  badge?: string
  action?: ReactNode
}

type NavigationSectionProps = {
  title?: string
  items: NavItem[]
  pathname: string
}

const isPathActive = (pathname: string, href: string) => {
  if (pathname === href) return true
  if (href !== '/app' && pathname.startsWith(href + '/')) return true
  return false
}

export const NavigationSection = ({ title, items, pathname }: NavigationSectionProps) => {
  return (
    <div>
      {title && <p className="px-3 text-xs font-medium text-muted-foreground mb-2">{title}</p>}
      <div className="space-y-1">
        {items.map(item => {
          const active = isPathActive(pathname, item.href)
          return (
            <div
              key={item.name}
              className={`
                flex items-center justify-between px-3 py-2 rounded-md text-sm font-medium transition-colors
                ${active
                  ? 'bg-secondary text-foreground'
                  : 'text-muted-foreground hover:text-foreground hover:bg-secondary'
                }
              `}
            >
              <Link
                href={item.href}
                aria-current={active ? 'page' : undefined}
                className="flex-1 flex items-center justify-between"
              >
                <span>{item.name}</span>
                {item.badge ? (
                  <span className="ml-3 text-[10px] px-1.5 py-0.5 rounded bg-accent text-accent-foreground">{item.badge}</span>
                ) : null}
              </Link>
              {item.action}
            </div>
          )
        })}
      </div>
    </div>
  )
}
