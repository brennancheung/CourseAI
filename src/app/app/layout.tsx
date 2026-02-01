'use client'

import { usePathname } from 'next/navigation'
import { ConvexProvider } from '@/providers/ConvexProvider'
import { ThemeProvider } from '@/components/common/ThemeProvider'
import { CommandPaletteProvider } from '@/components/command-palette/CommandPaletteProvider'
import { AppSidebar } from '@/components/layout/AppSidebar'
import { TopBar } from '@/components/layout/TopBar'
import { SidebarProvider, SidebarInset } from '@/components/ui/sidebar'

export default function AppLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname()

  return (
    <ConvexProvider>
      <ThemeProvider
        attribute="class"
        defaultTheme="system"
        enableSystem
        disableTransitionOnChange
      >
        <CommandPaletteProvider>
          <SidebarProvider>
            <div className="flex h-screen w-full">
              <AppSidebar pathname={pathname} />
              <SidebarInset className="flex-1 flex flex-col overflow-hidden">
                <TopBar />
                <main className="flex-1 overflow-auto p-8">
                  <div className="max-w-5xl mx-auto">{children}</div>
                </main>
              </SidebarInset>
            </div>
          </SidebarProvider>
        </CommandPaletteProvider>
      </ThemeProvider>
    </ConvexProvider>
  )
}
