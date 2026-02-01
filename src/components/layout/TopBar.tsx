import { SidebarTrigger } from '@/components/ui/sidebar'

export const TopBar = () => {
  return (
    <header className="h-14 px-8 flex items-center gap-4">
      <SidebarTrigger />
    </header>
  )
}
