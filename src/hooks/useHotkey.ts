import { useEffect } from 'react'
import { getModifierKey } from '@/lib/platform'

export const useHotkey = (key: string, callback: () => void) => {
  useEffect(() => {
    const modifierKey = getModifierKey()

    const handleKeyDown = (event: KeyboardEvent) => {
      const isModifierPressed = modifierKey === 'cmd' ? event.metaKey : event.ctrlKey

      if (isModifierPressed && event.key.toLowerCase() === key.toLowerCase()) {
        event.preventDefault()
        callback()
      }
    }

    window.addEventListener('keydown', handleKeyDown)

    return () => {
      window.removeEventListener('keydown', handleKeyDown)
    }
  }, [key, callback])
}
