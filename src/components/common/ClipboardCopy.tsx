'use client'

import { useState } from 'react'
import { Copy, Check } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { useTimeout } from '@/hooks/useTimeout'

type ClipboardCopyProps = {
  text: string
  variant?: 'default' | 'ghost' | 'outline'
  size?: 'default' | 'sm' | 'lg' | 'icon'
  label?: string
  className?: string
}

export const ClipboardCopy = ({ text, variant = 'ghost', size = 'icon', label, className }: ClipboardCopyProps) => {
  const [copied, setCopied] = useState(false)

  useTimeout(() => setCopied(false), copied ? 2000 : null)

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
    } catch (error) {
      console.error('Failed to copy:', error)
    }
  }

  return (
    <Button
      variant={variant}
      size={size}
      onClick={handleCopy}
      className={className}
      title={label ?? 'Copy to clipboard'}
    >
      {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
      {label && <span className="ml-2">{copied ? 'Copied!' : label}</span>}
    </Button>
  )
}
