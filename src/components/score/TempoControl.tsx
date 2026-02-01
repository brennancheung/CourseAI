'use client'

import { useState, useEffect, useCallback } from 'react'

type TempoControlProps = {
  value: number
  onChange: (tempo: number) => void
  min?: number
  max?: number
  defaultValue?: number
}

export function TempoControl({
  value,
  onChange,
  min = 40,
  max = 260,
  defaultValue = 80,
}: TempoControlProps) {
  // Use string state to properly handle leading zeros
  const [inputValue, setInputValue] = useState(String(value))

  // Sync with external value changes
  useEffect(() => {
    setInputValue(String(value))
  }, [value])

  const normalize = useCallback(
    (input: string) => {
      const num = parseInt(input, 10)
      if (isNaN(num) || num === 0) return defaultValue
      return Math.max(min, Math.min(max, num))
    },
    [min, max, defaultValue]
  )

  const commitValue = useCallback(() => {
    const normalized = normalize(inputValue)
    setInputValue(String(normalized))
    onChange(normalized)
  }, [inputValue, normalize, onChange])

  return (
    <div className="flex items-center gap-2 text-sm">
      <label htmlFor="tempo" className="text-muted-foreground">
        Tempo:
      </label>
      <input
        id="tempo"
        type="text"
        inputMode="numeric"
        pattern="[0-9]*"
        value={inputValue}
        onChange={(e) => setInputValue(e.target.value)}
        onBlur={commitValue}
        onKeyDown={(e) => {
          if (e.key === 'Enter') {
            commitValue()
            e.currentTarget.blur()
          }
        }}
        className="w-16 px-2 py-1 rounded border bg-background text-foreground text-center"
      />
      <span className="text-muted-foreground">BPM</span>
    </div>
  )
}
