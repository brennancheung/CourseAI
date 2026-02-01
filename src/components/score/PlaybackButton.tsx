'use client'

import type { PlaybackState } from './types'

type PlaybackButtonProps = {
  state: PlaybackState
  isLoading: boolean
  isLoaded: boolean
  onClick: () => void
}

export function PlaybackButton({
  state,
  isLoading,
  isLoaded,
  onClick,
}: PlaybackButtonProps) {
  const isPlaying = state === 'playing'
  const disabled = !isLoaded || isLoading

  const getTitle = () => {
    if (isLoading) return 'Loading audio...'
    if (!isLoaded) return 'Audio failed to load'
    return undefined
  }

  return (
    <button
      onClick={onClick}
      disabled={disabled}
      aria-label={isPlaying ? 'Stop' : 'Play'}
      title={getTitle()}
      className="flex items-center justify-center size-12 rounded-full bg-neutral-100 text-neutral-900 hover:bg-neutral-200 dark:bg-neutral-800 dark:text-white dark:hover:bg-neutral-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
    >
      {isLoading ? <LoadingIcon /> : isPlaying ? <StopIcon /> : <PlayIcon />}
    </button>
  )
}

function PlayIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 16 16" fill="currentColor">
      <path d="M4 2.5v11l9-5.5-9-5.5z" />
    </svg>
  )
}

function StopIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 16 16" fill="currentColor">
      <rect x="3" y="3" width="10" height="10" />
    </svg>
  )
}

function LoadingIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 16 16" fill="currentColor" className="animate-spin">
      <path d="M8 1a7 7 0 1 0 7 7h-2a5 5 0 1 1-5-5V1z" opacity="0.25" />
      <path d="M8 1v2a5 5 0 0 1 5 5h2a7 7 0 0 0-7-7z" />
    </svg>
  )
}
