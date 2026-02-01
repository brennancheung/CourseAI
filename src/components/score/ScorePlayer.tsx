'use client'

import { useRef, useEffect, useState, useCallback } from 'react'
import abcjs from 'abcjs'
import type { PlaybackState } from './types'
import { PlaybackButton } from './PlaybackButton'
import { TempoControl } from './TempoControl'
import { useTempoOptional } from '@/components/exercises/TempoContext'

// CSS for score styling
// The key insight: SVG elements inherit `color` and use it via `currentColor`.
// By setting color: black on the container, all SVG elements render correctly.
const scoreStyles = `
  .score-container {
    color-scheme: light;
    color: black;
  }
  .abcjs-cursor {
    stroke: #e11d48 !important;
    stroke-width: 2;
    opacity: 0;
    transition: opacity 0.1s;
  }
  .abcjs-cursor.abcjs-cursor-visible {
    opacity: 1;
  }
  .abcjs-note_selected path,
  .abcjs-note_selected text {
    fill: #e11d48 !important;
  }
`

type ScorePlayerProps = {
  abc: string
  title?: string
  initialTempo?: number
  onPlaybackStateChange?: (state: PlaybackState) => void
  className?: string
}

/**
 * Full-featured score player with playback controls, cursor, and tempo adjustment.
 * Uses standard ABCJS synth for audio playback.
 *
 * NOTE: SpessaSynth integration code is preserved in src/lib/spessasynth/ for future use
 * if we want to switch back to orchestral soundfonts.
 */
export function ScorePlayer({
  abc,
  title,
  initialTempo = 120,
  onPlaybackStateChange,
  className = '',
}: ScorePlayerProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const synthRef = useRef<InstanceType<typeof abcjs.synth.CreateSynth> | null>(null)
  const timingCallbacksRef = useRef<abcjs.TimingCallbacks | null>(null)
  const visualObjRef = useRef<abcjs.TuneObject | null>(null)
  const cursorRef = useRef<SVGLineElement | null>(null)
  const isPlayingRef = useRef(false)

  // Optional tempo context - allows external control of tempo (e.g., clickable tempo links)
  const tempoContext = useTempoOptional()

  const [playbackState, setPlaybackState] = useState<PlaybackState>('stopped')
  const [localTempo, setLocalTempo] = useState(initialTempo)
  const [isLoaded, setIsLoaded] = useState(false)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Use context tempo if available, otherwise local tempo
  const tempo = tempoContext?.tempo ?? localTempo

  // Update parent when playback state changes
  useEffect(() => {
    onPlaybackStateChange?.(playbackState)
  }, [playbackState, onPlaybackStateChange])

  // Helper to clear visual state
  const clearVisualState = useCallback(() => {
    cursorRef.current?.classList.remove('abcjs-cursor-visible')
    containerRef.current?.querySelectorAll('.abcjs-note_selected').forEach((el) => {
      el.classList.remove('abcjs-note_selected')
    })
  }, [])

  // Create event callback for timing - extracted to avoid duplication
  const createEventCallback = useCallback(
    (container: HTMLElement) => {
      return (event: abcjs.NoteTimingEvent | null): abcjs.EventCallbackReturn => {
        if (!event) {
          // End of playback
          isPlayingRef.current = false
          setPlaybackState('stopped')
          clearVisualState()
          return undefined
        }

        // Clear previous highlights
        container.querySelectorAll('.abcjs-note_selected').forEach((el) => {
          el.classList.remove('abcjs-note_selected')
        })

        // Highlight current notes
        event.elements?.forEach((elemArray) => {
          elemArray.forEach((elem) => {
            elem?.classList.add('abcjs-note_selected')
          })
        })

        // Update cursor position
        if (cursorRef.current && event.left !== undefined) {
          cursorRef.current.setAttribute('x1', String(event.left))
          cursorRef.current.setAttribute('x2', String(event.left))
          cursorRef.current.classList.add('abcjs-cursor-visible')
        }

        return undefined
      }
    },
    [clearVisualState]
  )

  // Initialize on mount - only depends on abc string
  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    let mounted = true
    let synth: InstanceType<typeof abcjs.synth.CreateSynth> | null = null
    let timingCallbacks: abcjs.TimingCallbacks | null = null
    let cursor: SVGLineElement | null = null

    const init = async () => {
      try {
        setError(null)
        setIsLoaded(false)
        setIsLoading(true)

        // Render the ABC notation
        const visualObjs = abcjs.renderAbc(container, abc, {
          responsive: 'resize',
          add_classes: true,
          paddingtop: 10,
          paddingbottom: 10,
          paddingleft: 10,
          paddingright: 10,
        })

        if (!mounted) return

        if (!visualObjs || visualObjs.length < 1) {
          setError('Failed to render score')
          setIsLoading(false)
          return
        }

        visualObjRef.current = visualObjs[0]

        // Create ABCJS synth for audio playback
        synth = new abcjs.synth.CreateSynth()
        synthRef.current = synth

        await synth.init({
          visualObj: visualObjs[0],
          options: {
            qpm: initialTempo,
            chordsOff: true, // Disable auto-accompaniment
          },
        })

        // Prime the synth (load samples)
        await synth.prime()

        if (!mounted) return

        // Create timing callbacks for cursor animation
        timingCallbacks = new abcjs.TimingCallbacks(visualObjs[0], {
          qpm: initialTempo,
          eventCallback: createEventCallback(container),
        })
        timingCallbacksRef.current = timingCallbacks

        // Create cursor line element
        const svg = container.querySelector('svg')
        if (svg) {
          cursor = document.createElementNS('http://www.w3.org/2000/svg', 'line')
          cursor.setAttribute('class', 'abcjs-cursor')
          cursor.setAttribute('x1', '0')
          cursor.setAttribute('y1', '0')
          cursor.setAttribute('x2', '0')
          cursor.setAttribute('y2', String(svg.getBoundingClientRect().height))
          svg.appendChild(cursor)
          cursorRef.current = cursor
        }

        setIsLoaded(true)
        setIsLoading(false)
      } catch (err) {
        if (!mounted) return
        setError(`Failed to initialize: ${err instanceof Error ? err.message : 'Unknown error'}`)
        setIsLoading(false)
      }
    }

    init()

    return () => {
      mounted = false
      timingCallbacks?.stop()
      synth?.stop()
      cursor?.remove()
      synthRef.current = null
      timingCallbacksRef.current = null
      cursorRef.current = null
      isPlayingRef.current = false
    }
  }, [abc, initialTempo, createEventCallback])

  // Toggle play/stop
  const togglePlayback = useCallback(async () => {
    const synth = synthRef.current
    const timingCallbacks = timingCallbacksRef.current

    if (!synth || !timingCallbacks || !isLoaded) return

    if (isPlayingRef.current) {
      // Stop
      isPlayingRef.current = false
      synth.stop()
      timingCallbacks.stop()
      setPlaybackState('stopped')
      clearVisualState()
    } else {
      // Play
      try {
        isPlayingRef.current = true
        setPlaybackState('playing')

        // Start audio and cursor together
        synth.start()
        timingCallbacks.start()
      } catch (err) {
        setError(`Failed to start playback: ${err instanceof Error ? err.message : 'Unknown error'}`)
        isPlayingRef.current = false
        setPlaybackState('stopped')
      }
    }
  }, [isLoaded, clearVisualState])

  // Ref to track the last applied tempo to avoid duplicate re-inits
  const lastAppliedTempoRef = useRef(initialTempo)

  const applyTempoChange = useCallback(
    async (newTempo: number) => {
      // Avoid re-applying the same tempo
      if (newTempo === lastAppliedTempoRef.current) return
      lastAppliedTempoRef.current = newTempo

      const visualObj = visualObjRef.current
      const container = containerRef.current

      // Stop current playback if playing
      if (isPlayingRef.current) {
        isPlayingRef.current = false
        synthRef.current?.stop()
        timingCallbacksRef.current?.stop()
        setPlaybackState('stopped')
        clearVisualState()
      }

      // Recreate synth and timing callbacks with new tempo
      if (visualObj && container) {
        try {
          // Recreate synth with new tempo
          const synth = new abcjs.synth.CreateSynth()
          await synth.init({
            visualObj,
            options: {
              qpm: newTempo,
              chordsOff: true,
            },
          })
          await synth.prime()
          synthRef.current = synth

          // Recreate timing callbacks
          timingCallbacksRef.current?.stop()
          timingCallbacksRef.current = new abcjs.TimingCallbacks(visualObj, {
            qpm: newTempo,
            eventCallback: createEventCallback(container),
          })
        } catch (err) {
          setError(`Failed to change tempo: ${err instanceof Error ? err.message : 'Unknown error'}`)
        }
      }
    },
    [clearVisualState, createEventCallback]
  )

  // Handle tempo changes from the slider control
  const handleTempoChange = useCallback(
    (newTempo: number) => {
      setLocalTempo(newTempo)
      // Also update context if available
      tempoContext?.setTempo(newTempo)
      applyTempoChange(newTempo)
    },
    [applyTempoChange, tempoContext]
  )

  // Watch for external tempo changes from context (e.g., clicking tempo links)
  const contextTempo = tempoContext?.tempo
  useEffect(() => {
    if (contextTempo !== undefined && isLoaded) {
      applyTempoChange(contextTempo)
    }
  }, [contextTempo, isLoaded, applyTempoChange])

  return (
    <div className={`score-player ${className}`}>
      <style dangerouslySetInnerHTML={{ __html: scoreStyles }} />

      {title && <h3 className="text-lg font-semibold mb-3">{title}</h3>}

      {/* Score display - always white background for sheet music */}
      <div
        ref={containerRef}
        className="score-container rounded-lg p-4 mb-4 overflow-x-auto border"
        style={{ backgroundColor: 'white' }}
      />

      {/* Error message */}
      {error && (
        <div className="text-red-500 text-sm mb-4 p-2 bg-red-100 dark:bg-red-900/30 rounded border border-red-300 dark:border-red-800">
          {error}
        </div>
      )}

      {/* Controls */}
      <div className="controls flex items-center gap-4">
        <PlaybackButton
          state={playbackState}
          isLoading={isLoading}
          isLoaded={isLoaded}
          onClick={togglePlayback}
        />
        <TempoControl value={tempo} onChange={handleTempoChange} />
      </div>
    </div>
  )
}
