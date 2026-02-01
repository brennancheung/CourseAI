/**
 * SpessaSynth wrapper for audio playback using SF2 soundfonts.
 *
 * This module provides a singleton synthesizer instance that loads a General MIDI
 * SF2 soundfont for instrument playback.
 *
 * Usage:
 * 1. Call initSpessaSynth() on user gesture (required by browser)
 * 2. Use getSynth() to get the synthesizer instance
 * 3. Call synth.noteOn/noteOff/programChange to play notes
 */
import type { WorkletSynthesizer } from 'spessasynth_lib'

let synthInstance: WorkletSynthesizer | null = null
let audioContext: AudioContext | null = null
let isInitialized = false
let initPromise: Promise<WorkletSynthesizer> | null = null

export type SpessaSynthLoadingState = 'idle' | 'loading-worklet' | 'loading-soundfont' | 'ready' | 'error'

let loadingState: SpessaSynthLoadingState = 'idle'
let loadingStateListeners: Array<(state: SpessaSynthLoadingState) => void> = []

function setLoadingState(state: SpessaSynthLoadingState) {
  loadingState = state
  loadingStateListeners.forEach((listener) => listener(state))
}

export function onLoadingStateChange(listener: (state: SpessaSynthLoadingState) => void): () => void {
  loadingStateListeners.push(listener)
  // Immediately call with current state
  listener(loadingState)
  // Return unsubscribe function
  return () => {
    loadingStateListeners = loadingStateListeners.filter((l) => l !== listener)
  }
}

export function getLoadingState(): SpessaSynthLoadingState {
  return loadingState
}

/**
 * Initialize SpessaSynth with a General MIDI soundfont.
 * Must be called in response to a user gesture (click, tap, etc.).
 *
 * Returns the synthesizer instance, caching it for subsequent calls.
 */
export async function initSpessaSynth(): Promise<WorkletSynthesizer> {
  // Return existing instance if already initialized
  if (synthInstance && isInitialized) {
    return synthInstance
  }

  // Return existing promise if initialization is in progress
  if (initPromise) {
    return initPromise
  }

  // Start initialization
  initPromise = doInit()
  return initPromise
}

async function doInit(): Promise<WorkletSynthesizer> {
  try {
    setLoadingState('loading-worklet')

    // Create audio context
    audioContext = new AudioContext()

    // Add the AudioWorklet module
    await audioContext.audioWorklet.addModule('/spessasynth_processor.min.js')

    // Dynamically import spessasynth_lib (must be dynamic for client-side only)
    const { WorkletSynthesizer } = await import('spessasynth_lib')

    // Create the synthesizer
    synthInstance = new WorkletSynthesizer(audioContext)

    setLoadingState('loading-soundfont')

    // Fetch and load the soundfont (Sonatina Symphonic Orchestra)
    const sf2Response = await fetch('/soundfonts/sonatina-converted.sf2')
    if (!sf2Response.ok) {
      throw new Error(`Failed to fetch soundfont: ${sf2Response.status}`)
    }
    const sf2Data = await sf2Response.arrayBuffer()

    // Add the soundfont to the synthesizer
    await synthInstance.soundBankManager.addSoundBank(sf2Data, 'gm')

    // Wait for synthesizer to be ready
    await synthInstance.isReady

    // Connect to audio destination AFTER synth is ready
    synthInstance.connect(audioContext.destination)

    isInitialized = true
    setLoadingState('ready')

    return synthInstance
  } catch (error) {
    setLoadingState('error')
    initPromise = null
    throw error
  }
}

/**
 * Get the synthesizer instance, or null if not yet initialized.
 */
export function getSynth(): WorkletSynthesizer | null {
  return synthInstance
}

/**
 * Get the audio context, or null if not yet initialized.
 */
export function getAudioContext(): AudioContext | null {
  return audioContext
}

/**
 * Check if the synthesizer is fully initialized and ready.
 */
export function isSpessaSynthReady(): boolean {
  return isInitialized
}

/**
 * Resume the audio context (needed after browser auto-suspends it).
 */
export async function resumeAudioContext(): Promise<void> {
  if (audioContext && audioContext.state === 'suspended') {
    await audioContext.resume()
  }
}
