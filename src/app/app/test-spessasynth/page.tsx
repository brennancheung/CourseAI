'use client'

import { useState, useRef, useEffect } from 'react'
import { Copy, Check } from 'lucide-react'

/**
 * Minimal SpessaSynth test page - no ABCJS, no ScorePlayer, just raw SpessaSynth.
 * Following the official example exactly.
 */
export default function TestSpessaSynthPage() {
  const [status, setStatus] = useState('Click button to start')
  const [isLoading, setIsLoading] = useState(false)
  const [logs, setLogs] = useState<string[]>([])
  const [copied, setCopied] = useState(false)
  const synthRef = useRef<unknown>(null)
  const ctxRef = useRef<AudioContext | null>(null)

  const log = (message: string) => {
    const timestamp = new Date().toISOString().split('T')[1].slice(0, 12)
    const entry = `[${timestamp}] ${message}`
    console.log(entry)
    setLogs(prev => [...prev, entry])
  }

  // Reset copied state after 2 seconds
  useEffect(() => {
    if (!copied) return
    const timer = setTimeout(() => setCopied(false), 2000)
    return () => clearTimeout(timer)
  }, [copied])

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(logs.join('\n'))
      setCopied(true)
    } catch (error) {
      console.error('Failed to copy:', error)
    }
  }

  const handleClick = async () => {
    try {
      // If already initialized, just play a note
      if (synthRef.current && ctxRef.current) {
        setStatus('Playing note...')
        log('Re-playing note on existing synth')
        await ctxRef.current.resume()
        const synth = synthRef.current as {
          noteOn: (ch: number, note: number, vel: number) => void
          noteOff: (ch: number, note: number) => void
          programChange: (ch: number, prog: number) => void
        }
        synth.programChange(0, 16)
        synth.noteOn(0, 60, 127)
        log('noteOn(0, 60, 127) sent')
        setTimeout(() => {
          synth.noteOff(0, 60)
          log('noteOff(0, 60) sent')
          setStatus('Note finished. Click again to play another.')
        }, 1000)
        return
      }

      // First click - initialize everything
      setIsLoading(true)
      setLogs([])

      log('Starting initialization...')
      setStatus('Creating AudioContext...')

      const ctx = new AudioContext()
      ctxRef.current = ctx
      log(`AudioContext created, state: ${ctx.state}, sampleRate: ${ctx.sampleRate}`)

      setStatus('Loading AudioWorklet processor...')
      log('Loading /spessasynth_processor.min.js...')
      await ctx.audioWorklet.addModule('/spessasynth_processor.min.js')
      log('AudioWorklet module loaded successfully')

      setStatus('Creating synthesizer...')
      log('Importing spessasynth_lib...')
      const { WorkletSynthesizer } = await import('spessasynth_lib')
      log('Creating WorkletSynthesizer...')
      const synth = new WorkletSynthesizer(ctx)
      log(`Synth created: ${synth.constructor.name}`)

      setStatus('Fetching soundfont (248MB, please wait)...')
      log('Fetching /soundfonts/sonatina-converted.sf2...')
      const sf2Response = await fetch('/soundfonts/sonatina-converted.sf2')
      if (!sf2Response.ok) {
        throw new Error(`Failed to fetch soundfont: ${sf2Response.status}`)
      }
      const sf2Data = await sf2Response.arrayBuffer()
      log(`Soundfont fetched, size: ${sf2Data.byteLength} bytes`)

      setStatus('Adding soundfont to synth...')
      log('Calling soundBankManager.addSoundBank...')
      await synth.soundBankManager.addSoundBank(sf2Data, 'sonatina')
      log(`Soundfont added, banks: ${JSON.stringify(synth.soundBankManager.soundBankList.map((b: { id: string }) => b.id))}`)

      setStatus('Waiting for synth to be ready...')
      log('Awaiting synth.isReady...')
      await synth.isReady
      log('Synth is ready!')
      log(`Preset count: ${synth.presetList.length}`)
      log('--- ALL PRESETS ---')
      synth.presetList.forEach((p: { bank?: number; program: number; name: string }) => {
        log(`  Bank ${p.bank ?? '?'}, Program ${p.program}: ${p.name}`)
      })
      log('--- END PRESETS ---')

      setStatus('Connecting to audio output...')
      log('Calling synth.connect(ctx.destination)...')
      const connectedNode = synth.connect(ctx.destination)
      log(`Connected! Returned: ${connectedNode.constructor.name}`)

      synthRef.current = synth

      setStatus('Resuming AudioContext...')
      log('Calling ctx.resume()...')
      await ctx.resume()
      log(`AudioContext resumed, state: ${ctx.state}`)

      setStatus('Playing test note (C4 on first instrument)...')
      log('Calling programChange(0, 0) for first instrument...')
      synth.programChange(0, 0)

      log('Calling noteOn(0, 60, 127)...')
      synth.noteOn(0, 60, 127)
      log('noteOn sent! You should hear a note NOW.')

      setTimeout(() => {
        synth.noteOff(0, 60)
        log('noteOff(0, 60) sent')
        setStatus('Test complete. Did you hear a note? Try the program buttons below.')
        setIsLoading(false)
      }, 1000)

    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      log(`ERROR: ${message}`)
      console.error('Error:', error)
      setStatus(`Error: ${message}`)
      setIsLoading(false)
    }
  }

  return (
    <div className="p-8 max-w-2xl mx-auto">
      <h1 className="text-2xl font-bold mb-4">SpessaSynth Test Page</h1>
      <p className="text-muted-foreground mb-6">
        Minimal test to isolate SpessaSynth from ABCJS and other components.
      </p>

      <div className="flex flex-wrap gap-4">
        <button
          onClick={handleClick}
          disabled={isLoading}
          className="px-6 py-3 bg-primary text-primary-foreground rounded-lg text-lg disabled:opacity-50"
        >
          {isLoading ? 'Loading...' : 'Load Sonatina & Play'}
        </button>

        <button
          onClick={() => {
            log('Testing basic Web Audio oscillator...')
            const ctx = new AudioContext()
            const osc = ctx.createOscillator()
            const gain = ctx.createGain()
            osc.connect(gain)
            gain.connect(ctx.destination)
            gain.gain.value = 0.3
            osc.frequency.value = 440
            osc.start()
            log('Oscillator started at 440Hz - you should hear a tone')
            setTimeout(() => {
              osc.stop()
              ctx.close()
              log('Oscillator stopped')
            }, 500)
          }}
          className="px-6 py-3 bg-secondary text-secondary-foreground rounded-lg text-lg"
        >
          Test Basic Audio
        </button>
      </div>

      {/* Program test buttons - only show after synth is loaded */}
      {synthRef.current !== null && (
        <div className="mt-4">
          <p className="text-sm text-muted-foreground mb-2">Test different programs (click after loading):</p>
          <div className="flex flex-wrap gap-2">
            {[0, 40, 41, 42, 43, 48, 56, 60, 73].map((program) => (
              <button
                key={program}
                onClick={() => {
                  const synth = synthRef.current as {
                    noteOn: (ch: number, note: number, vel: number) => void
                    noteOff: (ch: number, note: number) => void
                    programChange: (ch: number, prog: number) => void
                  }
                  ctxRef.current?.resume()
                  log(`Testing program ${program}...`)
                  synth.programChange(0, program)
                  synth.noteOn(0, 60, 127)
                  setTimeout(() => synth.noteOff(0, 60), 1000)
                }}
                className="px-3 py-1 text-sm bg-muted rounded hover:bg-muted/80"
              >
                P{program}
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="mt-6 p-4 bg-muted rounded-lg">
        <p className="font-mono text-sm font-semibold">{status}</p>
      </div>

      <div className="mt-6">
        <div className="flex items-center justify-between mb-2">
          <h2 className="font-semibold">Logs</h2>
          <button
            onClick={handleCopy}
            className="flex items-center gap-2 px-3 py-1 text-sm bg-secondary rounded hover:bg-secondary/80"
            title="Copy logs to clipboard"
          >
            {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
            {copied ? 'Copied!' : 'Copy'}
          </button>
        </div>
        <div className="p-4 bg-black text-green-400 rounded-lg font-mono text-xs max-h-96 overflow-y-auto">
          {logs.length === 0 ? (
            <p className="text-gray-500">Logs will appear here...</p>
          ) : (
            logs.map((entry, i) => (
              <div key={i}>{entry}</div>
            ))
          )}
        </div>
      </div>

      <div className="mt-6 text-sm text-muted-foreground">
        <p><strong>Expected:</strong> You should hear a piano note for 1 second.</p>
        <p className="mt-2"><strong>If no sound:</strong> Copy logs above and paste them.</p>
      </div>
    </div>
  )
}
