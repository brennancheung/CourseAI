#!/usr/bin/env npx tsx
/**
 * SFZ to SF2 Converter
 *
 * Converts Sonatina Symphonic Orchestra SFZ files to a single SF2 soundfont.
 * Uses spessasynth_core for SF2 writing.
 */

import * as fs from 'fs'
import * as path from 'path'

// Parse note name or MIDI number to MIDI number
function noteToMidi(note: string): number {
  // First check if it's a raw MIDI number (0-127)
  const midiNum = parseInt(note)
  if (!isNaN(midiNum) && midiNum >= 0 && midiNum <= 127) {
    return midiNum
  }

  const noteMap: Record<string, number> = {
    'c': 0, 'c#': 1, 'db': 1, 'd': 2, 'd#': 3, 'eb': 3,
    'e': 4, 'f': 5, 'f#': 6, 'gb': 6, 'g': 7, 'g#': 8,
    'ab': 8, 'a': 9, 'a#': 10, 'bb': 10, 'b': 11
  }

  const match = note.toLowerCase().match(/^([a-g][#b]?)(-?\d+)$/)
  if (!match) {
    console.warn(`Could not parse note: ${note}`)
    return 60 // Default to middle C
  }

  const [, noteName, octaveStr] = match
  const octave = parseInt(octaveStr)
  const semitone = noteMap[noteName]

  if (semitone === undefined) {
    console.warn(`Unknown note name: ${noteName}`)
    return 60
  }

  return (octave + 1) * 12 + semitone
}

// SFZ region data
interface SfzRegion {
  sample: string
  lokey: number
  hikey: number
  pitchKeycenter: number
  tune: number
  volume: number
}

// Recursively expand SFZ content, handling #include
function expandSfzIncludes(content: string, baseDir: string, seen: Set<string> = new Set()): string {
  const lines = content.split('\n')
  const result: string[] = []

  for (const line of lines) {
    const trimmed = line.trim()

    if (trimmed.startsWith('#include')) {
      const match = trimmed.match(/#include\s+"([^"]+)"/)
      if (match) {
        const includePath = path.resolve(baseDir, match[1])
        if (!seen.has(includePath) && fs.existsSync(includePath)) {
          seen.add(includePath)
          const includeContent = fs.readFileSync(includePath, 'utf-8')
          const expanded = expandSfzIncludes(includeContent, path.dirname(includePath), seen)
          result.push(expanded)
        }
      }
    } else {
      result.push(line)
    }
  }

  return result.join('\n')
}

// Parse SFZ file
function parseSfz(content: string, sfzDir: string): SfzRegion[] {
  // First expand all includes
  const expanded = expandSfzIncludes(content, sfzDir)
  const lines = expanded.split('\n')

  const regions: SfzRegion[] = []
  let groupDefaults: Partial<SfzRegion> = {}
  let currentRegion: Partial<SfzRegion> | null = null

  for (const line of lines) {
    const trimmed = line.trim()

    // Skip comments and empty lines
    if (trimmed.startsWith('//') || trimmed === '') continue

    if (trimmed === '<group>' || trimmed.startsWith('<group>')) {
      // Save current region if exists
      if (currentRegion?.sample) {
        regions.push({ ...groupDefaults, ...currentRegion } as SfzRegion)
      }
      currentRegion = null
      groupDefaults = {
        lokey: 0,
        hikey: 127,
        pitchKeycenter: 60,
        tune: 0,
        volume: 0
      }
      continue
    }

    if (trimmed === '<region>' || trimmed.startsWith('<region>')) {
      // Save current region if exists
      if (currentRegion?.sample) {
        regions.push({ ...groupDefaults, ...currentRegion } as SfzRegion)
      }
      currentRegion = {}
      continue
    }

    // Skip other headers
    if (trimmed.startsWith('<')) continue

    // Parse opcode=value pairs
    // They can be on one line space-separated or one per line
    const opcodeMatches = trimmed.matchAll(/([a-z_]+)=([^\s]+)/gi)

    for (const match of opcodeMatches) {
      const key = match[1].toLowerCase()
      const value = match[2]

      const target = currentRegion !== null ? currentRegion : groupDefaults

      switch (key) {
        case 'sample': {
          // Resolve relative path
          const samplePath = path.resolve(sfzDir, value.replace(/\\/g, '/'))
          target.sample = samplePath
          break
        }
        case 'lokey':
          target.lokey = noteToMidi(value)
          break
        case 'hikey':
          target.hikey = noteToMidi(value)
          break
        case 'pitch_keycenter':
          target.pitchKeycenter = noteToMidi(value)
          break
        case 'tune':
          target.tune = parseInt(value) || 0
          break
        case 'volume':
          target.volume = parseFloat(value) || 0
          break
      }
    }
  }

  // Don't forget the last region
  if (currentRegion?.sample) {
    regions.push({ ...groupDefaults, ...currentRegion } as SfzRegion)
  }

  return regions
}

// Read WAV file and extract audio data
function readWav(filePath: string): { sampleRate: number; data: Int16Array } | null {
  try {
    if (!fs.existsSync(filePath)) {
      return null
    }

    const stat = fs.statSync(filePath)
    if (stat.isDirectory()) {
      return null
    }

    const buffer = fs.readFileSync(filePath)
    if (buffer.length < 44) {
      return null
    }

    const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength)

    // Check RIFF header
    const riff = String.fromCharCode(buffer[0], buffer[1], buffer[2], buffer[3])
    if (riff !== 'RIFF') {
      return null
    }

    const wave = String.fromCharCode(buffer[8], buffer[9], buffer[10], buffer[11])
    if (wave !== 'WAVE') {
      return null
    }

    // Find fmt and data chunks
    let offset = 12
    let sampleRate = 44100
    let bitsPerSample = 16
    let numChannels = 1
    let dataStart = 0
    let dataSize = 0

    while (offset < buffer.length - 8) {
      const chunkId = String.fromCharCode(buffer[offset], buffer[offset+1], buffer[offset+2], buffer[offset+3])
      const chunkSize = view.getUint32(offset + 4, true)

      if (chunkId === 'fmt ') {
        numChannels = view.getUint16(offset + 10, true)
        sampleRate = view.getUint32(offset + 12, true)
        bitsPerSample = view.getUint16(offset + 22, true)
      } else if (chunkId === 'data') {
        dataStart = offset + 8
        dataSize = chunkSize
        break
      }

      offset += 8 + chunkSize
      if (chunkSize % 2 !== 0) offset++ // Padding
    }

    if (dataStart === 0 || dataSize === 0) {
      return null
    }

    // Extract audio data as 16-bit PCM (mono, take left channel if stereo)
    const bytesPerSample = bitsPerSample / 8
    const frameSize = bytesPerSample * numChannels
    const numFrames = Math.floor(dataSize / frameSize)
    const data = new Int16Array(numFrames)

    for (let i = 0; i < numFrames; i++) {
      const frameOffset = dataStart + i * frameSize

      if (bitsPerSample === 16) {
        data[i] = view.getInt16(frameOffset, true)
      } else if (bitsPerSample === 24) {
        const b0 = buffer[frameOffset]
        const b1 = buffer[frameOffset + 1]
        const b2 = buffer[frameOffset + 2]
        const val = (b2 << 16) | (b1 << 8) | b0
        const signed = val >= 0x800000 ? val - 0x1000000 : val
        data[i] = Math.round(signed / 256)
      } else if (bitsPerSample === 8) {
        data[i] = (buffer[frameOffset] - 128) * 256
      }
    }

    return { sampleRate, data }
  } catch {
    return null
  }
}

// Write SF2 file manually (since spessasynth_core API is complex)
function writeSf2(
  outputPath: string,
  presets: Array<{
    name: string
    program: number
    regions: Array<{
      sampleData: Int16Array
      sampleRate: number
      lokey: number
      hikey: number
      rootKey: number
      tune: number
      volume: number
    }>
  }>
) {
  // Collect all unique samples
  const allSamples: Array<{
    name: string
    data: Int16Array
    sampleRate: number
    rootKey: number
    tune: number
  }> = []

  const sampleMap = new Map<Int16Array, number>()

  for (const preset of presets) {
    for (const region of preset.regions) {
      if (!sampleMap.has(region.sampleData)) {
        sampleMap.set(region.sampleData, allSamples.length)
        allSamples.push({
          name: `Sample${allSamples.length}`,
          data: region.sampleData,
          sampleRate: region.sampleRate,
          rootKey: region.rootKey,
          tune: region.tune
        })
      }
    }
  }

  console.log(`  Total unique samples: ${allSamples.length}`)

  // Calculate sizes
  let sampleDataSize = 0
  for (const sample of allSamples) {
    sampleDataSize += sample.data.length * 2 + 46 // 46 zero samples at end
  }
  sampleDataSize += 46 // Terminal sample

  // Build sample data buffer
  const sampleData = new Int16Array(sampleDataSize / 2)
  let sampleOffset = 0
  const sampleOffsets: number[] = []

  for (const sample of allSamples) {
    sampleOffsets.push(sampleOffset)
    sampleData.set(sample.data, sampleOffset)
    sampleOffset += sample.data.length + 23 // 46 bytes / 2 = 23 samples padding
  }

  // Build SF2 structure
  const chunks: ArrayBuffer[] = []

  // Helper to write string padded to length
  function writeString(str: string, len: number): Uint8Array {
    const arr = new Uint8Array(len)
    for (let i = 0; i < Math.min(str.length, len - 1); i++) {
      arr[i] = str.charCodeAt(i)
    }
    return arr
  }

  // Helper to write 16-bit LE
  function write16(val: number): Uint8Array {
    const arr = new Uint8Array(2)
    arr[0] = val & 0xff
    arr[1] = (val >> 8) & 0xff
    return arr
  }

  // Helper to write 32-bit LE
  function write32(val: number): Uint8Array {
    const arr = new Uint8Array(4)
    arr[0] = val & 0xff
    arr[1] = (val >> 8) & 0xff
    arr[2] = (val >> 16) & 0xff
    arr[3] = (val >> 24) & 0xff
    return arr
  }

  // INFO chunk
  const infoChunks: Uint8Array[] = []

  // ifil - version
  infoChunks.push(new Uint8Array([0x69, 0x66, 0x69, 0x6c])) // "ifil"
  infoChunks.push(write32(4))
  infoChunks.push(write16(2)) // major
  infoChunks.push(write16(1)) // minor

  // isng - sound engine
  const isng = "EMU8000"
  infoChunks.push(new Uint8Array([0x69, 0x73, 0x6e, 0x67])) // "isng"
  infoChunks.push(write32(isng.length + 1))
  infoChunks.push(writeString(isng, isng.length + 1))
  if ((isng.length + 1) % 2 !== 0) infoChunks.push(new Uint8Array(1))

  // INAM - name
  const inam = "Sonatina Symphonic"
  infoChunks.push(new Uint8Array([0x49, 0x4e, 0x41, 0x4d])) // "INAM"
  infoChunks.push(write32(inam.length + 1))
  infoChunks.push(writeString(inam, inam.length + 1))
  if ((inam.length + 1) % 2 !== 0) infoChunks.push(new Uint8Array(1))

  let infoSize = 4 // "INFO"
  for (const chunk of infoChunks) infoSize += chunk.length

  // sdta chunk (sample data)
  const sdtaSize = 4 + 4 + 4 + sampleDataSize // "sdta" + "smpl" header + data

  // pdta chunk - this is complex, let's build it
  const pdtaChunks: Uint8Array[] = []

  // phdr - preset headers (38 bytes each)
  const phdrData: Uint8Array[] = []
  let presetBagIndex = 0
  for (let i = 0; i < presets.length; i++) {
    const preset = presets[i]
    phdrData.push(writeString(preset.name, 20))
    phdrData.push(write16(preset.program)) // preset
    phdrData.push(write16(0)) // bank
    phdrData.push(write16(presetBagIndex)) // preset bag index
    phdrData.push(write32(0)) // library
    phdrData.push(write32(0)) // genre
    phdrData.push(write32(0)) // morphology
    presetBagIndex += 1 // One zone per preset
  }
  // Terminal
  phdrData.push(writeString("EOP", 20))
  phdrData.push(write16(0))
  phdrData.push(write16(0))
  phdrData.push(write16(presetBagIndex))
  phdrData.push(write32(0))
  phdrData.push(write32(0))
  phdrData.push(write32(0))

  let phdrSize = 0
  for (const d of phdrData) phdrSize += d.length

  pdtaChunks.push(new Uint8Array([0x70, 0x68, 0x64, 0x72])) // "phdr"
  pdtaChunks.push(write32(phdrSize))
  pdtaChunks.push(...phdrData)

  // pbag - preset bags (4 bytes each)
  const pbagData: Uint8Array[] = []
  let genIndex = 0
  let modIndex = 0
  for (let i = 0; i < presets.length; i++) {
    pbagData.push(write16(genIndex)) // gen index
    pbagData.push(write16(modIndex)) // mod index
    genIndex += 1 // instrument generator
  }
  // Terminal
  pbagData.push(write16(genIndex))
  pbagData.push(write16(modIndex))

  let pbagSize = 0
  for (const d of pbagData) pbagSize += d.length

  pdtaChunks.push(new Uint8Array([0x70, 0x62, 0x61, 0x67])) // "pbag"
  pdtaChunks.push(write32(pbagSize))
  pdtaChunks.push(...pbagData)

  // pmod - preset modulators (10 bytes each, just terminal)
  pdtaChunks.push(new Uint8Array([0x70, 0x6d, 0x6f, 0x64])) // "pmod"
  pdtaChunks.push(write32(10))
  pdtaChunks.push(new Uint8Array(10))

  // pgen - preset generators (4 bytes each)
  const pgenData: Uint8Array[] = []
  for (let i = 0; i < presets.length; i++) {
    // instrument generator
    pgenData.push(write16(41)) // instrument generator type
    pgenData.push(write16(i))  // instrument index
  }
  // Terminal
  pgenData.push(write16(0))
  pgenData.push(write16(0))

  let pgenSize = 0
  for (const d of pgenData) pgenSize += d.length

  pdtaChunks.push(new Uint8Array([0x70, 0x67, 0x65, 0x6e])) // "pgen"
  pdtaChunks.push(write32(pgenSize))
  pdtaChunks.push(...pgenData)

  // inst - instrument headers (22 bytes each)
  const instData: Uint8Array[] = []
  let instBagIndex = 0
  for (let i = 0; i < presets.length; i++) {
    instData.push(writeString(presets[i].name, 20))
    instData.push(write16(instBagIndex))
    instBagIndex += presets[i].regions.length
  }
  // Terminal
  instData.push(writeString("EOI", 20))
  instData.push(write16(instBagIndex))

  let instSize = 0
  for (const d of instData) instSize += d.length

  pdtaChunks.push(new Uint8Array([0x69, 0x6e, 0x73, 0x74])) // "inst"
  pdtaChunks.push(write32(instSize))
  pdtaChunks.push(...instData)

  // ibag - instrument bags (4 bytes each)
  const ibagData: Uint8Array[] = []
  let igenIndex = 0
  let imodIndex = 0
  for (const preset of presets) {
    for (let j = 0; j < preset.regions.length; j++) {
      ibagData.push(write16(igenIndex))
      ibagData.push(write16(imodIndex))
      igenIndex += 4 // keyRange, rootKey, sampleID, sampleModes
    }
  }
  // Terminal
  ibagData.push(write16(igenIndex))
  ibagData.push(write16(imodIndex))

  let ibagSize = 0
  for (const d of ibagData) ibagSize += d.length

  pdtaChunks.push(new Uint8Array([0x69, 0x62, 0x61, 0x67])) // "ibag"
  pdtaChunks.push(write32(ibagSize))
  pdtaChunks.push(...ibagData)

  // imod - instrument modulators (just terminal)
  pdtaChunks.push(new Uint8Array([0x69, 0x6d, 0x6f, 0x64])) // "imod"
  pdtaChunks.push(write32(10))
  pdtaChunks.push(new Uint8Array(10))

  // igen - instrument generators (4 bytes each)
  const igenData: Uint8Array[] = []
  for (const preset of presets) {
    for (const region of preset.regions) {
      const sampleIdx = sampleMap.get(region.sampleData) ?? 0

      // keyRange
      igenData.push(write16(43)) // keyRange type
      igenData.push(write16((region.hikey << 8) | region.lokey))

      // overridingRootKey
      igenData.push(write16(58)) // overridingRootKey type
      igenData.push(write16(region.rootKey))

      // sampleID
      igenData.push(write16(53)) // sampleID type
      igenData.push(write16(sampleIdx))

      // sampleModes
      igenData.push(write16(54)) // sampleModes type
      igenData.push(write16(0)) // no loop
    }
  }
  // Terminal
  igenData.push(write16(0))
  igenData.push(write16(0))

  let igenSize = 0
  for (const d of igenData) igenSize += d.length

  pdtaChunks.push(new Uint8Array([0x69, 0x67, 0x65, 0x6e])) // "igen"
  pdtaChunks.push(write32(igenSize))
  pdtaChunks.push(...igenData)

  // shdr - sample headers (46 bytes each)
  const shdrData: Uint8Array[] = []
  for (let i = 0; i < allSamples.length; i++) {
    const sample = allSamples[i]
    const startOffset = sampleOffsets[i]
    const endOffset = startOffset + sample.data.length

    shdrData.push(writeString(sample.name, 20))
    shdrData.push(write32(startOffset)) // start
    shdrData.push(write32(endOffset))   // end
    shdrData.push(write32(startOffset)) // loop start
    shdrData.push(write32(endOffset - 1)) // loop end
    shdrData.push(write32(sample.sampleRate))
    shdrData.push(new Uint8Array([sample.rootKey])) // original pitch
    shdrData.push(new Uint8Array([sample.tune >= 0 ? sample.tune : 256 + sample.tune])) // pitch correction
    shdrData.push(write16(0)) // sample link
    shdrData.push(write16(1)) // sample type (mono)
  }
  // Terminal
  shdrData.push(writeString("EOS", 20))
  shdrData.push(write32(0))
  shdrData.push(write32(0))
  shdrData.push(write32(0))
  shdrData.push(write32(0))
  shdrData.push(write32(0))
  shdrData.push(new Uint8Array([0]))
  shdrData.push(new Uint8Array([0]))
  shdrData.push(write16(0))
  shdrData.push(write16(0))

  let shdrSize = 0
  for (const d of shdrData) shdrSize += d.length

  pdtaChunks.push(new Uint8Array([0x73, 0x68, 0x64, 0x72])) // "shdr"
  pdtaChunks.push(write32(shdrSize))
  pdtaChunks.push(...shdrData)

  let pdtaSize = 4 // "pdta"
  for (const chunk of pdtaChunks) pdtaSize += chunk.length

  // Calculate total size
  const totalSize = 4 + // "sfbk"
    4 + 4 + infoSize + // LIST + size + INFO
    4 + 4 + sdtaSize + // LIST + size + sdta
    4 + 4 + pdtaSize   // LIST + size + pdta

  // Write file
  const output = new Uint8Array(8 + totalSize)
  let pos = 0

  function writeBytes(arr: Uint8Array) {
    output.set(arr, pos)
    pos += arr.length
  }

  // RIFF header
  writeBytes(new Uint8Array([0x52, 0x49, 0x46, 0x46])) // "RIFF"
  writeBytes(write32(totalSize))
  writeBytes(new Uint8Array([0x73, 0x66, 0x62, 0x6b])) // "sfbk"

  // INFO LIST
  writeBytes(new Uint8Array([0x4c, 0x49, 0x53, 0x54])) // "LIST"
  writeBytes(write32(infoSize))
  writeBytes(new Uint8Array([0x49, 0x4e, 0x46, 0x4f])) // "INFO"
  for (const chunk of infoChunks) writeBytes(chunk)

  // sdta LIST
  writeBytes(new Uint8Array([0x4c, 0x49, 0x53, 0x54])) // "LIST"
  writeBytes(write32(sdtaSize))
  writeBytes(new Uint8Array([0x73, 0x64, 0x74, 0x61])) // "sdta"
  writeBytes(new Uint8Array([0x73, 0x6d, 0x70, 0x6c])) // "smpl"
  writeBytes(write32(sampleDataSize))

  // Write sample data
  const sampleBytes = new Uint8Array(sampleData.buffer)
  writeBytes(sampleBytes)

  // pdta LIST
  writeBytes(new Uint8Array([0x4c, 0x49, 0x53, 0x54])) // "LIST"
  writeBytes(write32(pdtaSize))
  writeBytes(new Uint8Array([0x70, 0x64, 0x74, 0x61])) // "pdta"
  for (const chunk of pdtaChunks) writeBytes(chunk)

  fs.writeFileSync(outputPath, output)
  console.log(`  Written ${output.length} bytes`)
}

async function main() {
  const ssoPath = '/Users/brennan/Downloads/sso-4.0/Sonatina Symphonic Orchestra'
  const outputPath = '/Users/brennan/code/orchestral/public/soundfonts/sonatina-converted.sf2'

  // Find all SFZ files (skip the "includes" and notation folders for simplicity)
  const sfzFiles: string[] = []

  function findSfzFiles(dir: string) {
    const entries = fs.readdirSync(dir, { withFileTypes: true })
    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name)
      if (entry.isDirectory()) {
        // Skip includes folder
        if (entry.name !== 'includes') {
          findSfzFiles(fullPath)
        }
      } else if (entry.name.endsWith('.sfz') && !entry.name.includes('KS')) {
        // Skip KS (keyswitch) variants for now
        sfzFiles.push(fullPath)
      }
    }
  }

  findSfzFiles(ssoPath)
  console.log(`Found ${sfzFiles.length} SFZ files`)

  // Process each SFZ file into a preset
  const presets: Array<{
    name: string
    program: number
    regions: Array<{
      sampleData: Int16Array
      sampleRate: number
      lokey: number
      hikey: number
      rootKey: number
      tune: number
      volume: number
    }>
  }> = []

  const sampleCache = new Map<string, { data: Int16Array; sampleRate: number }>()
  let programNumber = 0

  for (const sfzFile of sfzFiles) {
    const sfzDir = path.dirname(sfzFile)
    const sfzName = path.basename(sfzFile, '.sfz')

    console.log(`Processing: ${sfzName}`)

    const sfzContent = fs.readFileSync(sfzFile, 'utf-8')
    const regions = parseSfz(sfzContent, sfzDir)

    if (regions.length === 0) {
      console.log(`  Skipping (no regions)`)
      continue
    }

    const presetRegions: typeof presets[0]['regions'] = []

    for (const region of regions) {
      // Load or reuse sample
      let cached = sampleCache.get(region.sample)

      if (!cached) {
        const wavData = readWav(region.sample)
        if (!wavData) {
          continue
        }
        cached = wavData
        sampleCache.set(region.sample, cached)
      }

      presetRegions.push({
        sampleData: cached.data,
        sampleRate: cached.sampleRate,
        lokey: region.lokey,
        hikey: region.hikey,
        rootKey: region.pitchKeycenter,
        tune: region.tune,
        volume: region.volume
      })
    }

    if (presetRegions.length === 0) {
      console.log(`  Skipping (no valid samples)`)
      continue
    }

    presets.push({
      name: sfzName.substring(0, 20),
      program: programNumber,
      regions: presetRegions
    })

    console.log(`  Added: ${presetRegions.length} regions, program ${programNumber}`)
    programNumber++
  }

  console.log(`\nTotal: ${presets.length} presets`)

  // Write SF2
  console.log(`\nWriting SF2 to: ${outputPath}`)
  writeSf2(outputPath, presets)

  const stat = fs.statSync(outputPath)
  console.log(`Done! File size: ${(stat.size / 1024 / 1024).toFixed(2)} MB`)
}

main().catch(console.error)
