/**
 * Pre-computed data for the VAE Latent Space Widget.
 *
 * Contains:
 * - Encoded Fashion-MNIST items positioned in a 2D latent space
 * - "Decoded" pixel grids for various latent space regions
 * - Data for both Autoencoder mode (scattered points, gaps) and VAE mode (smooth density)
 *
 * All images are 14x14 (196 values, 0-255 integers) matching the autoencoder widget format.
 */

export type LatentPoint = {
  id: string
  label: string
  /** Category for coloring */
  category: 'tshirt' | 'trouser' | 'sneaker' | 'bag'
  /** Position in normalized latent space [-3, 3] */
  x: number
  y: number
  /** 14x14 pixel grid for this item's reconstruction */
  pixels: number[]
}

export type SampledResult = {
  /** Position in normalized latent space [-3, 3] */
  x: number
  y: number
  /** 14x14 pixel grid — garbage in AE mode, plausible in VAE mode */
  aePixels: number[]
  vaePixels: number[]
}

// ---------------------------------------------------------------------------
// Category colors
// ---------------------------------------------------------------------------

export const CATEGORY_COLORS: Record<LatentPoint['category'], string> = {
  tshirt: '#6366f1',   // indigo
  trouser: '#f59e0b',  // amber
  sneaker: '#10b981',  // emerald
  bag: '#f43f5e',      // rose
}

export const CATEGORY_LABELS: Record<LatentPoint['category'], string> = {
  tshirt: 'T-Shirt',
  trouser: 'Trouser',
  sneaker: 'Sneaker',
  bag: 'Bag',
}

// ---------------------------------------------------------------------------
// Helper: generate a plausible Fashion-MNIST-style 14x14 pixel grid
// ---------------------------------------------------------------------------

function mulberry32(seed: number) {
  return () => {
    let t = (seed += 0x6d2b79f5)
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function clamp(v: number, min: number, max: number): number {
  if (v < min) return min
  if (v > max) return max
  return v
}

// Generate a rough T-shirt-like shape scaled by intensity
function generateTshirt(intensity: number, seed: number): number[] {
  const rng = mulberry32(seed)
  const base = [
    0,0,0,0,5,15,25,25,15,5,0,0,0,0,
    0,0,5,40,90,130,155,155,130,90,40,5,0,0,
    0,15,80,150,170,80,35,35,80,170,150,80,15,0,
    5,70,160,185,175,160,150,150,160,175,185,160,70,5,
    10,85,170,190,180,165,155,155,165,180,190,170,85,10,
    0,45,140,185,180,170,160,160,170,180,185,140,45,0,
    0,10,95,175,175,168,158,158,168,175,175,95,10,0,
    0,5,80,170,172,165,155,155,165,172,170,80,5,0,
    0,5,75,165,168,162,152,152,162,168,165,75,5,0,
    0,5,70,160,165,160,150,150,160,165,160,70,5,0,
    0,5,65,155,162,158,148,148,158,162,155,65,5,0,
    0,5,60,150,160,155,145,145,155,160,150,60,5,0,
    0,0,40,130,150,148,140,140,148,150,130,40,0,0,
    0,0,10,60,90,95,95,95,95,90,60,10,0,0,
  ]
  return base.map(v => clamp(Math.round(v * intensity + (rng() - 0.5) * 20), 0, 255))
}

function generateTrouser(intensity: number, seed: number): number[] {
  const rng = mulberry32(seed)
  const base = [
    0,0,5,50,90,120,140,140,120,90,50,5,0,0,
    0,0,10,65,110,145,160,160,145,110,65,10,0,0,
    0,0,15,70,115,148,162,162,148,115,70,15,0,0,
    0,0,15,72,118,150,145,145,150,118,72,15,0,0,
    0,0,15,75,120,145,90,90,145,120,75,15,0,0,
    0,0,15,78,122,140,30,30,140,122,78,15,0,0,
    0,0,15,80,125,135,10,10,135,125,80,15,0,0,
    0,0,15,82,128,130,5,5,130,128,82,15,0,0,
    0,0,15,85,130,128,3,3,128,130,85,15,0,0,
    0,0,15,85,132,125,2,2,125,132,85,15,0,0,
    0,0,15,88,135,122,0,0,122,135,88,15,0,0,
    0,0,12,85,132,118,0,0,118,132,85,12,0,0,
    0,0,10,78,125,110,0,0,110,125,78,10,0,0,
    0,0,5,55,95,85,0,0,85,95,55,5,0,0,
  ]
  return base.map(v => clamp(Math.round(v * intensity + (rng() - 0.5) * 20), 0, 255))
}

function generateSneaker(intensity: number, seed: number): number[] {
  const rng = mulberry32(seed)
  const base = [
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,10,30,0,0,
    0,0,0,0,0,0,0,0,5,35,80,110,25,0,
    0,0,0,0,0,0,0,15,55,100,135,140,60,0,
    0,0,0,0,0,5,25,65,105,135,150,148,80,5,
    0,0,0,0,8,35,75,115,140,150,155,145,90,10,
    0,0,0,10,45,85,120,140,150,152,148,135,85,12,
    0,0,5,40,85,120,142,155,160,155,145,128,78,10,
    0,5,25,68,110,138,155,165,168,162,150,130,72,8,
    5,20,55,95,130,152,168,178,180,175,162,140,82,10,
    15,50,90,130,160,180,195,205,205,195,180,158,95,15,
    10,40,75,110,140,165,185,195,195,188,170,148,85,12,
    0,8,25,50,75,90,100,105,105,100,88,65,30,0,
  ]
  return base.map(v => clamp(Math.round(v * intensity + (rng() - 0.5) * 20), 0, 255))
}

function generateBag(intensity: number, seed: number): number[] {
  const rng = mulberry32(seed)
  const base = [
    0,0,0,0,15,40,55,55,40,15,0,0,0,0,
    0,0,0,20,60,30,5,5,30,60,20,0,0,0,
    0,0,5,45,50,10,0,0,10,50,45,5,0,0,
    0,5,40,95,130,145,150,150,145,130,95,40,5,0,
    0,10,65,135,165,178,185,185,178,165,135,65,10,0,
    0,10,70,140,170,182,188,188,182,170,140,70,10,0,
    0,10,68,138,168,180,185,185,180,168,138,68,10,0,
    0,10,65,135,165,178,182,182,178,165,135,65,10,0,
    0,10,62,132,162,175,180,180,175,162,132,62,10,0,
    0,10,60,128,158,172,178,178,172,158,128,60,10,0,
    0,10,58,125,155,168,175,175,168,155,125,58,10,0,
    0,8,55,120,150,165,170,170,165,150,120,55,8,0,
    0,5,42,105,140,155,162,162,155,140,105,42,5,0,
    0,0,15,55,85,98,105,105,98,85,55,15,0,0,
  ]
  return base.map(v => clamp(Math.round(v * intensity + (rng() - 0.5) * 20), 0, 255))
}

// Generate random noise (garbage for AE gaps)
function generateGarbage(seed: number): number[] {
  const rng = mulberry32(seed)
  return Array.from({ length: 196 }, () => {
    const v = rng()
    // Mostly dark with random noise patches
    return Math.round(v * v * 120 + rng() * 30)
  })
}

// Generate a blurred/blended shape (for VAE high-beta or between-category regions)
function generateBlurry(category: LatentPoint['category'], blurAmount: number, seed: number): number[] {
  const rng = mulberry32(seed)
  const generators = {
    tshirt: generateTshirt,
    trouser: generateTrouser,
    sneaker: generateSneaker,
    bag: generateBag,
  }
  const base = generators[category](0.7, seed)
  // Apply blur by averaging with neighbors
  const result = [...base]
  const blurPasses = Math.round(blurAmount * 3)
  for (let pass = 0; pass < blurPasses; pass++) {
    const prev = [...result]
    for (let i = 0; i < 196; i++) {
      const row = Math.floor(i / 14)
      const col = i % 14
      let sum = prev[i] * 2
      let count = 2
      if (col > 0) { sum += prev[i - 1]; count++ }
      if (col < 13) { sum += prev[i + 1]; count++ }
      if (row > 0) { sum += prev[i - 14]; count++ }
      if (row < 13) { sum += prev[i + 14]; count++ }
      result[i] = Math.round(sum / count + (rng() - 0.5) * 5)
    }
  }
  return result.map(v => clamp(v, 0, 255))
}

// ---------------------------------------------------------------------------
// Encoded Fashion-MNIST points in the 2D latent space
// Positions chosen to reflect realistic clustering with some overlap
// ---------------------------------------------------------------------------

export const ENCODED_POINTS: LatentPoint[] = [
  // T-shirts cluster: upper-left region
  { id: 't1', label: 'T-Shirt 1', category: 'tshirt', x: -1.8, y: 1.5, pixels: generateTshirt(1.0, 100) },
  { id: 't2', label: 'T-Shirt 2', category: 'tshirt', x: -1.2, y: 1.8, pixels: generateTshirt(0.95, 200) },
  { id: 't3', label: 'T-Shirt 3', category: 'tshirt', x: -2.1, y: 1.0, pixels: generateTshirt(1.05, 300) },
  { id: 't4', label: 'T-Shirt 4', category: 'tshirt', x: -1.5, y: 1.2, pixels: generateTshirt(0.9, 400) },
  { id: 't5', label: 'T-Shirt 5', category: 'tshirt', x: -0.8, y: 1.6, pixels: generateTshirt(1.0, 500) },
  { id: 't6', label: 'T-Shirt 6', category: 'tshirt', x: -1.0, y: 0.9, pixels: generateTshirt(0.92, 600) },
  { id: 't7', label: 'T-Shirt 7', category: 'tshirt', x: -2.4, y: 1.3, pixels: generateTshirt(0.88, 700) },
  { id: 't8', label: 'T-Shirt 8', category: 'tshirt', x: -1.6, y: 2.0, pixels: generateTshirt(1.02, 800) },
  { id: 't9', label: 'T-Shirt 9', category: 'tshirt', x: -0.5, y: 1.3, pixels: generateTshirt(0.97, 900) },
  { id: 't10', label: 'T-Shirt 10', category: 'tshirt', x: -1.3, y: 0.7, pixels: generateTshirt(1.0, 1000) },
  { id: 't11', label: 'T-Shirt 11', category: 'tshirt', x: -1.9, y: 0.5, pixels: generateTshirt(0.93, 1100) },
  { id: 't12', label: 'T-Shirt 12', category: 'tshirt', x: -0.3, y: 1.1, pixels: generateTshirt(1.01, 1200) },

  // Trousers cluster: lower-left region
  { id: 'tr1', label: 'Trouser 1', category: 'trouser', x: -1.5, y: -1.2, pixels: generateTrouser(1.0, 2100) },
  { id: 'tr2', label: 'Trouser 2', category: 'trouser', x: -1.0, y: -1.5, pixels: generateTrouser(0.95, 2200) },
  { id: 'tr3', label: 'Trouser 3', category: 'trouser', x: -2.0, y: -1.0, pixels: generateTrouser(1.03, 2300) },
  { id: 'tr4', label: 'Trouser 4', category: 'trouser', x: -1.3, y: -0.8, pixels: generateTrouser(0.9, 2400) },
  { id: 'tr5', label: 'Trouser 5', category: 'trouser', x: -0.7, y: -1.8, pixels: generateTrouser(1.0, 2500) },
  { id: 'tr6', label: 'Trouser 6', category: 'trouser', x: -1.8, y: -1.5, pixels: generateTrouser(0.92, 2600) },
  { id: 'tr7', label: 'Trouser 7', category: 'trouser', x: -0.4, y: -1.1, pixels: generateTrouser(0.98, 2700) },
  { id: 'tr8', label: 'Trouser 8', category: 'trouser', x: -1.6, y: -0.5, pixels: generateTrouser(1.05, 2800) },
  { id: 'tr9', label: 'Trouser 9', category: 'trouser', x: -2.3, y: -1.3, pixels: generateTrouser(0.88, 2900) },
  { id: 'tr10', label: 'Trouser 10', category: 'trouser', x: -0.2, y: -1.4, pixels: generateTrouser(1.0, 3000) },
  { id: 'tr11', label: 'Trouser 11', category: 'trouser', x: -1.1, y: -2.0, pixels: generateTrouser(0.94, 3100) },
  { id: 'tr12', label: 'Trouser 12', category: 'trouser', x: -1.7, y: -0.3, pixels: generateTrouser(0.96, 3200) },

  // Sneakers cluster: right region
  { id: 's1', label: 'Sneaker 1', category: 'sneaker', x: 1.5, y: -0.5, pixels: generateSneaker(1.0, 3100) },
  { id: 's2', label: 'Sneaker 2', category: 'sneaker', x: 1.8, y: 0.2, pixels: generateSneaker(0.95, 3200) },
  { id: 's3', label: 'Sneaker 3', category: 'sneaker', x: 1.2, y: -1.0, pixels: generateSneaker(1.03, 3300) },
  { id: 's4', label: 'Sneaker 4', category: 'sneaker', x: 2.0, y: -0.3, pixels: generateSneaker(0.92, 3400) },
  { id: 's5', label: 'Sneaker 5', category: 'sneaker', x: 1.0, y: 0.0, pixels: generateSneaker(1.0, 3500) },
  { id: 's6', label: 'Sneaker 6', category: 'sneaker', x: 1.6, y: 0.8, pixels: generateSneaker(0.9, 3600) },
  { id: 's7', label: 'Sneaker 7', category: 'sneaker', x: 2.3, y: 0.1, pixels: generateSneaker(1.02, 3700) },
  { id: 's8', label: 'Sneaker 8', category: 'sneaker', x: 1.3, y: 0.5, pixels: generateSneaker(0.97, 3800) },
  { id: 's9', label: 'Sneaker 9', category: 'sneaker', x: 0.8, y: -0.7, pixels: generateSneaker(1.0, 3900) },
  { id: 's10', label: 'Sneaker 10', category: 'sneaker', x: 1.9, y: -0.8, pixels: generateSneaker(0.88, 4000) },
  { id: 's11', label: 'Sneaker 11', category: 'sneaker', x: 2.1, y: 0.5, pixels: generateSneaker(0.94, 4100) },
  { id: 's12', label: 'Sneaker 12', category: 'sneaker', x: 0.5, y: -0.2, pixels: generateSneaker(1.01, 4200) },

  // Bags cluster: upper-right region
  { id: 'b1', label: 'Bag 1', category: 'bag', x: 1.0, y: 1.5, pixels: generateBag(1.0, 4100) },
  { id: 'b2', label: 'Bag 2', category: 'bag', x: 1.5, y: 1.2, pixels: generateBag(0.95, 4200) },
  { id: 'b3', label: 'Bag 3', category: 'bag', x: 0.5, y: 1.8, pixels: generateBag(1.02, 4300) },
  { id: 'b4', label: 'Bag 4', category: 'bag', x: 1.8, y: 1.6, pixels: generateBag(0.9, 4400) },
  { id: 'b5', label: 'Bag 5', category: 'bag', x: 0.8, y: 1.0, pixels: generateBag(1.0, 4500) },
  { id: 'b6', label: 'Bag 6', category: 'bag', x: 1.3, y: 2.0, pixels: generateBag(0.93, 4600) },
  { id: 'b7', label: 'Bag 7', category: 'bag', x: 2.0, y: 1.0, pixels: generateBag(0.98, 4700) },
  { id: 'b8', label: 'Bag 8', category: 'bag', x: 0.3, y: 1.3, pixels: generateBag(1.05, 4800) },
  { id: 'b9', label: 'Bag 9', category: 'bag', x: 1.6, y: 1.8, pixels: generateBag(0.87, 4900) },
  { id: 'b10', label: 'Bag 10', category: 'bag', x: 1.1, y: 0.8, pixels: generateBag(1.0, 5000) },
  { id: 'b11', label: 'Bag 11', category: 'bag', x: 0.6, y: 2.2, pixels: generateBag(0.92, 5100) },
  { id: 'b12', label: 'Bag 12', category: 'bag', x: 2.2, y: 1.4, pixels: generateBag(0.96, 5200) },
]

// ---------------------------------------------------------------------------
// Pre-computed sampling results for specific latent space regions
// These are used when the user clicks to "sample" from a region
// ---------------------------------------------------------------------------

/** Find the nearest category to a point based on cluster centers */
function nearestCategory(x: number, y: number): LatentPoint['category'] {
  const centers = {
    tshirt: { x: -1.4, y: 1.2 },
    trouser: { x: -1.3, y: -1.2 },
    sneaker: { x: 1.5, y: -0.2 },
    bag: { x: 1.2, y: 1.4 },
  }
  let nearest: LatentPoint['category'] = 'tshirt'
  let minDist = Infinity
  for (const [cat, center] of Object.entries(centers)) {
    const dist = Math.sqrt((x - center.x) ** 2 + (y - center.y) ** 2)
    if (dist < minDist) {
      minDist = dist
      nearest = cat as LatentPoint['category']
    }
  }
  return nearest
}

/** Check if a point is near any encoded point (within a threshold) */
function isNearEncodedPoint(x: number, y: number, threshold: number): boolean {
  return ENCODED_POINTS.some(p =>
    Math.sqrt((x - p.x) ** 2 + (y - p.y) ** 2) < threshold
  )
}

/**
 * Get decoded result for a sampled point.
 * In AE mode: garbage unless very close to an encoded point.
 * In VAE mode: plausible image based on nearest category, with blurriness increasing with beta.
 */
export function getDecodedSample(
  x: number,
  y: number,
  mode: 'ae' | 'vae',
  beta: number,
): number[] {
  const seed = Math.round(x * 1000 + y * 7919)

  if (mode === 'ae') {
    // In AE mode, only points very close to encoded items produce recognizable output
    const nearPoint = ENCODED_POINTS.find(p =>
      Math.sqrt((x - p.x) ** 2 + (y - p.y) ** 2) < 0.3
    )
    if (nearPoint) {
      return nearPoint.pixels
    }
    return generateGarbage(seed)
  }

  // VAE mode: generate a plausible image based on nearest category
  const cat = nearestCategory(x, y)
  const generators = {
    tshirt: generateTshirt,
    trouser: generateTrouser,
    sneaker: generateSneaker,
    bag: generateBag,
  }

  // At very low beta (near 0), behave like autoencoder
  if (beta < 0.1) {
    if (!isNearEncodedPoint(x, y, 0.4)) {
      return generateGarbage(seed)
    }
    return generators[cat](0.9, seed)
  }

  // At very high beta, everything is blurry and generic
  if (beta > 3.0) {
    return generateBlurry(cat, beta / 2.0, seed)
  }

  // Normal VAE: plausible output with some blur proportional to beta
  const blurAmount = Math.max(0, (beta - 0.5) * 0.5)
  if (blurAmount > 0.3) {
    return generateBlurry(cat, blurAmount, seed)
  }
  // Use seeded RNG so the same (x, y) always produces the same decoded image.
  // A real VAE decoder is deterministic given a fixed z.
  const intensityRng = mulberry32(seed + 9973)
  return generators[cat](0.85 + intensityRng() * 0.15, seed)
}

// ---------------------------------------------------------------------------
// Pre-computed sample points for the "Sample Random Point" button
// These ensure consistent, visually meaningful results
// ---------------------------------------------------------------------------

export const PRESET_SAMPLES: Array<{ x: number; y: number }> = [
  // In gaps (between clusters)
  { x: 0.0, y: 0.0 },    // dead center — gap in AE, blend in VAE
  { x: -0.5, y: 0.0 },   // between tshirt/trouser and sneaker/bag
  { x: 0.5, y: 0.5 },    // between sneaker and bag clusters
  { x: -1.0, y: 0.0 },   // between tshirt and trouser clusters
  { x: 0.0, y: 1.5 },    // between tshirt and bag clusters
  { x: 0.0, y: -1.0 },   // between trouser and sneaker clusters
  // In clusters
  { x: -1.5, y: 1.3 },   // tshirt region
  { x: -1.2, y: -1.3 },  // trouser region
  { x: 1.6, y: -0.1 },   // sneaker region
  { x: 1.2, y: 1.5 },    // bag region
  // Edge of space
  { x: 2.5, y: 2.5 },    // far corner
  { x: -2.5, y: -2.5 },  // far corner
]
