import { ConvexHttpClient } from 'convex/browser'
import { config } from 'dotenv'

let client: ConvexHttpClient | null = null

export const getConvex = (): ConvexHttpClient => {
  if (client) return client

  // Load env files on first access
  config({ path: '.env.local', debug: false })
  config({ path: '.env', debug: false })

  const url = process.env.NEXT_PUBLIC_CONVEX_URL

  if (!url) {
    console.error('Error: NEXT_PUBLIC_CONVEX_URL environment variable is not set')
    console.error('Make sure you have a .env.local file with NEXT_PUBLIC_CONVEX_URL defined.')
    process.exit(1)
  }

  client = new ConvexHttpClient(url)
  return client
}
