import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'fs'
import { homedir } from 'os'
import { join } from 'path'

export interface CliConfig {
  // Config will be populated as needed
}

const CONFIG_DIR = join(homedir(), '.orchestral')
const CONFIG_FILE = join(CONFIG_DIR, 'config.json')

export const ensureConfigDir = () => {
  if (!existsSync(CONFIG_DIR)) {
    mkdirSync(CONFIG_DIR, { recursive: true })
  }
}

export const loadConfig = (): CliConfig => {
  try {
    if (!existsSync(CONFIG_FILE)) return {}
    const content = readFileSync(CONFIG_FILE, 'utf-8')
    return JSON.parse(content)
  } catch {
    return {}
  }
}

export const saveConfig = (config: CliConfig) => {
  ensureConfigDir()
  writeFileSync(CONFIG_FILE, JSON.stringify(config, null, 2))
}
