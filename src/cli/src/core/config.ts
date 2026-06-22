import { readFileSync, writeFileSync, mkdirSync, chmodSync } from 'fs'
import { existsSync } from 'fs'
import { homedir } from 'os'
import { join } from 'path'
import type { CliConfig } from './types'

const CONFIG_DIR = join(homedir(), '.tero')
const CONFIG_FILE = join(CONFIG_DIR, 'config.json')

export function loadConfig(): CliConfig {
  if (!existsSync(CONFIG_FILE)) return {}
  const raw = readFileSync(CONFIG_FILE, 'utf-8')
  return JSON.parse(raw)
}

export function saveConfig(config: CliConfig): void {
  mkdirSync(CONFIG_DIR, { recursive: true })
  const payload = JSON.stringify(config, null, 2)
  writeFileSync(CONFIG_FILE, payload, { encoding: 'utf-8', mode: 0o600 })
  chmodSync(CONFIG_FILE, 0o600)
}
