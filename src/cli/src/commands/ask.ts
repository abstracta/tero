import { writeFileSync, mkdirSync } from 'fs'
import { dirname, extname, resolve } from 'path'
import { homedir } from 'os'
import { AuthService } from '../core/auth'
import { TeroClient } from '../core/client'
import { renderMarkdown } from '../core/formatter'
import { TeroApiError } from '../core/types'

type AskOptions = {
  agentId: number
  message: string
  output?: string
}

type SseEvent = {
  type: string | null
  data: string
}

export async function askCommand(options: AskOptions): Promise<void> {
  const auth = new AuthService()
  const url = await auth.resolveUrl()
  const token = await auth.resolveToken(url)

  if (!url || !token) {
    console.error('No URL or token configured. Set TERO_URL or run: tero login')
    process.exit(1)
  }

  const client = new TeroClient(url, token)

  const threadResp = await client.post('/api/threads', { agentId: options.agentId })
  const { id: threadId } = await threadResp.json() as { id: number }

  const form = new FormData()
  form.append('text', options.message)
  form.append('origin', 'USER')
  const response = await client.streamPost(`/api/threads/${threadId}/messages`, form)

  const chunks: string[] = []

  for await (const event of iterSseEvents(response)) {
    if (event.type === null || event.type === 'message') {
      chunks.push(event.data)
      if (!options.output) {
        process.stdout.write(event.data)
      }
    } else if (event.type === 'error') {
      throw new TeroApiError(500, `Agent error: ${event.data || 'unknown'}`)
    }
  }

  const answer = chunks.join('')

  if (!options.output) {
    process.stdout.write('\n')
  }

  if (options.output) {
    const outputPath = options.output.startsWith('~') ? options.output.replace('~', homedir()) : resolve(options.output)
    mkdirSync(dirname(outputPath), { recursive: true })
    if (extname(outputPath).toLowerCase() === '.html') {
      const html = await renderMarkdown(answer)
      writeFileSync(outputPath, html, 'utf-8')
    } else {
      writeFileSync(outputPath, answer, 'utf-8')
    }
    console.log(`Saved reply to ${outputPath}`)
  }
}

async function* iterSseEvents(response: Response): AsyncGenerator<SseEvent> {
  const reader = response.body!.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const blocks = buffer.split('\r\n\r\n')
    buffer = blocks.pop()!

    for (const block of blocks) {
      if (!block.trim()) continue
      yield parseEventBlock(block)
    }
  }

  if (buffer.trim()) {
    yield parseEventBlock(buffer)
  }
}

function parseEventBlock(block: string): SseEvent {
  const lines = block.split('\r\n')
  let type: string | null = null
  const dataLines: string[] = []

  for (const line of lines) {
    if (line.startsWith('event: ')) {
      type = line.slice(7)
    } else if (line.startsWith('data: ')) {
      dataLines.push(line.slice(6))
    } else if (line.startsWith('data:')) {
      dataLines.push(line.slice(5))
    }
  }

  return { type, data: dataLines.join('\n') }
}
