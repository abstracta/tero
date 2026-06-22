#!/usr/bin/env node
import { program } from 'commander'
import { loginCommand } from './commands/login'
import { createApiKeyCommand } from './commands/createApiKey'
import { askCommand } from './commands/ask'
import { TeroApiError, AuthError } from './core/types'

program
  .name('tero')
  .version(process.env.TERO_CLI_VERSION ?? '0.1.0', '-v, --version')
  .description('Tero CLI - Authenticate and chat with agents from the terminal')

program
  .command('login')
  .description('Interactive browser login')
  .requiredOption('--url <url>', 'Tero API base URL')
  .action(async (opts) => {
    await loginCommand(opts.url)
  })

program
  .command('create-api-key')
  .description('Generate an API key for automation')
  .requiredOption('--name <name>', 'Name for the API key')
  .action(async (opts) => {
    await createApiKeyCommand(opts.name)
  })

program
  .command('ask')
  .description('Send a message to an agent')
  .requiredOption('--agent-id <id>', 'Agent ID to chat with', parseInt)
  .requiredOption('--message <message>', 'Message to send to the agent')
  .option('--output <path>', 'Save reply to file (markdown by default, HTML if .html)')
  .action(async (opts) => {
    await askCommand({ agentId: opts.agentId, message: opts.message, output: opts.output })
  })

async function main(): Promise<void> {
  try {
    await program.parseAsync()
  } catch (err) {
    if (err instanceof TeroApiError) {
      if (err.statusCode === 401) {
        console.error('Authentication failed. Run: tero login')
      } else {
        console.error(`API error (${err.statusCode}): ${err.detail}`)
      }
      process.exit(1)
    }
    if (err instanceof AuthError) {
      console.error(err.message)
      process.exit(1)
    }
    throw err
  }
}

main()
