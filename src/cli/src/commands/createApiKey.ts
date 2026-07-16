import { AuthService } from '../core/auth'
import { TeroClient } from '../core/client'

export async function createApiKeyCommand(name: string): Promise<void> {
  const auth = new AuthService()
  const url = await auth.resolveUrl()
  const token = await auth.resolveLoginToken()

  if (!url || !token) {
    console.error('No URL or login token configured. Run: tero login')
    process.exit(1)
  }

  const client = new TeroClient(url, token)
  const resp = await client.post('/api/api-keys', { name })
  const data = await resp.json() as { apiKey: string }
  console.log(JSON.stringify(data, null, 2))
}
