import { createServer } from 'http'
import { randomBytes, createHash } from 'crypto'
import { URL, URLSearchParams } from 'url'
import { loadConfig, saveConfig } from './config'
import { AuthError } from './types'
import type { Manifest, OidcEndpoints } from './types'

export class AuthService {
  async resolveUrl(): Promise<string | undefined> {
    const envUrl = process.env.TERO_URL
    if (envUrl) return envUrl
    return loadConfig().url
  }

  async resolveToken(url?: string): Promise<string | undefined> {
    const apiKey = process.env.TERO_API_KEY
    if (apiKey && url) {
      const config = loadConfig()
      if (config.apiKeyToken && config.apiKeyTokenExpiresAt && Date.now() < config.apiKeyTokenExpiresAt) {
        return config.apiKeyToken
      }
      const token = await this.exchangeApiKey(url, apiKey)
      const expiresAt = Date.now() + 30 * 60 * 1000
      saveConfig({ ...config, apiKeyToken: token, apiKeyTokenExpiresAt: expiresAt })
      return token
    }
    return loadConfig().token
  }

  async resolveLoginToken(): Promise<string | undefined> {
    return loadConfig().token
  }

  async browserLogin(teroUrl: string): Promise<string> {
    const manifest = await this.fetchManifest(teroUrl)
    const endpoints = await this.discoverOidcEndpoints(manifest.auth.url)
    const { verifier, challenge } = this.generatePkce()
    const state = randomBytes(16).toString('base64url')

    const callbackPort = 8400
    const redirectUri = `http://localhost:${callbackPort}/callback`

    const params = new URLSearchParams({
      response_type: 'code',
      client_id: manifest.auth.clientId,
      redirect_uri: redirectUri,
      scope: manifest.auth.scope,
      code_challenge: challenge,
      code_challenge_method: 'S256',
      state
    })
    const authorizeUrl = `${endpoints.authorization_endpoint}?${params}`

    const code = await this.waitForCallback(callbackPort, state, authorizeUrl)
    const token = await this.exchangeCode(endpoints.token_endpoint, manifest.auth.clientId, code, redirectUri, verifier)

    saveConfig({ url: teroUrl, token })
    return token
  }

  async exchangeApiKey(url: string, apiKey: string): Promise<string> {
    const resp = await fetch(`${url.replace(/\/$/, '')}/api/api-keys/token`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ api_key: apiKey })
    })
    if (!resp.ok) throw new AuthError(`API key authentication failed (${resp.status})`)
    const data = await resp.json() as { accessToken?: string }
    if (!data.accessToken) throw new AuthError('API key authentication missing accessToken')
    return data.accessToken
  }

  private async fetchManifest(teroUrl: string): Promise<Manifest> {
    const resp = await fetch(`${teroUrl.replace(/\/$/, '')}/manifest.json`)
    if (!resp.ok) throw new AuthError(`Failed to fetch manifest (${resp.status})`)
    return resp.json() as Promise<Manifest>
  }

  private async discoverOidcEndpoints(openidUrl: string): Promise<OidcEndpoints> {
    const resp = await fetch(`${openidUrl.replace(/\/$/, '')}/.well-known/openid-configuration`)
    if (!resp.ok) throw new AuthError(`Failed to fetch OIDC config (${resp.status})`)
    return resp.json() as Promise<OidcEndpoints>
  }

  private generatePkce(): { verifier: string, challenge: string } {
    const verifier = randomBytes(48).toString('base64url')
    const digest = createHash('sha256').update(verifier).digest()
    const challenge = digest.toString('base64url')
    return { verifier, challenge }
  }

  private waitForCallback(port: number, expectedState: string, authorizeUrl: string): Promise<string> {
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        server.close()
        reject(new AuthError('Login failed: timeout waiting for OAuth callback'))
      }, 120_000)

      const server = createServer((req, res) => {
        const url = new URL(req.url!, `http://localhost:${port}`)
        const code = url.searchParams.get('code')
        const receivedState = url.searchParams.get('state')
        const error = url.searchParams.get('error')

        if (error) {
          res.writeHead(400, { 'Content-Type': 'text/html' })
          res.end(`<html><body><h2>Login failed</h2><p>${error}</p></body></html>`)
          clearTimeout(timeout)
          server.close()
          reject(new AuthError(`Login failed: ${error}`))
          return
        }

        if (code && receivedState === expectedState) {
          res.writeHead(200, { 'Content-Type': 'text/html' })
          res.end('<html><body><h2>Login successful!</h2><p>You can close this tab.</p></body></html>')
          clearTimeout(timeout)
          server.close()
          resolve(code)
          return
        }

        res.writeHead(400, { 'Content-Type': 'text/html' })
        res.end('<html><body><h2>Login failed</h2><p>Invalid state</p></body></html>')
        clearTimeout(timeout)
        server.close()
        reject(new AuthError('Login failed: state mismatch'))
      })

      server.listen(port, '127.0.0.1', () => {
        import('child_process').then(({ exec }) => {
          const cmd = process.platform === 'win32' ? `start "" "${authorizeUrl}"`
            : process.platform === 'darwin' ? `open "${authorizeUrl}"`
            : `xdg-open "${authorizeUrl}"`
          exec(cmd)
        })
      })
    })
  }

  private async exchangeCode(tokenEndpoint: string, clientId: string, code: string, redirectUri: string, codeVerifier: string): Promise<string> {
    const resp = await fetch(tokenEndpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams({
        grant_type: 'authorization_code',
        client_id: clientId,
        code,
        redirect_uri: redirectUri,
        code_verifier: codeVerifier
      })
    })
    if (!resp.ok) throw new AuthError(`Token exchange failed (${resp.status})`)
    const data = await resp.json() as { access_token?: string }
    if (!data.access_token) throw new AuthError('Token exchange missing access_token')
    return data.access_token
  }
}
