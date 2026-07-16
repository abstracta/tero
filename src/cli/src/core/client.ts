import { TeroApiError } from './types'

export class TeroClient {
  private baseUrl: string
  private token: string

  constructor(baseUrl: string, token: string) {
    this.baseUrl = baseUrl.replace(/\/$/, '')
    this.token = token
  }

  async get(path: string): Promise<Response> {
    const resp = await fetch(`${this.baseUrl}${path}`, {
      headers: this.headers()
    })
    this.raiseForStatus(resp)
    return resp
  }

  async post(path: string, body?: Record<string, unknown>): Promise<Response> {
    const resp = await fetch(`${this.baseUrl}${path}`, {
      method: 'POST',
      headers: { ...this.headers(), 'Content-Type': 'application/json' },
      body: body ? JSON.stringify(body) : undefined
    })
    this.raiseForStatus(resp)
    return resp
  }

  async streamPost(path: string, body?: Record<string, unknown> | FormData): Promise<Response> {
    const headers = this.headers()
    const fetchBody = body instanceof FormData ? body : body ? JSON.stringify(body) : undefined
    if (!(body instanceof FormData) && body) {
      headers['Content-Type'] = 'application/json'
    }
    const resp = await fetch(`${this.baseUrl}${path}`, {
      method: 'POST',
      headers,
      body: fetchBody
    })
    if (!resp.ok) {
      const text = await resp.text()
      throw new TeroApiError(resp.status, text)
    }
    return resp
  }

  private headers(): Record<string, string> {
    return { Authorization: `Bearer ${this.token}` }
  }

  private raiseForStatus(resp: Response): void {
    if (resp.ok) return
    throw new TeroApiError(resp.status, resp.statusText)
  }
}
