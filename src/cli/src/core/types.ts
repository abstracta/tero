export type CliConfig = {
  url?: string
  token?: string
  apiKeyToken?: string
  apiKeyTokenExpiresAt?: number
}

export type ManifestAuth = {
  url: string
  clientId: string
  scope: string
}

export type Manifest = {
  id: string
  contactEmail: string
  auth: ManifestAuth
}

export type OidcEndpoints = {
  authorization_endpoint: string
  token_endpoint: string
}

export class TeroApiError extends Error {
  constructor(public statusCode: number, public detail: string) {
    super(`API error (${statusCode}): ${detail}`)
  }
}

export class AuthError extends Error {
  constructor(message: string) {
    super(message)
  }
}
