import { AuthService } from '../core/auth'

export async function loginCommand(url: string): Promise<void> {
  const auth = new AuthService()
  console.log('Opening browser for login...')
  await auth.browserLogin(url)
  console.log('Login successful! Configuration saved to ~/.tero/config.json')
}
