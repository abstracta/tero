import { ref, onMounted, onBeforeUnmount } from 'vue'
import {
  toolAuthManager,
  ToolAuthType,
  type AuthTokenRequestData,
  type AuthTokenResult,
  type ToolAuthCallbacks
} from './toolAuth.js'

export function useToolAuthModal() {
  const showToolAuthModal = ref(false)
  const toolAuthType = ref<ToolAuthType>(ToolAuthType.TOKEN)
  const toolId = ref('')

  let authTokenResolve: ((result: AuthTokenResult) => void) | null = null
  let oauthCancelCallback: (() => void) | null = null

  const handleAuthTokenRequest = async (request: AuthTokenRequestData): Promise<AuthTokenResult> => {
    return new Promise((resolve) => {
      authTokenResolve = resolve
      toolId.value = request.toolId
      toolAuthType.value = ToolAuthType.TOKEN
      showToolAuthModal.value = true
    })
  }

  const handleOAuthShow = (oauthToolId: string, onCancel: () => void): void => {
    toolId.value = oauthToolId
    oauthCancelCallback = onCancel
    toolAuthType.value = ToolAuthType.OAUTH
    showToolAuthModal.value = true
  }

  const handleOAuthHide = (): void => {
    showToolAuthModal.value = false
    oauthCancelCallback = null
  }

  const submitAuth = (token: string) => {
    showToolAuthModal.value = false
    if (authTokenResolve) {
      authTokenResolve({ success: true, token })
      authTokenResolve = null
    }
  }

  const closeAuth = () => {
    showToolAuthModal.value = false

    if (authTokenResolve) {
      authTokenResolve({ success: false, cancelled: true })
      authTokenResolve = null
    }

    if (oauthCancelCallback) {
      oauthCancelCallback()
      oauthCancelCallback = null
    }
  }

  onMounted(() => {
    const callbacks: ToolAuthCallbacks = {
      onShowOAuth: handleOAuthShow,
      onHideOAuth: handleOAuthHide,
      onRequestToken: handleAuthTokenRequest
    }
    toolAuthManager.register(callbacks)
  })

  onBeforeUnmount(() => {
    toolAuthManager.unregister()
  })

  return {
    showToolAuthModal,
    toolAuthType,
    toolId,
    submitAuth,
    closeAuth
  }
}
