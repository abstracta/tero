import { HttpError, ApiService } from "@/services/api";
import {
  createAuthFlowHandler,
  OAuthRequest,
  AuthenticationError,
  toolAuthManager,
  type HttpErrorLike
} from "@tero/common/utils/toolAuth.js";

const isHttpError = (e: any): e is HttpErrorLike => {
  return e instanceof HttpError;
};

const createOAuthRequiredFlow = (api: ApiService) => async (oauthRequest: OAuthRequest): Promise<void> => {
  return new Promise((resolve, reject) => {
    const popupWidth = 600;
    const popupHeight = 600;
    const left = window.screenX + (window.outerWidth - popupWidth) / 2;
    const top = window.screenY + (window.outerHeight - popupHeight) / 2;
    const popup = window.open(oauthRequest.url, 'oauth', `popup,width=${popupWidth},height=${popupHeight},left=${left},top=${top}`);
    if (!popup) {
      reject(new AuthenticationError('authenticationWindowBlocked'));
      return;
    }

    // using broadcast channel instead of just using popup and window opener since some mcp servers (like sentry)
    // can set Cross Origin Opener Policy (COOP) which prevents the popup from accessing the opener window
    const channel = new BroadcastChannel('oauth_channel');

    const cleanup = () => {
      toolAuthManager.hideOAuthRequired();
      channel.close();
      channel.removeEventListener('message', handleCallback);
    };

    const handleCancel = () => {
      cleanup();
      popup.close();
      reject(new AuthenticationError('authenticationCancelled'));
    };

    const handleCallback = async (event: MessageEvent) => {
      const data = event.data;
      if (data.type === 'oauth_callback' && data.state === oauthRequest.state) {
        try {
          cleanup();
          if (data.error) {
            await api.deleteToolAuth(data.toolId, data.state);
            reject(new AuthenticationError(data.error == 'access_denied' ? 'authenticationAccessDenied' : 'authenticationUnknownError'));
          } else {
            await api.completeToolOauthAuth(data.toolId, oauthRequest.agentId, data.state, data.code);
            resolve();
          }
        } catch (error) {
          if (error instanceof HttpError && error.status === 400) {
            try {
              const body = JSON.parse(error.body);
              if (body.detail && body.detail === "Authentication cancelled") {
                reject(new AuthenticationError('authenticationCancelled'));
                return;
              }
            } catch (_) {
            }
          }
          reject(error);
        }
      }
    };

    channel.addEventListener('message', handleCallback);

    toolAuthManager.showOAuthRequired(oauthRequest.toolId, handleCancel);
  });
};

export const handleToolAuthRequestsIn = async <T>(
  fn: () => Promise<T>,
  api: ApiService,
  options?: { skipTokenAuth?: boolean }
): Promise<T> => {
  const handler = createAuthFlowHandler({
    isHttpError,
    handleOAuthFlow: createOAuthRequiredFlow(api),
    completeAuthToken: options?.skipTokenAuth
      ? undefined
      : (toolId, agentId, token) => api.completeToolAuthTokenAuth(toolId, agentId, token),
  });
  return handler(fn);
};
