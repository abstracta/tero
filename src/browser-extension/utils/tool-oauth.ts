import { browser } from "wxt/browser";
import { HttpServiceError } from "./http";
import { fetchJson } from "./http";
import { Agent } from "./agent";
import { AuthService } from "./auth";
import { 
  createAuthFlowHandler,
  OAuthRequest, 
  AuthenticationError,
  toolAuthManager,
  type HttpErrorLike 
} from "@tero/common/utils/toolAuth.js";

const isHttpServiceError = (e: any): e is HttpErrorLike => {
  return e instanceof HttpServiceError;
};

const createOAuthExtensionFlow = (agent: Agent, authService?: AuthService) => async (
  oauthRequest: OAuthRequest
): Promise<void> => {
  try {
    const redirectUrl = await openOAuthPopup(oauthRequest);
    await processOAuthCallback(redirectUrl, oauthRequest, agent, authService);
  } catch (error: any) {
    if (error instanceof HttpServiceError && error.status === 400 && error.body) {
      let detail: string | undefined;
      try {
        detail = JSON.parse(error.body).detail;
      } catch(e) { }
      if (detail === "Authentication cancelled") {
        throw new AuthenticationError("authenticationCancelled");
      }
    }
    throw error;
  } finally {
    toolAuthManager.hideOAuthRequired();
  }
};

const openOAuthPopup = (oauthRequest: OAuthRequest): Promise<string> => {
  return new Promise<string>((resolve, reject) => {
    let windowId: number | undefined;

    const handleCancel = () => {
      if (windowId) {
        browser.windows.remove(windowId).catch(() => {});
      }
      reject(new AuthenticationError("authenticationCancelled"));
    };

    toolAuthManager.showOAuthRequired(oauthRequest.toolId, handleCancel);

    browser.windows.create({
      url: oauthRequest.url,
      type: 'popup',
      width: 600,
      height: 600,
    }).then((window) => {
      if (!window?.tabs?.[0]) {
        reject(new Error('Failed to create popup'));
        return;
      }

      windowId = window.id!;
      const tabId = window.tabs[0].id!;

      const cleanup = () => {
        browser.tabs.onUpdated.removeListener(tabUpdateListener);
        browser.windows.onRemoved.removeListener(windowRemovedListener);
        browser.windows.remove(windowId!).catch(() => {});
      };

      const isCallbackUrl = (url: string): boolean => {
        try {
          const parsedUrl = new URL(url);
          return parsedUrl.pathname.match(/\/tools\/[^\/]+\/oauth-callback/) !== null && 
                 (parsedUrl.searchParams.has('code') || parsedUrl.searchParams.has('error'));
        } catch (_) {
          return false;
        }
      };

      const tabUpdateListener = (updatedTabId: number, changeInfo: { url?: string }) => {
        if (updatedTabId !== tabId) return;
        if (changeInfo.url && isCallbackUrl(changeInfo.url)) {
          cleanup();
          resolve(changeInfo.url);
        }
      };

      const windowRemovedListener = (removedWindowId: number) => {
        if (removedWindowId === windowId) {
          cleanup();
          reject(new AuthenticationError("authenticationCancelled"));
        }
      };

      browser.tabs.onUpdated.addListener(tabUpdateListener);
      browser.windows.onRemoved.addListener(windowRemovedListener);
    }).catch(reject);
  });
};

const processOAuthCallback = async (
  redirectUrl: string, 
  oauthRequest: OAuthRequest, 
  agent: Agent,
  authService?: AuthService
): Promise<void> => {
  const url = new URL(redirectUrl);
  const code = url.searchParams.get("code");
  const state = url.searchParams.get("state");
  const error = url.searchParams.get("error");
  const toolId = oauthRequest.toolId;

  if (state !== oauthRequest.state) {
    throw new AuthenticationError("authenticationStateMismatch");
  }

  if (error) {
    if (toolId) {
      await deleteToolAuth(agent, toolId, state!, authService);
    }
    throw new AuthenticationError(error == "access_denied" ? "authenticationAccessDenied" : "authenticationUnknownError");
  }

  if (!code || !toolId) {
    throw new AuthenticationError("authenticationCancelled");
  }

  await completeToolOAuth(agent, toolId, oauthRequest.agentId, state!, code, authService);
};

const completeToolOAuth = async (
  agent: Agent,
  toolId: string,
  agentId: number,
  state: string,
  code: string,
  authService?: AuthService
): Promise<void> => {
  await fetchJson(
    `${agent.url}/api/tools/${toolId}/agents/${agentId}/auth`,
    await Agent.buildHttpRequest("PUT", { state, code }, authService)
  );
};

const completeToolAuthToken = async (
  agent: Agent,
  toolId: string,
  agentId: number,
  authToken: string,
  authService?: AuthService
): Promise<void> => {
  await fetchJson(
    `${agent.url}/api/tools/${toolId}/agents/${agentId}/auth`,
    await Agent.buildHttpRequest("PUT", { auth_token: authToken }, authService)
  );
};

const deleteToolAuth = async (
  agent: Agent,
  toolId: string,
  state: string,
  authService?: AuthService
): Promise<void> => {
  await fetchJson(
    `${agent.url}/api/tools/${toolId}/oauth/${state}`,
    await Agent.buildHttpRequest("DELETE", undefined, authService)
  );
};

export const handleToolAuthRequestsIn = async <T>(
  fn: () => Promise<T>,
  agent: Agent,
  authService?: AuthService
): Promise<T> => {
  const handler = createAuthFlowHandler({
    isHttpError: isHttpServiceError,
    handleOAuthFlow: createOAuthExtensionFlow(agent, authService),
    completeAuthToken: (toolId, agentId, token) => completeToolAuthToken(agent, toolId, agentId, token, authService),
  });
  return handler(fn);
};
