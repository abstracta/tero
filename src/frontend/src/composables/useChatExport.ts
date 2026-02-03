import { ApiService, Thread } from '@/services/api'
import { useErrorHandler } from './useErrorHandler'

export function useChatExport() {
  const api = new ApiService()
  const { handleError } = useErrorHandler()

  const validateMessagesForExport = (messages: any[] | undefined, errorMessage?: string): boolean => {
    if (!messages || messages.length === 0) {
      if (errorMessage) {
        handleError(new Error(errorMessage))
      }
      return false
    }
    return true
  }

  const exportChatAsJson = (thread: Thread, threadName: string, agentName: string, messages: any[]): void => {
    try {
      if (!validateMessagesForExport(messages)) {
        return
      }
      api.exportChatAsJson(thread.id, threadName, agentName, messages)
    } catch (error) {
      handleError(error)
    }
  }

  const exportChatAsMarkdown = (thread: Thread, threadName: string, agentName: string, messages: any[]): void => {
    try {
      if (!validateMessagesForExport(messages)) {
        return
      }
      api.exportChatAsMarkdown(thread.id, threadName, agentName, messages)
    } catch (error) {
      handleError(error)
    }
  }

  return {
    exportChatAsJson,
    exportChatAsMarkdown,
    validateMessagesForExport
  }
}
