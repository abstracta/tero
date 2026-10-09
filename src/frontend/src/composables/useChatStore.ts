import { reactive } from 'vue'
import { useRouter } from 'vue-router'
import { ApiService, Agent, Thread, AgentNotFoundError } from '@/services/api'
import { useSidebar } from '@/composables/useSidebar'
import { agentsPageSize, useAgentStore } from '@/composables/useAgentStore'
import { useErrorHandler } from '@/composables/useErrorHandler'
import type { ChatUiMessage } from '@tero/common/components/chat/ChatMessage.vue'

export interface ThreadUiState {
  messages: ChatUiMessage[]
  showMessageIndexes: Record<number, number>
  streamingResponse: boolean
}

const maxChats = 20;
export const chatsPerPage = 5;
const chatsStore = reactive({
  chats: [] as Thread[],
  chatsByAgent: {} as Record<number, { chats: Thread[], hasMoreChats: boolean }>,
  currentChat: null as Thread | null,
  threadUiStates: {} as Record<number, ThreadUiState>,
  setChats(chats: Thread[]) {
    this.chats = chats
  },
  addChat(chat: Thread) {
    // the remove is necessary in case we are reusing an empty thread
    this.spliceChat(chat)
    this.chats.unshift(chat)
    if (this.chats.length > maxChats) {
      this.chats.splice(maxChats)
    }
    this.chatsByAgent[chat.agent.id] ??= { chats: [], hasMoreChats: false }
    this.chatsByAgent[chat.agent.id].chats.unshift(chat)
  },
  moveChatToTop(chat: Thread) {
    // we remove it and re add it so it is shown at top of list
    this.spliceChat(chat)
    this.addChat(chat)
  },
  updateChat(chat: Thread) {
    this.spliceChat(chat, chat)
  },
  // a chat is listed both in recent chats and in its agent chats
  spliceChat(chat: Thread, ...replacement: Thread[]) {
    for (const chats of [this.chats, this.chatsByAgent[chat.agent.id]?.chats ?? []]) {
      const index = chats.findIndex((c) => c.id === chat.id)
      if (index !== -1) chats.splice(index, 1, ...replacement)
    }
  }
})

export function useChatStore() {
  const api = new ApiService()
  const router = useRouter()
  const { isGroupedViewEnabled } = useSidebar()
  const { agentsStore, loadAgents } = useAgentStore()
  const { handleError } = useErrorHandler()

  function clearThreadUiState(threadId: number) {
    delete chatsStore.threadUiStates[threadId]
  }

  async function refreshSidebar() {
    return isGroupedViewEnabled.value ? loadGroupedChats() : Promise.all([loadAgents(), loadChats()])
  }

  async function recoverFromAgentNotFound(threadId?: number) {
    if (threadId !== undefined) clearThreadUiState(threadId)
    if (threadId === undefined || chatsStore.currentChat?.id !== threadId) await refreshSidebar()
  }

  async function newChat(agent: Agent) {
    let thread: Thread
    try {
      thread = await api.startThread(agent.id)
    } catch (error) {
      if (!(error instanceof AgentNotFoundError)) throw error
      void recoverFromAgentNotFound().catch(handleError)
      return
    }
    thread.agent = agent
    router.push(`/chat/${thread.id}`)
  }

  async function openChat(threadId: number) {
    router.push(`/chat/${threadId}`)
  }

  async function deleteChat(chat: Thread) {
    await api.deleteThread(chat.id)
    chatsStore.spliceChat(chat)
    clearThreadUiState(chat.id)
    // we just reload entire list in case there are more chats to show and we aren't due to current limit
    if (!isGroupedViewEnabled.value) await loadChats()
    if (router.currentRoute.value.path.startsWith(`/chat/${chat.id}`)) {
      router.push('/')
    }
  }

  async function loadChats() {
    chatsStore.setChats(await api.findLastChats(maxChats, true))
  }

  async function loadGroupedChats(append = false) {
    const agents = await api.findUserAgents('', agentsPageSize + 1, append ? agentsStore.agents.length : 0, chatsPerPage)
    agentsStore.setAgents(agents, append)
    if (!append) chatsStore.chatsByAgent = {}
    for (const agent of agents) {
      if (append && chatsStore.chatsByAgent[agent.id]) continue
      chatsStore.chatsByAgent[agent.id] = { chats: (agent.chats ?? []).map((chat) => ({ ...chat, agent })), hasMoreChats: agent.hasMoreChats ?? false }
    }
  }

  async function loadMoreChats(agentId: number) {
    const group = chatsStore.chatsByAgent[agentId]
    const chats = await api.findLastChats(chatsPerPage + 1, true, agentId, group.chats.length)
    group.chats.push(...chats.slice(0, chatsPerPage).filter((chat) => !group.chats.some((c) => c.id === chat.id)))
    group.hasMoreChats = chats.length > chatsPerPage
  }

  async function updateChat(chat: Thread, moveChatToTop = true) {
    const updatedChat = await api.updateThread(chat)
    if (moveChatToTop) updatedChat.lastMessage = new Date()
    if (chatsStore.currentChat?.id === chat.id) {
      chatsStore.currentChat = updatedChat
    }
    if(moveChatToTop) chatsStore.moveChatToTop(updatedChat)
    else chatsStore.updateChat(updatedChat)
  }

  function setCurrentChat(chat: Thread) {
    chatsStore.currentChat = chat
  }

  function refreshChat(thread: Thread, moveChatToTop = true) {
    if (chatsStore.currentChat?.id === thread.id) {
      chatsStore.currentChat = thread
    }
    if (moveChatToTop) chatsStore.moveChatToTop(thread)
    else chatsStore.updateChat(thread)
  }

  return { chatsStore, loadChats, loadGroupedChats, refreshSidebar, recoverFromAgentNotFound, loadMoreChats, newChat, deleteChat, openChat, updateChat, refreshChat, setCurrentChat, clearThreadUiState }
}
