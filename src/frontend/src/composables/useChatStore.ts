import { reactive } from 'vue'
import { useRouter } from 'vue-router'
import { ApiService, Agent, Thread } from '@/services/api'
import type { ChatUiMessage } from '@tero/common/components/chat/ChatMessage.vue'

export interface ThreadUiState {
  messages: ChatUiMessage[]
  showMessageIndexes: Record<number, number>
  streamingResponse: boolean
}

const maxChats = 20;
const chatsStore = reactive({
  chats: [] as Thread[],
  currentChat: null as Thread | null,
  threadUiStates: {} as Record<number, ThreadUiState>,
  setChats(chats: Thread[]) {
    this.chats = chats
  },
  addChat(chat: Thread) {
    // the remove is necessary in case we are reusing an empty thread
    this.removeChat(chat.id)
    this.chats.unshift(chat)
    if (this.chats.length > maxChats) {
      this.chats.splice(maxChats)
    }
  },
  removeChat(threadId: number) {
    this.chats = this.chats.filter((c) => c.id !== threadId)
  },
  moveChatToTop(chat: Thread) {
    // we remove it and re add it so it is shown at top of list
    this.removeChat(chat.id)
    this.addChat(chat)
  },
  updateChat(chat: Thread) {
    const index = this.chats.findIndex((c) => c.id === chat.id)
    if (index === -1) return
    this.chats.splice(index, 1, chat)
  }
})

export function useChatStore() {
  const api = new ApiService()
  const router = useRouter()

  async function newChat(agent: Agent) {
    const thread = await api.startThread(agent.id)
    thread.agent = agent
    router.push(`/chat/${thread.id}`)
  }

  async function openChat(threadId: number) {
    router.push(`/chat/${threadId}`)
  }

  async function deleteChat(threadId: number) {
    await api.deleteThread(threadId)
    delete chatsStore.threadUiStates[threadId]
    // we just reload entire list in case there are more chats to show and we aren't due to current limit
    await loadChats()
    if (router.currentRoute.value.path.startsWith(`/chat/${threadId}`)) {
      router.push('/')
    }
  }

  async function loadChats() {
    chatsStore.setChats(await api.findLastChats(maxChats, true))
  }

  async function updateChat(chat: Thread, moveChatToTop = true) {
    const updatedChat = await api.updateThread(chat)
    updatedChat.lastMessage = new Date()
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

  return { chatsStore, loadChats, newChat, deleteChat, openChat, updateChat, refreshChat, setCurrentChat }
}
