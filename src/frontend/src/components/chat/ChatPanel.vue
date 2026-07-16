<script lang="ts" setup>
import { ref, onMounted, reactive, watch, computed } from 'vue';
import { useRouter } from 'vue-router';
import { useI18n } from 'vue-i18n'
import { ApiService, Agent, ThreadMessageOrigin, ThreadMessage, ThreadMessagePart, HttpError, findManifest, TeamRoleStatus, Role, GLOBAL_TEAM_ID, UserProfile } from '@/services/api';
import { useChatStore } from '@/composables/useChatStore';
import { useAgentStore } from '@/composables/useAgentStore';
import { useBudgetStore } from '@/composables/useBudgetStore';
import { useErrorHandler } from '@/composables/useErrorHandler';
import { ChatUiMessage, type StatusUpdate } from '@tero/common/components/chat/ChatMessage.vue';
import { generateChatHtml, downloadChatHtml } from '@tero/common/utils/chatExport.js';
import { useAgentPromptStore } from '@/composables/useAgentPromptStore';
import ChatPanelHeader from './ChatPanelHeader.vue';
import { handleToolAuthRequestsIn } from '@/services/toolAuth';
import { AuthenticationError } from '@tero/common/utils/toolAuth.js';
import { UserFeedback, AgentPrompt, UploadedFile, FileStatus } from '../../../../common/src/utils/domain';
import ChatInput from '../../../../common/src/components/chat/ChatInput.vue';
import ChatStarters from '../../../../common/src/components/chat/ChatStarters.vue';
import { loadUserProfile } from '@/composables/useUserProfile';

const props = defineProps({
  threadId: {
    type: Number,
    required: true
  },
  editingAgent: {
    type: Boolean,
    default: false
  },
});
const emit = defineEmits(['newChat', 'selectChat']);

const { t } = useI18n();
const router = useRouter();
const api = new ApiService();
const { chatsStore, updateChat, refreshChat, newChat, setCurrentChat } = useChatStore();
const threadUiStates = chatsStore.threadUiStates
const { agentsStore, updateAgent, setCurrentAgent } = useAgentStore();
const { agentsPromptStore, loadAgentPrompts, removePrompt, updatePrompt, newPrompt } = useAgentPromptStore();
const { updateBudget } = useBudgetStore();
const { handleError } = useErrorHandler();

const messages = ref<ChatUiMessage[]>([]);
const chatsContainerRef = ref<HTMLElement | null>(null);
const streamingResponse = computed(() => chat.value ? (threadUiStates[chat.value.id]?.streamingResponse ?? false) : false)
const showMessageIndexes = ref<Record<number, number>>({})
const inputText = ref('');
const chatInputRef = ref<InstanceType<typeof ChatInput>>()

const attachedFiles = ref<UploadedFile[]>([]);
const feedbackLoadingMessageId = ref<number | undefined>(undefined);
const showPastChats = ref<boolean>(false);
const testCaseLoading = ref(false);
const currentUser = ref<UserProfile | undefined>(undefined)
const disablePublishGlobal = ref(false)

const chat = computed(() => chatsStore.currentChat);

const starterPrompts = computed(() =>
  agentsPromptStore.prompts.filter(p => p.starter)
)
const shareablePrompts = computed(() => {
  const agentOnGlobalTeam = agentsStore.currentAgent?.team?.id === GLOBAL_TEAM_ID
  const isOwnerOrEditor = currentUser.value?.teams.some(t =>
    t.status === TeamRoleStatus.ACCEPTED &&
    t.id === GLOBAL_TEAM_ID &&
    (t.role === Role.TEAM_OWNER || t.role === Role.TEAM_EDITOR)
  ) ?? false
  return !disablePublishGlobal.value || !agentOnGlobalTeam || isOwnerOrEditor
})
const contactEmail = ref<string>('');

onMounted(async () => {
  agentsPromptStore.prompts = [];
  const [user, manifest] = await Promise.all([loadUserProfile(), findManifest()])
  currentUser.value = user ?? undefined
  disablePublishGlobal.value = manifest.disablePublishGlobal
  contactEmail.value = manifest.contactEmail
  await loadChatData(props.threadId);
})

const getOrCreateThreadUiState = (threadId: number) => {
  if (!threadUiStates[threadId]) {
    threadUiStates[threadId] = {
      messages: [],
      showMessageIndexes: {},
      streamingResponse: false
    }
  }
  return threadUiStates[threadId]
}

const syncUiFromThreadState = (threadId: number) => {
  const state = getOrCreateThreadUiState(threadId)
  messages.value = state.messages
  showMessageIndexes.value = state.showMessageIndexes
}

const syncThreadStateFromUi = (threadId: number) => {
  const state = getOrCreateThreadUiState(threadId)
  state.messages = messages.value
  state.showMessageIndexes = showMessageIndexes.value
}

const setThreadStreaming = (threadId: number, isStreaming: boolean) => {
  getOrCreateThreadUiState(threadId).streamingResponse = isStreaming
}

const loadChatData = async (threadId: number) => {
  syncUiFromThreadState(threadId)
  try {
    const thread = await api.findThread(threadId);
    setCurrentChat(thread);
    // The agent is already set when editing an agent, this is a solution to avoid agent being empty on some cases
    if(!props.editingAgent) setCurrentAgent(thread.agent);
    if (!threadUiStates[threadId]?.messages.length) {
      messages.value = (await api.findThreadMessages(threadId)).map(thread => mapThreadMessageToChatUi(thread));
    }
    await loadAgentPrompts(thread.agent.id);
    await chatInputRef.value?.reloadPrompts();
    await chatInputRef.value?.focus();
    syncThreadStateFromUi(threadId)
    scrollToLastChatMessage(false);
  } catch (e) {
    handleError(e);
  }
};

function mapThreadMessageToChatUi(
  threadMsg: ThreadMessage,
  parent?: ChatUiMessage
): ChatUiMessage {
  const { text, isSuccess } = mapThreadMessageTextAndStatus(threadMsg)
  const files = threadMsg.files || [] as UploadedFile[]
  const stopped = threadMsg.stopped || threadMsg.text === 'ERROR_INTERRUPTED'
  const uiMsg = new ChatUiMessage(text, files, threadMsg.origin === ThreadMessageOrigin.USER, true, isSuccess, [], parent, threadMsg.id, threadMsg.minutesSaved, threadMsg.feedbackText, threadMsg.hasPositiveFeedback, stopped, threadMsg.statusUpdates);
  for (const childThread of threadMsg.children) {
    const childUi = mapThreadMessageToChatUi(childThread, uiMsg)
    uiMsg.children.push(childUi)
  }
  return uiMsg
}

function mapThreadMessageTextAndStatus(threadMsg: ThreadMessage): { text: string; isSuccess: boolean } {
  if (threadMsg.origin === ThreadMessageOrigin.USER) {
    return { text: threadMsg.text, isSuccess: true }
  }

  if (threadMsg.text === 'ERROR_RECURSION_LIMIT_EXCEEDED') {
    return { text: t('recursionLimitExceeded'), isSuccess: false }
  }

  if (threadMsg.text === 'ERROR_MODEL_RATE_LIMIT') {
    return { text: t('modelRateLimitExceeded'), isSuccess: false }
  }

  if (threadMsg.text === 'ERROR_GENERIC') {
    return { text: t('agentAnswerError', { contactEmail: contactEmail.value }), isSuccess: false }
  }

  if (threadMsg.text === 'ERROR_INTERRUPTED') {
    return { text: '', isSuccess: false }
  }

  return { text: threadMsg.text, isSuccess: true }
}

const scrollToLastChatMessage = (smooth = true) => {
  chatsContainerRef.value?.scrollTo({
    top: chatsContainerRef.value.scrollHeight,
    behavior: smooth ? 'smooth' : 'auto'
  });
}

watch(() => props.threadId, async (newThreadId) => {
  inputText.value = "";
  attachedFiles.value = [];
  agentsPromptStore.prompts = [];
  await loadChatData(newThreadId);
});

watch(messages, ()=>{
  showMessageIndexes.value = buildMessageTreeIndexes(messages.value, showMessageIndexes.value, 0);
})

function buildMessageTreeIndexes(
  rootMessages: ChatUiMessage[],
  showMessageIndexes: Record<number, number>,
  startDepth: number = 0
): Record<number, number> {
  // Remove indexes at depths >= startDepth
  Object.keys(showMessageIndexes)
    .map(k => parseInt(k))
    .filter(depth => depth >= startDepth)
    .forEach(depth => {
      delete showMessageIndexes[depth];
    });

  function traverse(nodes: ChatUiMessage[], depth: number) {
    if (!nodes || nodes.length === 0) {
      return;
    }
    if (showMessageIndexes[depth] === undefined) {
      showMessageIndexes[depth] = nodes.length - 1;
    }

    let idx = showMessageIndexes[depth];
    if (idx < 0 || idx >= nodes.length) {
      idx = nodes.length - 1;
      showMessageIndexes[depth] = idx;
    }
    traverse(nodes[idx].children, depth + 1);
  }

  traverse(rootMessages, 0);
  return showMessageIndexes;
}

const onSendUserMessage = async () => {
  const text = inputText.value;
  const files = attachedFiles.value;
  attachedFiles.value = [];

  const { message } = getSelectedBranch(messages.value, showMessageIndexes.value)!;
  if (
    message?.isSuccess === false &&
    !message.isUser &&
    message.parent?.isUser &&
    message.parent.id != null &&
    !(message.stopped && !message.text)
  ) {
    await handleEditMessage(message.parent.id, text, files);
    return;
  }

  await sendUserMessage(text, files);
};

const sendUserMessage = async (text:string, files: UploadedFile[] = [], editMessageId?: number)=>{
  const activeThreadId = chat.value!.id
  setThreadStreaming(activeThreadId, true)
  const userUIMessage = ChatUiMessage.userMessage(text, files);
  const parentMessageId = appendMessage(userUIMessage, editMessageId)
  const answerMsg = reactive(ChatUiMessage.agentMessage(undefined))
  userUIMessage.addChild(answerMsg)

  const startTime = new Date()
  const initialStatusUpdate: StatusUpdate = {
    action: 'statusProcessing',
    timestamp: startTime
  }
  answerMsg.addStatusUpdate(initialStatusUpdate)

  await updateChat(chat.value!)
  await updateAgent(chat.value!.agent.id)
  try {
    await handleToolAuthRequestsIn(async () => {
      const answer = await api.sendMessage(activeThreadId, text, files, parentMessageId, props.editingAgent);
      scrollToLastChatMessage();
      await processAnswer(answer, answerMsg, userUIMessage, activeThreadId)
    }, api)
  } catch (e) {
    await processAnswerError(e, answerMsg, userUIMessage)
  } finally {
    answerMsg.isComplete = true
    setThreadStreaming(activeThreadId, false)
    delete threadUiStates[activeThreadId]
    await updateBudget()
  }
}

function getSelectedBranch(
  rootMessages: ChatUiMessage[],
  showMessageIndexes: Record<number, number>
): { message?: ChatUiMessage; lastDepth: number } {
  let currentMessages = rootMessages;
  let message: ChatUiMessage | undefined;
  let depth = 0;
  while (
    currentMessages &&
    currentMessages.length > 0 &&
    showMessageIndexes.hasOwnProperty(depth)
  ) {
    const idx = showMessageIndexes[depth];
    if (idx < 0 || idx >= currentMessages.length) {
      break;
    }
    message = currentMessages[idx];
    currentMessages = message.children;
    depth++;
  }
  const lastDepth = message ? depth - 1 : 0;
  return { message, lastDepth };
}

const appendMessage = (message: ChatUiMessage, editMessageId?: number) : number | undefined => {
  if (!messages.value.length) {
    messages.value.push(message);
    showMessageIndexes.value[0] = 0;
    showMessageIndexes.value[1] = 0;
    return undefined;
  } else if (editMessageId != null) {
    const res = findMessageById(editMessageId)!
    if (res.message.parent) {
      res.message.parent?.addChild(message);
      showMessageIndexes.value[res.depth] = res.message.parent!.children.length - 1 || 0;
    } else {
      messages.value.push(message);
      showMessageIndexes.value[res.depth] = messages.value.length - 1;
    }
    return res.message.parent?.id;
  } else {
    const res = getSelectedBranch(messages.value, showMessageIndexes.value)!;
    res.message!.addChild(message);
    showMessageIndexes.value[res.lastDepth + 1] = 0;
    showMessageIndexes.value[res.lastDepth + 2] = 0;
    return res.message!.id;
  }
}

const findMessageById = (id: number) => {
  for (const root of messages.value || []) {
    const result = root.findById(id);
    if (result) {
      return result;
    }
  }
  return undefined;
};

const processAnswer = async (
  answer: AsyncIterable<ThreadMessagePart>,
  answerMsg: ChatUiMessage,
  userUIMessage: ChatUiMessage,
  threadId: number,
) => {
  let buffer = ''
  // Batch token updates every 100ms to avoid re-rendering on every incoming token.
  const intervalId = setInterval(() => { buffer = flushStreamBuffer(buffer, answerMsg) }, 100)

  try {
    for await (const part of answer) {
      if (part.answerText) {
        buffer += part.answerText
      } else if (part.userMessage) {
        userUIMessage.id = part.userMessage.id
        userUIMessage.files = part.userMessage.files || []
      } else if (part.metadata) {
        const lastStatus = answerMsg.statusUpdates[answerMsg.statusUpdates.length - 1]
        lastStatus.timestamp = new Date()
        answerMsg.completeStatus()
        answerMsg.id = part.metadata.answerMessageId
        answerMsg.minutesSaved = part.metadata.minutesSaved
        answerMsg.stopped = part.metadata.stopped
        answerMsg.files = part.metadata.files
        setThreadStreaming(threadId, false)
      } else if (part.messageUpdated) {
        if (answerMsg.id === part.messageUpdated.messageId) {
          answerMsg.minutesSaved = part.messageUpdated.minutesSaved
        }
      } else if (part.threadUpdated) {
        const thread = chatsStore.chats.find((c) => c.id === threadId)
          ?? (chat.value?.id === threadId ? chat.value : undefined)
        if (thread) {
          refreshChat({ ...thread, name: part.threadUpdated.name }, chat.value?.id === threadId)
        }
      } else if (part.status) {
        const statusUpdate: StatusUpdate = {
          action: part.status.action,
          toolName: part.status.toolName,
          description: part.status.description,
          args: part.status.args,
          step: part.status.step,
          result: part.status.result,
          timestamp: new Date()
        }
        answerMsg.addStatusUpdate(statusUpdate)
        if (chat.value?.id === threadId) scrollToLastChatMessage()
      }
    }
  } finally {
    clearInterval(intervalId)
    flushStreamBuffer(buffer, answerMsg)
  }
}

const flushStreamBuffer = (buffer: string, answerMsg: ChatUiMessage): string => {
  if (buffer) answerMsg.text += buffer
  return ''
}

const processAnswerError = async (e: unknown, answerMsg: ChatUiMessage, userUIMessage: ChatUiMessage) => {
  const contactEmail = (await findManifest()).contactEmail
  let text = ""
  if (e instanceof HttpError && e.body === 'recursionLimitExceeded') {
    text = t('recursionLimitExceeded')
  } else if (e instanceof HttpError && e.status === 429 && (e.body?.includes?.('quotaExceeded') || e.message.includes('quotaExceeded'))) {
    text = t('quotaExceeded', { contactEmail })
  } else if (e instanceof HttpError && e.body === 'modelRateLimitExceeded') {
    text = t('modelRateLimitExceeded')
  } else if (e instanceof AuthenticationError) {
    text = t(e.errorCode)
  } else {
    console.error(e)
    text = t('agentAnswerError', { contactEmail })
  }

  if (!answerMsg.text) {
    answerMsg.text += text
    answerMsg.isSuccess = false
    userUIMessage.files = userUIMessage.files.map(file => ({...file, status: file.status == FileStatus.PENDING ? FileStatus.ERROR : file.status}))
  } else {
    answerMsg.children.push(ChatUiMessage.agentErrorMessage(text))
  }
  answerMsg.completeStatus()
}

const savePrompt = async (prompt: AgentPrompt) => {
  if (prompt.id) {
    await updatePrompt(chat.value!.agent.id, prompt.id, prompt);
  } else {
    await newPrompt(chat.value!.agent.id, prompt);
  }
}

const handleFileChange = (files: UploadedFile[]) => {
  attachedFiles.value = files;
};

const stopResponse = async () => {
  try {
    await api.stopMessageResponse(chat.value!.id)
  } catch (e) {
    handleError(e);
  }
}

const handleEditMessage = async (messageId:number, text:string, files: UploadedFile[] = []) => {
  const {message} = findMessageById(messageId)!;
  await sendUserMessage(text, files, message.id);
}

const handleOnSelectMessageBranch = (depth:number, index:number)=>{
  showMessageIndexes.value[depth] = index;
  showMessageIndexes.value = buildMessageTreeIndexes(messages.value, showMessageIndexes.value, depth +1);
}

const handleMessageFeedbackChange = async (messageId: number, feedback?: UserFeedback) => {
  const { message } = findMessageById(messageId)!;
  try {
      if (feedback === undefined) {
        feedbackLoadingMessageId.value = messageId;
      }
      const updatedMessage = await api.updateThreadMessage(chat.value!.id, messageId, { hasPositiveFeedback: feedback?.isPositive,
        minutesSaved: feedback?.minutesSaved, feedbackText: feedback?.text});
      message.hasPositiveFeedback = updatedMessage.hasPositiveFeedback;
      message.minutesSaved = updatedMessage.minutesSaved;
      message.feedbackText = updatedMessage.feedbackText;
  }
  catch (e) {
    handleError(e);
  } finally {
    feedbackLoadingMessageId.value = undefined;
  }
}

const onNewChat = async () => {
  if (props.editingAgent) {
    emit('newChat');
    return
  }
  try {
    await newChat(new Agent(chat.value!.agent.id, chat.value!.agent.name));
  } catch (e) {
    handleError(e);
  }
};

const onShowPastChats = ()=>{
  showPastChats.value = !showPastChats.value;
}

const handleViewFile = (file: UploadedFile) => {
  window.open(`/chat/${chat.value!.id}/files/${file.id}`, '_blank')
}

const handleCreateTestCase = async () => {
  try {
    testCaseLoading.value = true;
    const { message } = getSelectedBranch(messages.value, showMessageIndexes.value);
    const testCase = await api.addTestCase(chat.value!.agent.id, chat.value!.id, message?.id)
    router.push(`/agents/${chat.value!.agent.id}?testcaseId=${testCase.thread.id}`)
  } catch (e) {
    handleError(e)
  } finally {
    testCaseLoading.value = false;
  }
}

const handleExportChat = async () => {
  try {
    const chartImages: string[] = []
    const chartDivs = chatsContainerRef.value?.querySelectorAll('.echarts')
    if (chartDivs?.length) {
      const echarts = await import('echarts')
      chartDivs.forEach(div => {
        const instance = echarts.getInstanceByDom(div as HTMLElement)
        if (instance) chartImages.push(instance.getDataURL({ type: 'png', pixelRatio: 2 }))
      })
    }
    const { message } = getSelectedBranch(messages.value, showMessageIndexes.value)
    const branch: ChatUiMessage[] = []
    for (let m = message; m; m = m.parent) branch.unshift(m)
    const html = generateChatHtml({
      messages: branch,
      agentName: agentsStore.currentAgent?.name,
      chatName: chat.value?.name,
      chartImages
    })
    const pad = (n: number) => String(n).padStart(2, '0');
    const now: Date = new Date();
    const fileName: string = `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}-${pad(now.getHours())}${pad(now.getMinutes())}`;
    downloadChatHtml(html, `${fileName}.html`)
  } catch (e) {
    handleError(e)
  }
}

</script>

<template>
  <FlexCard>
    <template #header>
      <ChatPanelHeader
        v-if="chat && agentsStore.currentAgent"
        :chat="chat"
        :agent="agentsStore.currentAgent"
        :editing-agent="editingAgent"
        :test-case-loading="testCaseLoading"
        :has-messages="messages.length > 0"
        @new-chat="onNewChat"
        @show-past-chats="onShowPastChats"
        @create-test-case="handleCreateTestCase"
        @export-chat="handleExportChat"
      />
    </template>
    <div class="flex-1 flex flex-col min-h-0 relative">
      <ChatSearch v-if="chat && agentsStore.currentAgent && showPastChats"
        :current-chat="chat"
        :agent="agentsStore.currentAgent"
        @close="showPastChats = false"
        @select-chat="emit('selectChat', $event)"
        class="absolute"></ChatSearch>
      <div class="flex-1 overflow-y-auto" ref="chatsContainerRef">
        <div class="max-w-[837px] mx-auto w-full p-2 sm:p-4">
          <div class="flex flex-col">
            <ChatMessagesList
              :is-editing-agent="editingAgent"
              :depth="0"
              :show-message-indexes="showMessageIndexes"
              :messages="messages"
              :actions-enabled="!streamingResponse"
              :feedback-loading-message-id="feedbackLoadingMessageId"
              @prompt-create="chatInputRef?.createPromptFromMessage"
              @edit-message="handleEditMessage"
              @select-message-branch="handleOnSelectMessageBranch"
              @feedback-change="handleMessageFeedbackChange"
              @view-file="handleViewFile"
            />
          </div>
        </div>
        <ChatStarters v-if="messages.length === 0 && starterPrompts.length > 0" :prompts-starters="starterPrompts" :chat-input-ref="chatInputRef!" />
      </div>
      <ChatInput
        v-model="inputText"
        ref="chatInputRef"
        :initial-files="attachedFiles"
        :chat="{
          supportsStopResponse: () => true,
          findPrompts: async () => agentsPromptStore.prompts,
          savePrompt: savePrompt,
          deletePrompt: async (id: number) => await removePrompt(chat!.agent.id, id),
          supportsFileUpload: () => true,
          supportsTranscriptions: () => true,
          transcribe: async (blob: Blob) => await api.transcribeAudio(chat!.id, blob),
          handleError: handleError
        }"
        :is-answering="streamingResponse"
        :enable-prompts="true"
        :shareable-prompts="shareablePrompts"
        @send="onSendUserMessage"
        @files-change="handleFileChange"
        @stop="stopResponse">
      </ChatInput>
    </div>
  </FlexCard>
</template>

<i18n lang="json">
  {
    "en": {
      "authenticationWindowBlocked": "The authentication popup could not be opened. Please check that popups are allowed for this site and try again.",
      "authenticationCancelled": "The authentication was cancelled. Please, try again and complete the authentication to use this agent.",
      "authenticationAccessDenied": "The authentication was denied by the server. Please verify that you actually have the permissions necessary to use it.",
      "agentAnswerError": "I can't help with that message. Edit it or send a new one. If the problem continues, [contact the support team](mailto:{contactEmail}?subject=Question%20issue)",
      "quotaExceeded": "You have reached the monthly usage quota. Contact [support](mailto:{contactEmail}?subject=Tero%20Monthly%20Limit) to increase your monthly quota or wait for the next month.",
      "recursionLimitExceeded": "The step limit for this response was reached. \n Try a shorter task or break your request into smaller parts. You can review the thought process to improve the use of the agent and avoid steps that you identify as unnecessary. \n To increase the limit, go to the agent settings and adjust the thought process steps limit.",
      "modelRateLimitExceeded": "The AI model is currently experiencing high demand. Please try again in a few moments."
    },
    "es": {
      "authenticationWindowBlocked": "No se pudo abrir la ventana de autenticación. Por favor, verifica que las ventanas emergentes estén permitidas para este sitio y vuelve a intentarlo.",
      "authenticationCancelled": "La autenticación fue cancelada. Por favor, inténtelo de nuevo y complete la autenticación para usar este agente o esta herramienta.",
      "authenticationAccessDenied": "La autenticación fue denegada por el servidor. Por favor, verifica que tengas los permisos necesarios para usarlo.",
      "agentAnswerError": "No puedo ayudarte con ese mensaje. Probá editarlo o enviar uno nuevo. Si el problema continúa, podés [contactar al equipo de soporte](mailto:{contactEmail}?subject=Question%20issue)",
      "quotaExceeded": "Ha alcanzado la cuota de uso mensual. Contacte a [soporte](mailto:{contactEmail}?subject=Tero%20Monthly%20Limit) para aumentar su cuota mensual o espere al próximo mes.",
      "recursionLimitExceeded": "Se alcanzó el límite de pasos de esta respuesta. \n Intenta con una tarea más corta o divide la solicitud en partes más pequeñas. Puedes revisar el proceso de pensamiento para mejorar el uso del agente y evitar pasos que identifiques que no sean necesarios. \n Para aumentar el límite, ve a la configuración del agente y ajusta el límite de pasos del proceso de pensamiento.",
      "modelRateLimitExceeded": "El modelo de IA está experimentando alta demanda en este momento. Por favor, intentá de nuevo en unos instantes."
    }
  }
</i18n>
