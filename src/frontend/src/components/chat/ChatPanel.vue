<script lang="ts" setup>
import { ref, onMounted, reactive, watch, computed } from 'vue';
import { useI18n } from 'vue-i18n'
import { ApiService, Agent, ThreadMessageOrigin, ThreadMessage, ThreadMessagePart, HttpError, findManifest, TeamRoleStatus, Role } from '@/services/api';
import { useChatStore } from '@/composables/useChatStore';
import { useAgentStore } from '@/composables/useAgentStore';
import { useBudgetStore } from '@/composables/useBudgetStore';
import { useErrorHandler } from '@/composables/useErrorHandler';
import { ChatUiMessage, type StatusUpdate } from '@tero/common/components/chat/ChatMessage.vue';
import { useAgentPromptStore } from '@/composables/useAgentPromptStore';
import ChatPanelHeader from './ChatPanelHeader.vue';
import { AuthenticationError, handleOAuthRequestsIn } from '@/services/toolOAuth';
import { UserFeedback, AgentPrompt, UploadedFile, FileStatus } from '../../../../common/src/utils/domain';
import ChatInput from '../../../../common/src/components/chat/ChatInput.vue';
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
const api = new ApiService();
const { chatsStore, updateChat, newChat, setCurrentChat } = useChatStore();
const { agentsStore, updateAgent, setCurrentAgent } = useAgentStore();
const { agentsPromptStore, loadAgentPrompts, removePrompt, updatePrompt, newPrompt } = useAgentPromptStore();
const { updateBudget } = useBudgetStore();
const { handleError } = useErrorHandler();

const messages = ref<ChatUiMessage[]>([]);
const chatsContainerRef = ref<HTMLElement | null>(null);
const streamingResponse = ref(false);
const showMessageIndexes = ref<Record<number, number>>({})
const inputText = ref('');
const chatInputRef = ref<InstanceType<typeof ChatInput>>()

const attachedFiles = ref<UploadedFile[]>([]);
const feedbackLoadingMessageId = ref<number | undefined>(undefined);
const showPastChats = ref<boolean>(false);
const shareablePrompts = ref(false)

const chat = computed(() => chatsStore.currentChat);

const starterPrompts = computed(() =>
  agentsPromptStore.prompts.filter(p => p.starter)
)

onMounted(async () => {
  agentsPromptStore.prompts = [];
  await loadChatData(props.threadId);
  const user = await loadUserProfile()
  if (user?.teams.some(t => t.status === TeamRoleStatus.ACCEPTED && t.role === Role.TEAM_OWNER)) {
    shareablePrompts.value = true
  }
})

const loadChatData = async (threadId: number) => {
  try {
    const thread = await api.findThread(threadId);
    setCurrentChat(thread);
    // The agent is already set when editing an agent, this is a solution to avoid agent being empty on some cases
    if(!props.editingAgent) setCurrentAgent(thread.agent);
    messages.value = (await api.findThreadMessages(threadId)).map(thread => mapThreadMessageToChatUi(thread));
    await loadAgentPrompts(thread.agent.id);
    await chatInputRef.value?.focus();
    scrollToLastChatMessage(false);
  } catch (e) {
    handleError(e);
  }
};

function mapThreadMessageToChatUi(
  threadMsg: ThreadMessage,
  parent?: ChatUiMessage
): ChatUiMessage {
  const files = threadMsg.files || [] as UploadedFile[]
  const uiMsg = new ChatUiMessage(threadMsg.text, files, threadMsg.origin === ThreadMessageOrigin.USER, true, true, [], parent, threadMsg.id, threadMsg.minutesSaved, threadMsg.feedbackText, threadMsg.hasPositiveFeedback, threadMsg.stopped, threadMsg.statusUpdates);
  for (const childThread of threadMsg.children) {
    const childUi = mapThreadMessageToChatUi(childThread, uiMsg)
    uiMsg.children.push(childUi)
  }
  return uiMsg
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
  await sendUserMessage(text, files);
};

const sendUserMessage = async (text:string, files: UploadedFile[] = [], editMessageId?: number)=>{
  streamingResponse.value = true

  const {message, lastDepth} = getSelectedBranch(messages.value, showMessageIndexes.value)!;
  if(message?.isSuccess === false) {
    message.parent!.children = message.parent!.children.filter(m => m.id !== message.id) as ChatUiMessage[];
    delete showMessageIndexes.value[lastDepth];
  }

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
    await handleOAuthRequestsIn(async () => {
      const answer = await api.sendMessage(chat.value!.id, text, files, parentMessageId, props.editingAgent);
      scrollToLastChatMessage();
      await processAnswer(answer, answerMsg, userUIMessage)
    }, api)
  } catch (e) {
    await processAnswerError(e, answerMsg, userUIMessage)
  } finally {
    answerMsg.isComplete = true
    streamingResponse.value = false
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

const processAnswer = async (answer: AsyncIterable<ThreadMessagePart>, answerMsg: ChatUiMessage, userUIMessage: ChatUiMessage) => {
  let firstPart = true;
  for await (const part of answer) {
    if (part.answerText) {
      if (messages.value[0].children.length == 1 && firstPart) {
        // If is the first message then the chat name must have updated
        await updateChat(await api.findThread(chat.value!.id))
      }
      firstPart = false
      answerMsg.text += part.answerText
    }
    else if (part.userMessage) {
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
      scrollToLastChatMessage()
    }
  }
}

const processAnswerError = async (e: unknown, answerMsg: ChatUiMessage, userUIMessage: ChatUiMessage) => {
  const contactEmail = (await findManifest()).contactEmail
  let text = ""
  if (e instanceof HttpError && e.status === 429 && e.message.includes('quotaExceeded')) {
    text = t('quotaExceeded', { contactEmail })
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

</script>

<template>
  <FlexCard>
    <template #header>
      <ChatPanelHeader
        v-if="chat && agentsStore.currentAgent"
        :chat="chat"
        :agent="agentsStore.currentAgent"
        :messages="messages"
        :editing-agent="editingAgent"
        @new-chat="onNewChat"
        @show-past-chats="onShowPastChats"
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
        <div v-if="messages.length === 0 && starterPrompts.length > 0" class="flex flex-col gap-8 items-start max-w-[837px] mx-auto">
            <p class="text-left whitespace-pre-line text-lg">
              {{ t('starterText') }}
            </p>
            <div class="flex flex-wrap gap-4 text-light-gray ">
              <div v-for="prompt in starterPrompts" :key="prompt.id">
                <button
                  @click="chatInputRef?.selectPrompt(prompt)"
                  class="w-[185px] h-[94px] rounded-lg border border-auxiliar-gray p-3 text-left hover:border-abstracta shadow flex items-start">
                    <span class="line-clamp-3">
                      {{ prompt.name }}
                    </span>
                </button>
              </div>
            </div>
        </div>
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
      "authenticationWindowClosed": "The authentication window was closed before completing the process. Please, try again and keep the authentication popup window open until it finishes.",
      "authenticationCancelled": "The authentication was cancelled. Please, try again and complete the authentication to use this agent.",
      "authenticationAccessDenied": "The authentication was denied by the MCP server. Please verify that you actually have the permissions necessary to use it.",
      "agentAnswerError": "I am currently unable to complete your request. You can try again and if the issue persists contact [support](mailto:{contactEmail}?subject=Tero%20issue)",
      "quotaExceeded": "You have reached the monthly usage quota. Contact [support](mailto:{contactEmail}?subject=Tero%20Monthly%20Limit) to increase your monthly quota or wait for the next month.",
      "starterText": "Hi! 👋 \n How can I help you?"
    },
    "es": {
      "authenticationWindowClosed": "La ventana de autenticación se cerró antes de completar el proceso. Por favor, inténtelo de nuevo y mantenga abierta la ventana emergente de autenticación hasta que termine.",
      "authenticationCancelled": "La autenticación fue cancelada. Por favor, inténtelo de nuevo y complete la autenticación para usar este agente o esta herramienta.",
      "authenticationAccessDenied": "La autenticación fue denegada por el servidor MCP. Por favor, verifica que tengas los permisos necesarios para usarlo.",
      "agentAnswerError": "Ahora no puedo completar tu pedido. Puedes intentar de nuevo y si el problema persiste contactar a [soporte](mailto:{contactEmail}?subject=Tero%20issue)",
      "quotaExceeded": "Ha alcanzado la cuota de uso mensual. Contacte a [soporte](mailto:{contactEmail}?subject=Tero%20Monthly%20Limit) para aumentar su cuota mensual o espere al próximo mes.",
      "starterText": "Hola! 👋 \n ¿Cómo puedo ayudarte?"
    }
  }
</i18n>
