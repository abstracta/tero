<script lang="ts">
import { v4 as uuidv4 } from 'uuid'
import { UploadedFile } from '../../utils/domain'

export interface StatusUpdate {
  action: string
  toolName?: string
  description?: string
  args?: any
  step?: string
  result?: string | string[]
  timestamp: Date
}

export class ChatUiMessage {
  text: string
  files: UploadedFile[]
  isUser: boolean
  isComplete: boolean
  isSuccess: boolean
  // Field used as key for rendering the message in the UI since the id is not always available
  uuid: string
  parent: ChatUiMessage | undefined
  children: ChatUiMessage[] = []
  id?: number
  minutesSaved?: number
  feedbackText?: string
  hasPositiveFeedback?: boolean
  stopped?: boolean
  statusUpdates: StatusUpdate[] = []
  isStatusComplete: boolean = false

  constructor(
    text: string,
    files: UploadedFile[],
    isUser: boolean,
    isComplete: boolean,
    isSuccess: boolean,
    children: ChatUiMessage[] = [],
    parent?: ChatUiMessage,
    id?: number,
    minutesSaved?: number,
    feedbackText?: string,
    hasPositiveFeedback?: boolean,
    stopped?: boolean,
    statusUpdates?: StatusUpdate[]
  ) {
    this.text = text
    this.files = files
    this.isUser = isUser
    this.isComplete = isComplete
    this.isSuccess = isSuccess
    this.uuid = uuidv4()
    this.parent = parent
    this.children = children
    this.id = id
    this.minutesSaved = minutesSaved
    this.feedbackText = feedbackText
    this.hasPositiveFeedback = hasPositiveFeedback
    this.stopped = stopped
    this.statusUpdates = statusUpdates || []
    this.isStatusComplete = (statusUpdates?.length || 0) > 0
  }

  public addChild(child: ChatUiMessage): void {
    child.parent = this
    this.children.push(child)
  }

  public static userMessage(text: string, files: UploadedFile[] = []): ChatUiMessage {
    return new ChatUiMessage(text, files, true, true, true)
  }

  public static agentMessage(text?: string, files: UploadedFile[] = []): ChatUiMessage {
    return new ChatUiMessage(text || '', files, false, text !== undefined, true)
  }

  public static agentErrorMessage(text: string): ChatUiMessage {
    return new ChatUiMessage(text || '', [], false, true, false)
  }

  public static fromJsonObject(obj: { text: string; files: UploadedFile[]; isUser: boolean; isComplete: boolean; isSuccess: boolean }): ChatUiMessage {
    return new ChatUiMessage(obj.text, obj.files, obj.isUser, obj.isComplete, obj.isSuccess)
  }

  public findById(targetId: number): { message: ChatUiMessage; depth: number } | undefined {
    const dfs = (node: ChatUiMessage, depth: number): { message: ChatUiMessage; depth: number } | undefined => {
      if (node.id === targetId) {
        return { message: node, depth }
      }
      for (const child of node.children) {
        const result = dfs(child, depth + 1)
        if (result) {
          return result
        }
      }
      return undefined
    }

    return dfs(this, 0)
  }

  public addStatusUpdate(statusUpdate: StatusUpdate): void {
    this.statusUpdates.push(statusUpdate)
  }

  public completeStatus(): void {
    this.isStatusComplete = true
  }
}
</script>

<script lang="ts" setup>
import { computed, nextTick, ref, onMounted, onBeforeUnmount } from 'vue'
import { escapeHtml } from 'markdown-it/lib/common/utils'
import { useI18n } from 'vue-i18n'
import { IconEditCircle, IconSquareRoundedPlus, IconChevronLeft, IconChevronRight } from '@tabler/icons-vue'
import { initializeResizeObserver, renderMarkDown, initializeCodeActionHandler } from '../../utils/formatter'
import { UserFeedback } from "../../utils/domain";
import ChatInput from './ChatInput.vue';

const props = defineProps<{
  message: ChatUiMessage
  actionsEnabled: boolean
  siblingsCount: number
  selectedIndex: number
  isLastMessage: boolean
  isEditingAgent: boolean
  isFeedbackLoading?: boolean
}>()
const emit = defineEmits<{
  (e: 'promptCreate', text: string): void
  (e: 'editMessage', text: string, files: UploadedFile[]): void
  (e: 'selectMessageBranch', index: number): void
  (e: 'feedbackChange', feedback?: UserFeedback): void
  (e: 'viewFile', file: UploadedFile): void
}>()

const { t } = useI18n()
const messageElement = ref<HTMLElement | null>(null)
const showEditingMessage = ref<boolean>(false)
const editingMessageText = ref<string>('')
const editingMessageTextError = ref<string | null>(null)
const editInputRef = ref<InstanceType<typeof ChatInput>>()
const attachedEditFiles = ref<UploadedFile[]>([])

let observer: (() => void) | null = null

onMounted(() => {
  initializeCodeActionHandler(t)
  if (messageElement.value) {
    observer = initializeResizeObserver(messageElement.value)
  }
})

onBeforeUnmount(() => {
  observer?.()
})

const renderedMsg = computed(() => {
  return renderMarkDown(props.message.text, props.message.isComplete, t, messageElement)
})

const handleEditFileChange = (files: UploadedFile[]) => {
  attachedEditFiles.value = files
}

const handleShowEditMessage = async (text: string) => {
  editingMessageTextError.value = null
  editingMessageText.value = text
  attachedEditFiles.value = [...props.message.files]
  showEditingMessage.value = true
  await nextTick()
  editInputRef.value?.focus()
}

const handleEditMessageCancel = () => {
  showEditingMessage.value = false
  editingMessageText.value = ''
}

const onEditSubmit = () => {
  emit('editMessage', editingMessageText.value, attachedEditFiles.value)
  showEditingMessage.value = false
}

const onKeyDown = (event: KeyboardEvent) => {
  editingMessageTextError.value = null
  if (event.key == 'Escape') {
    handleEditMessageCancel()
  }
}

const handlePreviousMessage = () => {
  if (props.selectedIndex > 0) {
    emit('selectMessageBranch', props.selectedIndex - 1)
  }
}

const handleNextMessage = () => {
  if (props.selectedIndex < props.siblingsCount - 1) {
    emit('selectMessageBranch', props.selectedIndex + 1)
  }
}

const handleRetryMessage = () => {
  if (props.message.parent) {
    emit('editMessage', props.message.parent.text, props.message.parent.files)
  }
}
</script>

<template>
  <div class="p-2 py-3 formatted-text">
    <div v-if="message.isUser" class="flex flex-col items-end gap-2 justify-end">
      <div class="flex gap-4 rounded-xl justify-self-end overflow-hidden" :class="`${showEditingMessage ? 'w-full border-1 py-3 px-4' : 'bg-surface-muted max-w-3/4 p-4'}`">
        <div v-if="!showEditingMessage" class="overflow-x-auto">
          <div class="break-words" v-html="message.text ? escapeHtml(message.text).replace(/\n/g, '<br/>') : ''"></div>
          <div v-if="message.files && message.files.length" class="mt-2">
            <ChatAttachments variant="message" :message-id="message.id" :attached-files="message.files" style="max-width: 80vw" @view-file="emit('viewFile', $event)" />
          </div>
        </div>
        <div v-else class="flex flex-col gap-3 w-full">
          <div class="relative">
            <ChatInput ref="editInputRef" v-model="editingMessageText" :initial-files="attachedEditFiles" @files-change="handleEditFileChange" @keydown="onKeyDown" @send="onEditSubmit"
              :chat="{
                findPrompts: async () => [],
                savePrompt: async () => {},
                deletePrompt: async () => {},
                supportsStopResponse: () => false,
                supportsFileUpload: () => true,
                supportsTranscriptions: () => false,
                transcribe: async () => '',
                handleError: () => {}
              }">
              <template #rightActions>
                <div class="flex gap-3">
                  <SimpleButton @click="handleEditMessageCancel" shape="square" variant="secondary">
                    {{ t('cancelButton') }}
                  </SimpleButton>
                  <SimpleButton @click="onEditSubmit" variant="primary" shape="square" :disabled="!editingMessageText.trim()">
                    {{ t('sendButton') }}
                  </SimpleButton>
                </div>
              </template>
            </ChatInput>
          </div>
        </div>
      </div>
      <div v-if="!showEditingMessage" class="flex gap-2" :class="!actionsEnabled ? 'invisible' : ''">
        <SimpleIcon interactive v-tooltip.bottom="t('editMessageButton')" @click="handleShowEditMessage(message.text!)" :icon="IconEditCircle" />
        <SimpleIcon interactive v-tooltip.bottom="t('createPromptButton')" @click="emit('promptCreate', message.text)" :icon="IconSquareRoundedPlus" />
        <div v-if="siblingsCount > 1" class="flex items-center justify-end">
          <SimpleIcon interactive :icon="IconChevronLeft" :class="selectedIndex == 0 ? '!text-content-muted hover:!text-content-muted !cursor-default' : ''" @click="handlePreviousMessage" />
          <span>{{ selectedIndex + 1 }}/{{ siblingsCount }}</span>
          <SimpleIcon interactive :icon="IconChevronRight" :class="selectedIndex + 1 == siblingsCount ? '!text-content-muted hover:!text-content-muted !cursor-default' : ''" @click="handleNextMessage" />
        </div>
      </div>
    </div>
    <div v-else class="flex flex-col gap-2 max-w-full">
      <ChatStatus
        v-if="message.statusUpdates"
        :status-updates="message.statusUpdates"
        :is-complete="message.isStatusComplete"
      />
      <div v-if="message.text" class="flex gap-4 min-w-[0]">
        <div v-html="renderedMsg" ref="messageElement" class="flex flex-col w-full leading-tight gap-2 overflow-x-auto"></div>
      </div>
      <div v-if="message.files && message.files.length" class="mt-2">
        <ChatAttachments variant="message" :message-id="message.id" :attached-files="message.files" style="max-width: 80vw" @view-file="emit('viewFile', $event)" />
      </div>
      <div v-if="message.stopped" class="bg-surface-muted rounded-xl p-2 my-4 flex items-center justify-between gap-2">
        <span class="text-content-muted text-sm">{{ t('stoppedMessage') }}</span>
        <SimpleButton v-if="isLastMessage" @click="handleRetryMessage" variant="primary" shape="square">
          {{ t('retryButton') }}
        </SimpleButton>
      </div>
      <div class="flex justify-end items-center">
        <ChatMessageResponseFeedback v-if="isLastMessage && !isEditingAgent" :message="message" :actions-enabled="actionsEnabled" :is-feedback-loading="isFeedbackLoading" @feedback-change="emit('feedbackChange', $event)" />
      </div>
    </div>
  </div>
</template>

<i18n lang="json">
{
  "en": {
    "createPromptButton": "Create prompt from message",
    "editMessageButton": "Edit message",
    "sendButton": "Send",
    "cancelButton": "Cancel",
    "copyCodeButton": "Copy",
    "downloadCodeButton": "Download",
    "copiedMessage": "Copied!",
    "stoppedMessage": "The response was interrupted before it finished.",
    "retryButton": "Regenerate"
  },
  "es": {
    "createPromptButton": "Crear prompt a partir de mensaje",
    "editMessageButton": "Editar mensaje",
    "sendButton": "Enviar",
    "cancelButton": "Cancelar",
    "copyCodeButton": "Copiar",
    "downloadCodeButton": "Descargar",
    "copiedMessage": "Copiado!",
    "stoppedMessage": "La generación se interrumpió antes de finalizar.",
    "retryButton": "Generar de nuevo"
  }
}
</i18n>

<style>
@import '@/assets/styles.css';
.p-inputnumber .p-inputtext {
  @apply max-w-full;
}
.p-inputnumber.rounded-xl .p-inputtext {
  @apply !rounded-xl;
}
</style>
<style lang="scss">
@use 'three-dots' with (
  $dot-width: 5px,
  $dot-height: 5px,
  $dot-color: var(--color-primary)
);
</style>
