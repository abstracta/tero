<script lang="ts">
import { v4 as uuidv4 } from 'uuid'
import { UploadedFile } from '../../utils/domain'

export interface StatusResultItem {
  content?: string
  fileName?: string
  title?: string
  path?: string
}

export interface StatusDetailRef {
  path: string
  title: string
  content: string
}

export interface StatusUpdate {
  action: string
  toolName?: string
  agentName?: string
  description?: string
  args?: any
  argsPath?: string
  step?: string
  result?: string | Array<string | StatusResultItem>
  errorId?: string
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
  collapsedName?: string

  constructor(
    text: string,
    files: UploadedFile[],
    isUser: boolean,
    isComplete: boolean,
    isSuccess: boolean,
    collapsedName?: string,
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
    this.collapsedName = collapsedName
  }

  public addChild(child: ChatUiMessage): void {
    child.parent = this
    this.children.push(child)
  }

  public static userMessage(text: string, files: UploadedFile[] = [], collapsedName?: string): ChatUiMessage {
    return new ChatUiMessage(text, files, true, true, true, collapsedName)
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
import { buildAgentResponseCopyHtml } from '../../utils/chatExport'
import { UserFeedback } from "../../utils/domain";
import ChatInput from './ChatInput.vue';
import PromptChip from './PromptChip.vue';
import PromptContentDialog from './PromptContentDialog.vue';
import CopyButton from '../common/CopyButton.vue';

const props = defineProps<{
  message: ChatUiMessage
  actionsEnabled: boolean
  siblingsCount: number
  selectedIndex: number
  isLastMessage: boolean
  isEditingAgent: boolean
  readOnly?: boolean
  beforeSend?: () => Promise<boolean>
  isFeedbackLoading?: boolean
  maxFileSizeMb?: number
}>()
const emit = defineEmits<{
  (e: 'promptCreate', text: string): void
  (e: 'editMessage', text: string, files: UploadedFile[], collapsedName?: string): void
  (e: 'selectMessageBranch', index: number): void
  (e: 'feedbackChange', feedback?: UserFeedback): void
  (e: 'viewFile', file: UploadedFile): void
  (e: 'openDetail', message: ChatUiMessage): void
}>()

const { t } = useI18n()
const messageElement = ref<HTMLElement | null>(null)
const showEditingMessage = ref<boolean>(false)
const editingMessageText = ref<string>('')
const editingMessageTextError = ref<string | null>(null)
const editInputRef = ref<InstanceType<typeof ChatInput>>()
const showCollapsedMessage = ref(false)
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
  editingMessageText.value = props.message.collapsedName ? '' : text
  attachedEditFiles.value = [...props.message.files]
  showEditingMessage.value = true
  showCollapsedMessage.value = false
  await nextTick()
  if (props.message.collapsedName) {
    editInputRef.value?.setCollapsedPrompt(props.message.collapsedName, text)
  } else {
    editInputRef.value?.focus()
  }
}

const handleEditMessageCancel = () => {
  showEditingMessage.value = false
  editingMessageText.value = ''
}

const onEditSubmit = (text: string, files: UploadedFile[], collapsedName?: string) => {
  emit('editMessage', text, files, collapsedName)
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
    emit('editMessage', props.message.parent.text, props.message.parent.files, props.message.parent.collapsedName)
  }
}

const getCopyHtml = () => buildAgentResponseCopyHtml(props.message.text, messageElement.value)
</script>

<template>
  <div class="p-2 py-3 formatted-text">
    <div v-if="message.isUser" class="flex flex-col items-end gap-2 justify-end">
      <ChatAttachments
        v-if="message.collapsedName && !showEditingMessage"
        class="w-full max-w-[80vw] justify-end"
        variant="message"
        :attached-files="message.files"
        @view-file="emit('viewFile', $event)">
        <template #leading>
          <PromptChip
            :name="message.collapsedName"
            selectable
            @select="showCollapsedMessage = true" />
        </template>
      </ChatAttachments>
      <div v-if="!message.collapsedName || showEditingMessage" class="flex gap-4 rounded-xl justify-self-end overflow-hidden" :class="`${showEditingMessage ? 'w-full border-1 py-3 px-4' : 'bg-surface-muted max-w-3/4 p-4'}`">
        <div v-if="!showEditingMessage" class="overflow-x-auto">
          <div class="break-words" v-html="message.text ? escapeHtml(message.text).replace(/\n/g, '<br/>') : ''"></div>
          <div v-if="message.files && message.files.length" class="mt-2">
            <ChatAttachments variant="message" :message-id="message.id" :attached-files="message.files" style="max-width: 80vw" @view-file="emit('viewFile', $event)" />
          </div>
        </div>
        <div v-else class="flex flex-col gap-3 w-full">
          <div class="relative">
            <ChatInput ref="editInputRef" v-model="editingMessageText" :initial-files="attachedEditFiles" :max-file-size-mb="maxFileSizeMb" :before-send="beforeSend" @files-change="handleEditFileChange" @keydown="onKeyDown" @send="onEditSubmit"
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
                  <SimpleButton @click="editInputRef?.sendMessage()" variant="primary" shape="square" :disabled="!message.collapsedName && !editingMessageText.trim()">
                    {{ t('sendButton') }}
                  </SimpleButton>
                </div>
              </template>
            </ChatInput>
          </div>
        </div>
      </div>
      <div v-if="!showEditingMessage && (!readOnly || siblingsCount > 1)" class="flex gap-2" :class="!actionsEnabled ? 'invisible' : ''">
        <SimpleIcon v-if="!readOnly" interactive v-tooltip.bottom="t('editMessageButton')" @click="handleShowEditMessage(message.text!)" :icon="IconEditCircle" />
        <SimpleIcon v-if="!readOnly" interactive v-tooltip.bottom="t('createPromptButton')" @click="emit('promptCreate', message.text)" :icon="IconSquareRoundedPlus" />
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
        @open-detail="emit('openDetail', message)"
      />
      <div v-if="message.text" class="flex gap-4 min-w-[0]">
        <div v-html="renderedMsg" ref="messageElement" class="flex flex-col w-full leading-tight gap-2 overflow-x-auto"></div>
      </div>
      <div v-if="message.files && message.files.length" class="mt-2">
        <ChatAttachments variant="message" :message-id="message.id" :attached-files="message.files" style="max-width: 80vw" @view-file="emit('viewFile', $event)" />
      </div>
      <div v-if="message.stopped" class="bg-surface-muted rounded-xl p-2 my-4 flex items-center justify-between gap-2">
        <span class="text-content-muted text-sm">{{ t('stoppedMessage') }}</span>
        <SimpleButton v-if="isLastMessage && !readOnly" @click="handleRetryMessage" variant="primary" shape="square">
          {{ t('retryButton') }}
        </SimpleButton>
      </div>
      <div v-if="message.text || (isLastMessage && !isEditingAgent)" class="flex items-center justify-end gap-2 w-full">
        <ChatMessageResponseFeedback v-if="isLastMessage && !isEditingAgent" :message="message" :actions-enabled="actionsEnabled" :is-feedback-loading="isFeedbackLoading" @feedback-change="emit('feedbackChange', $event)" />
        <CopyButton
          v-if="message.text"
          :text="message.text"
          :html="getCopyHtml"
          v-tooltip.bottom="t('copyResponseButton')"
        />
      </div>
    </div>
    <PromptContentDialog :visible="showCollapsedMessage" :title="message.collapsedName" @close="showCollapsedMessage = false">
      <div class="min-h-0 flex-1 overflow-y-auto whitespace-pre-wrap break-words rounded-xl bg-surface p-4 text-sm outline-1 outline-auxiliar-gray">{{ message.text }}</div>
    </PromptContentDialog>
  </div>
</template>

<i18n lang="json">
{
  "en": {
    "createPromptButton": "Create prompt from message",
    "copyResponseButton": "Copy response",
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
    "copyResponseButton": "Copiar respuesta",
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
