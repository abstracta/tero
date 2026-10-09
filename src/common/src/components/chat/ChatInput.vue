<script lang="ts">
import { AgentPrompt } from '../../utils/domain'
import ChatAttachments from './ChatAttachments.vue'

export interface AgentChatController {
  supportsStopResponse(): boolean
  findPrompts(): Promise<AgentPrompt[]>
  savePrompt(p: AgentPrompt): Promise<void>
  deletePrompt(id: number): Promise<void>
  supportsFileUpload(): boolean
  supportsTranscriptions(): boolean
  transcribe(blob: Blob): Promise<string>
  handleError(error: unknown): void
}
</script>

<script setup lang="ts">
import { ref, computed, nextTick, onMounted, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { type Textarea }from 'primevue'
import AudioTranscription from './AudioTranscription.vue'
import PromptChip from './PromptChip.vue'
import PromptContentDialog from './PromptContentDialog.vue'
import FileInput from '../common/FileInput.vue'
import { type ErrorMessage } from '../common/ErrorBox.vue'
import { UploadedFile } from '../../utils/domain'
import {
  parsePromptVariables,
  renderPromptTemplate,
  defaultVariableValues,
  type PromptVariable
} from '../../utils/promptVariables'
import EmojiConvertor from 'emoji-js'
import { IconBook2, IconEditCircle, IconPaperclip, IconMicrophone, IconX, IconArrowUp, IconPlayerStopFilled, IconLoader2 } from '@tabler/icons-vue'
import SimpleIcon from '../common/SimpleIcon.vue'

type InsertedPrompt = {
  name: string
  content: string
}

const props = defineProps<{
  modelValue: string
  chat: AgentChatController
  isAnswering?: boolean
  initialFiles?: UploadedFile[]
  maxFileSizeMb?: number
  enablePrompts?: boolean
  borderless?: boolean
  shareablePrompts?: boolean
  readonlyPrompts?: boolean
  insertPromptAsText?: boolean
  beforeSend?: () => Promise<boolean>
}>()

const emit = defineEmits<{
  (e: 'update:modelValue', value: string): void
  (e: 'send', text: string, files: UploadedFile[], collapsedName?: string): void
  (e: 'stop'): void
  (e: 'filesChange', files: UploadedFile[]): void
  (e: 'viewFile', file: UploadedFile): void
}>()

const { t } = useI18n()

// this is a workaround to get the textarea element from the primevue textarea component
type TextareaInstance = InstanceType<typeof Textarea> & { $el: HTMLTextAreaElement }
const inputRef = ref<TextareaInstance | null>(null);
const emoji = new EmojiConvertor()
const lastCursorPosition = ref<number>(0);

const agentPrompts = ref<AgentPrompt[]>([])
const isShowingPrompts = ref(false);
const selectedPromptIndex = ref<number>(0);
const selectedPrompt = ref<AgentPrompt>();
const insertedPrompt = ref<InsertedPrompt>();
const showCollapsedPromptEditor = ref(false);
const showDiscardConfirmation = ref(false);
const collapsedPromptDraft = ref('');
const promptVariables = ref<PromptVariable[]>([]);
const promptVariableValues = ref<Record<string, string>>({});
const editingPrompt = ref<AgentPrompt>();

const MAX_FILES = 5
const fileInputRef = ref<InstanceType<typeof FileInput> | null>(null)
const attachedFiles = ref<UploadedFile[]>(props.initialFiles || [])
const attachedFilesError = ref<ErrorMessage | undefined>(undefined)
const isPreparingSend = ref(false)

const transcriptionRef = ref<InstanceType<typeof AudioTranscription> | null>(null)
const isRecordingAudio = ref<boolean>(false);
const isWaitingTranscript = ref<boolean>(false);
const isCancellingTranscription = ref<boolean>(false);

const inputText = computed({
  get: () => props.modelValue,
  set: v  => emit('update:modelValue', emoji.replace_colons(v))
})

const filteredPrompts = computed<AgentPrompt[]>(() => {
  const filter = inputText.value.startsWith("/") ? inputText.value.substring(1, inputText.value.length).toLocaleLowerCase() : ""
  const promptList = filter != "" ? agentPrompts.value.filter(p => p.name?.toLocaleLowerCase().includes(filter)) : agentPrompts.value
  const starters = promptList.filter(p => p.starter).sort((a, b) => (a.name || '').localeCompare(b.name || ''))
  const prompts = promptList.filter(p => !p.starter).sort((a, b) => (a.name || '').localeCompare(b.name || ''))
  return [...starters, ...prompts]
})

const isTranscribing = computed(() => isRecordingAudio.value || isWaitingTranscript.value)

onMounted(async () => {
  await loadAgentPrompts()
})

const loadAgentPrompts = async () => {
  try {
    agentPrompts.value = await props.chat.findPrompts()
  } catch (error) {
    props.chat.handleError(error)
  }
}

watch(inputText, async() => {
  if (inputText.value.startsWith('/') && props.enablePrompts !== false) {
    if (!isShowingPrompts.value) {
      selectedPromptIndex.value = 0;
      isShowingPrompts.value = true
    }
  } else {
    isShowingPrompts.value = false
  }
  // we need this because if we use border-none in text area then resize stops working
  // and if we use border-transparent then we start seeing a scroll when the text has more than 1 line
  await resetTextareaHeight()
})

const resetTextareaHeight = async () => {
  await nextTick(() => {
    const textareaEl = getTextareaEl()
    if (textareaEl) {
      textareaEl.style.height = 'auto'
      textareaEl.style.height = textareaEl.scrollHeight + 'px'
    }
  })
}

watch(() => props.initialFiles, (newFiles) => {
  attachedFiles.value = newFiles || []
})

const onKeydown = async (e: KeyboardEvent) => {
  if (e.key === 'Enter' && !e.shiftKey && !isShowingPrompts.value) {
    e.preventDefault()
    await sendMessage()
  }
  else if (isShowingPrompts.value) {
    if (e.key === 'Enter') {
      e.preventDefault();
      usePrompt(filteredPrompts.value[selectedPromptIndex.value])
    } else if (e.key === 'Escape') {
      isShowingPrompts.value = false;
    } else if (e.key === 'ArrowUp' && selectedPromptIndex.value > 0) {
      e.preventDefault()
      selectedPromptIndex.value--
    } else if (e.key === 'ArrowDown' && selectedPromptIndex.value < filteredPrompts.value.length - 1) {
      e.preventDefault()
      selectedPromptIndex.value++
    }
  }
}

const onPaste = (event: ClipboardEvent) => {
  if (!props.chat.supportsFileUpload()) return

  const items = event.clipboardData?.items
  if (!items?.length) return

  const files = Array.from(items)
    .filter((item) => item.kind === 'file')
    .map((item) => item.getAsFile())
    .filter((file): file is File => file !== null)

  if (!files.length) return

  event.preventDefault()
  resetAttachedFilesError()
  ;(fileInputRef.value as { addFiles?: (files: File[]) => void } | null)?.addFiles?.(files)
}

const sendMessage = async () => {
  let text = insertedPrompt.value?.content ?? inputText.value
  if (props.isAnswering || isPreparingSend.value || (text.trim() === '' && attachedFiles.value.length === 0)) {
    return
  }
  if (props.beforeSend) {
    isPreparingSend.value = true
    try {
      if (!await props.beforeSend()) return
    } finally {
      isPreparingSend.value = false
    }
    text = insertedPrompt.value?.content ?? inputText.value
    if (text.trim() === '' && attachedFiles.value.length === 0) return
  }
  emit('send', text, attachedFiles.value, insertedPrompt.value?.name)
  inputText.value = ''
  attachedFiles.value = []
  insertedPrompt.value = undefined
  showCollapsedPromptEditor.value = false
}

const usePrompt = async (prompt: AgentPrompt) => {
  const promptText = prompt.content || "";
  const variables = parsePromptVariables(promptText);
  selectedPrompt.value = prompt;
  if (variables.length > 0) {
    promptVariables.value = variables;
    promptVariableValues.value = defaultVariableValues(variables);
  } else {
    await placePrompt(prompt, promptText);
    selectedPrompt.value = undefined;
  }
}

const setCollapsedPrompt = (name: string, content: string) => {
  insertedPrompt.value = { name, content }
  inputText.value = ''
  showCollapsedPromptEditor.value = false
}

const placePrompt = async (prompt: AgentPrompt, content: string) => {
  if (props.insertPromptAsText || !prompt.collapsed) {
    insertedPrompt.value = undefined
    showCollapsedPromptEditor.value = false
    await setInputValue(content);
    return
  }
  setCollapsedPrompt(prompt.name!.trim(), content)
  isShowingPrompts.value = false
}

const openCollapsedPromptEditor = () => {
  collapsedPromptDraft.value = insertedPrompt.value!.content
  showCollapsedPromptEditor.value = true
}

const confirmCollapsedPromptEdit = () => {
  insertedPrompt.value!.content = collapsedPromptDraft.value
  showCollapsedPromptEditor.value = false
}

const requestCloseCollapsedPromptEditor = () => {
  if (collapsedPromptDraft.value !== insertedPrompt.value?.content) {
    showDiscardConfirmation.value = true
    return
  }
  showCollapsedPromptEditor.value = false
}

const discardCollapsedPromptEdit = () => {
  showDiscardConfirmation.value = false
  showCollapsedPromptEditor.value = false
}

const removeInsertedPrompt = () => {
  insertedPrompt.value = undefined
  showCollapsedPromptEditor.value = false
  focus()
}

const setInputValue = async(value: string) => {
  inputText.value = value;
  await focus();
}

const focus = async () => {
  await nextTick(() => {
    inputRef.value?.$el.focus();
  });
}

const selectPrompt = async (prompt: AgentPrompt) => {
  selectedPromptIndex.value = filteredPrompts.value.indexOf(prompt);
  await usePrompt(prompt);
  isShowingPrompts.value = false;
}

const submitPromptVariables = async () => {
  const prompt = selectedPrompt.value;
  const promptText = prompt?.content || "";
  const newPromptText = renderPromptTemplate(promptText, promptVariableValues.value);
  closePromptVariablesEditor();
  if (prompt) {
    await placePrompt(prompt, newPromptText);
  }
}

const closePromptVariablesEditor = () => {
  promptVariables.value = [];
  promptVariableValues.value = {};
  selectedPrompt.value = undefined;
}

const togglePrompts = async () => {
  if (!isShowingPrompts.value && inputText.value == "" && !insertedPrompt.value) {
    inputText.value = "/"
    await focus();
  } else if (!editingPrompt.value) {
    isShowingPrompts.value = !isShowingPrompts.value;
  }
}

const createPromptFromMessage = async (text: string) => {
  editingPrompt.value = new AgentPrompt(undefined, undefined, text)
}

const showCreatePrompt = async () => {
  editingPrompt.value = new AgentPrompt();
}

const savePrompt = async (prompt: AgentPrompt) => {
  try {
    await props.chat.savePrompt(prompt);
    agentPrompts.value = agentPrompts.value.filter(p => p.id !== prompt.id);
    agentPrompts.value.push(prompt);
  } catch (e) {
    props.chat.handleError(e);
  }
}

const showEditPrompt = async (prompt: AgentPrompt) => {
  editingPrompt.value = prompt;
}

const closePromptEditor = async () => {
  editingPrompt.value = undefined;
}

const deletePrompt = async (promptId: number) => {
  try {
    await props.chat.deletePrompt(promptId);
    agentPrompts.value = agentPrompts.value.filter(p => p.id !== promptId);
  } catch (e) {
    props.chat.handleError(e);
  }
}

const startRecordingAudio = async () => {
  isRecordingAudio.value = true
  isWaitingTranscript.value = false
  isCancellingTranscription.value = false
  lastCursorPosition.value = getTextareaEl().selectionStart
}

const getTextareaEl = (): HTMLTextAreaElement => {
  return inputRef.value?.$el as HTMLTextAreaElement
}

const cancelRecordingAudio = async () => {
  isCancellingTranscription.value = true
  isRecordingAudio.value = false
  await focusInputAt(lastCursorPosition.value)
}

const focusInputAt = async (position: number) => {
  await nextTick()
  const textareaEl = getTextareaEl()
  if (!textareaEl) return
  textareaEl.focus()
  textareaEl.selectionStart = textareaEl.selectionEnd = position
}

const stopRecordingAudio = async () => {
  isRecordingAudio.value = false
  isWaitingTranscript.value = true
  await transcriptionRef.value?.stopRecording()
}

const handleAudioReady = async (blob: Blob) => {
  // check in case the trancription was cancelled after stopped
  if (isCancellingTranscription.value) {
    isCancellingTranscription.value = false
    return
  }

  try {
    const text = await props.chat.transcribe(blob)
    isWaitingTranscript.value = false
    // check again in case the trancription was cancelled
    if (isCancellingTranscription.value) {
        isCancellingTranscription.value = false
        return
    }

    const textareaEl = getTextareaEl()
    const trimmed = text.trim()
    if (!trimmed) {
        return
    }

    const start = textareaEl.selectionStart
    const end = textareaEl.selectionEnd
    inputText.value =
        inputText.value.slice(0, start) +
        trimmed +
        inputText.value.slice(end)


    await nextTick()
    await focusInputAt(start + trimmed.length)
  } catch (error) {
    isWaitingTranscript.value = false
    props.chat.handleError(error)
  }
}

const onFilesChange = (files: UploadedFile[]) => {
  attachedFiles.value = [...attachedFiles.value, ...files]
  emit('filesChange', attachedFiles.value)
}

const removeAttachedFile = (index: number) => {
  attachedFiles.value.splice(index, 1)
  if (attachedFiles.value.length < MAX_FILES) {
    resetAttachedFilesError()
  }
  emit('filesChange', attachedFiles.value)
}

const resetAttachedFilesError = () => {
  attachedFilesError.value = undefined
}

const openFileBrowser = () => {
  resetAttachedFilesError()
  fileInputRef.value?.triggerFileInput()
}

defineExpose({
  focus,
  createPromptFromMessage,
  selectPrompt,
  reloadPrompts: loadAgentPrompts,
  setCollapsedPrompt,
  sendMessage
})
</script>

<template>
  <div>
    <div class="flex-shrink-0 bg-surface" :class="!borderless ? 'p-2 sm:p-4' : ''">
      <div class="max-w-[837px] mx-auto w-full">
        <div
          class="flex flex-col gap-1 relative rounded-xl bg-surface"
          :class="!borderless ? 'border focus-within:border-abstracta shadow-sm p-2' : ''">
          <div class="w-full absolute -translate-x-2 -translate-y-full -mt-6">
            <PromptsPanel v-if="isShowingPrompts && !editingPrompt"
              :prompts="filteredPrompts"
              :selected-prompt-index="selectedPromptIndex"
              :readonly="readonlyPrompts"
              @prompt-create="showCreatePrompt"
              @prompt-select="selectPrompt"
              @prompt-edit="showEditPrompt"
              @prompt-delete="deletePrompt" />
          </div>
          <div class="w-full flex flex-col">
            <FileInput
              ref="fileInputRef"
              variant="zone"
              :disabled="!chat.supportsFileUpload()"
              :maxFiles="MAX_FILES"
              :maxFileSizeMb="maxFileSizeMb"
              :attachedFiles="attachedFiles"
              :allowedExtensions="['pdf', 'txt', 'md', 'csv', 'xlsx', 'xls', 'docx', 'png', 'jpg', 'jpeg', 'har', 'json', 'svg']"
              @filesChange="onFilesChange"
              @error="attachedFilesError = $event">
                <div class="w-full">
                  <div class="relative">
                    <!-- Using tailwindcss h-[] class is causing scroll to not working properly, using max-height inline style instead solves this issue -->
                    <Textarea
                      ref="inputRef"
                      v-show="!isTranscribing"
                      v-model="inputText"
                      @keydown="onKeydown"
                      @paste="onPaste"
                      :disabled="!!insertedPrompt"
                      :placeholder="insertedPrompt ? '' : t('placeholderNewMessage')"
                      auto-resize
                      :rows="1"
                      class="w-full !text-base !border-none !shadow-none min-h-[44px]"
                      :style="{ 'max-height': '200px' }" />
                    <span
                      v-if="insertedPrompt && !isTranscribing"
                      class="absolute left-3 top-2 text-sm text-content-muted"
                      v-tooltip.bottom="t('collapsedPromptHint')">
                      {{ t('promptAttached') }}
                    </span>
                  </div>
                  <AudioTranscription
                    ref="transcriptionRef"
                    v-show="isTranscribing"
                    :transcription="isTranscribing"
                    :error-handler="chat.handleError"
                    @audio-ready="handleAudioReady" />
                  <ErrorBox :error="attachedFilesError" />
                  <ChatAttachments
                    variant="input"
                    :attached-files="attachedFiles"
                    @remove-file="removeAttachedFile"
                    @view-file="emit('viewFile', $event)">
                    <template v-if="insertedPrompt" #leading>
                      <PromptChip :name="insertedPrompt.name">
                        <template #actions>
                          <SimpleIcon interactive v-tooltip.bottom="t('editInsertedPrompt')" :icon="IconEditCircle" @click="openCollapsedPromptEditor" />
                          <SimpleIcon interactive v-tooltip.bottom="t('removeInsertedPrompt')" :icon="IconX" class="hover:text-error-alt" @click="removeInsertedPrompt" />
                        </template>
                      </PromptChip>
                    </template>
                  </ChatAttachments>
                </div>
                <div class="flex w-full justify-between">
                  <div class="flex items-center gap-2 mx-2">
                    <SimpleButton
                      v-tooltip.top="MAX_FILES && attachedFiles.length >= MAX_FILES ? t('maxFilesReached', { max: MAX_FILES }) : t('manageFilesTooltip')"
                      :disabled="isTranscribing || attachedFiles.length >= MAX_FILES" v-show="chat.supportsFileUpload()"
                      @click="openFileBrowser">
                        <IconPaperclip/>
                    </SimpleButton>
                    <SimpleButton
                      v-if="enablePrompts"
                      :variant="isShowingPrompts ? 'primary' : undefined"
                      @click="togglePrompts"
                      :class="isShowingPrompts ? 'active' : ''"
                      v-tooltip.top="t('managePromptsTooltip')"
                      :disabled="isTranscribing">
                      <IconBook2/>
                    </SimpleButton>
                  </div>
                  <div class="flex items-center p-1 gap-3">
                    <slot name="rightActions">
                      <div v-if="chat.supportsTranscriptions()">
                        <SimpleButton v-if="isTranscribing" @click="cancelRecordingAudio">
                          <IconX/>
                        </SimpleButton>
                        <SimpleButton v-else @click="startRecordingAudio" v-tooltip.top="t('transcriptionTooltip')" :disabled="!!insertedPrompt">
                          <IconMicrophone/>
                        </SimpleButton>
                      </div>
                      <SimpleButton v-if="isWaitingTranscript" variant="primary" class="chat-primary-button" disabled>
                        <IconLoader2 class="animate-spin"/>
                      </SimpleButton>
                      <SimpleButton v-else-if="isRecordingAudio" @click="stopRecordingAudio" variant="primary" class="chat-primary-button">
                        <IconPlayerStopFilled/>
                      </SimpleButton>
                      <SimpleButton v-else-if="isAnswering" @click="emit('stop')" variant="primary" class="chat-primary-button" :disabled="!chat.supportsStopResponse()">
                        <IconPlayerStopFilled/>
                      </SimpleButton>
                      <SimpleButton v-else variant="primary" @click="sendMessage" class="chat-primary-button">
                        <IconArrowUp/>
                      </SimpleButton>
                    </slot>
                  </div>
                </div>
            </FileInput>
          </div>
        </div>
      </div>
    </div>
    <PromptEditor
      v-if="editingPrompt"
      :editing-prompt="editingPrompt"
      :shareable="shareablePrompts"
      :prompt-saver="savePrompt"
      :error-handler="chat.handleError"
      @close="closePromptEditor"
    />
    <PromptContentDialog :visible="showCollapsedPromptEditor" :title="insertedPrompt?.name" @close="requestCloseCollapsedPromptEditor">
      <Form class="flex flex-col flex-1 min-h-0 gap-3" @submit="confirmCollapsedPromptEdit">
        <div class="form-field flex flex-col flex-1">
          <InteractiveInput id="collapsed-prompt-text" class="flex-1" v-model="collapsedPromptDraft" name="collapsed-prompt-text" :rows="10" />
        </div>
        <div class="flex justify-end shrink-0 mt-4">
          <SimpleButton type="submit" :disabled="!collapsedPromptDraft.trim()" class="px-6" variant="primary" shape="rounded">
            {{ t('confirm') }}
          </SimpleButton>
        </div>
      </Form>
    </PromptContentDialog>
    <Dialog v-model:visible="showDiscardConfirmation"
      :header="t('discardChangesTitle')" :modal="true"
      :draggable="false" :resizable="false" :closable="false" class="max-w-150">
      <div class="flex flex-col gap-5">
        <div class="flex flex-row gap-2 items-start whitespace-pre-line">
          {{ t('discardChangesDescription') }}
        </div>
        <div class="flex flex-row gap-2 justify-end">
          <SimpleButton @click="showDiscardConfirmation = false" shape="square" variant="secondary">{{ t('continueEditing') }}</SimpleButton>
          <SimpleButton @click="discardCollapsedPromptEdit" shape="square" variant="error">{{ t('discardChanges') }}</SimpleButton>
        </div>
      </div>
    </Dialog>
    <Dialog :visible="promptVariables.length > 0"
      :header="t('promptVariablesTitle')" :modal="true"
      :draggable="false" :resizable="false" :closable="true" class="w-full max-w-[600px] m-4"
      @update:visible="closePromptVariablesEditor">
        <Form class="flex flex-col gap-5" @submit="submitPromptVariables">
          <div class="flex flex-col gap-3">
            <div v-for="(variable, index) in promptVariables" :key="variable.name" class="w-full">
              <div class="form-field gap-1 flex flex-col ">
                <label :for="variable.name">{{ variable.name }}</label>
                <Select v-if="variable.input.type === 'select'" v-model="promptVariableValues[variable.name]"
                  :options="variable.input.options" class="w-full"
                  :id="variable.name" :autofocus="index === 0"></Select>
                <!-- Using tailwindcss h-[] class is causing scroll to not working properly, using max-height inline style instead solves this issue -->
                <Textarea v-else v-model="promptVariableValues[variable.name]"
                  :auto-resize="true" :rows="1" class="min-h-[80px]" :style="{ 'max-height': '150px' }"
                  :id="variable.name" :placeholder="t('promptVariablePlaceholder')" :autofocus="index === 0"></Textarea>
              </div>
            </div>
          </div>
          <div class="flex flex-row gap-2 justify-end">
            <SimpleButton type="submit" variant="primary" shape="square">{{ t('confirm') }}</SimpleButton>
          </div>
        </Form>
    </Dialog>
  </div>
</template>

<style>
@import '../../assets/styles.css';

.chat-primary-button {
  @apply h-12 w-12 relative;
}

#collapsed-prompt-text {
  height: 100%;
}

#collapsed-prompt-text textarea {
  min-height: 100%;
}
</style>

<i18n lang="json">
{
  "en":{
    "placeholderNewMessage": "Type your message here...",
    "managePromptsTooltip": "Manage Prompts",
    "promptVariablesTitle": "Set prompt variables",
    "promptVariablePlaceholder": "Set a value for the variable",
    "confirm": "Confirm",
    "transcriptionTooltip": "Transcription",
    "maxFilesReached": "Maximum {max} files allowed",
    "manageFilesTooltip": "Attach files",
    "editInsertedPrompt": "Edit",
    "removeInsertedPrompt": "Remove",
    "promptAttached": "Prompt attached",
    "collapsedPromptHint": "While a collapsed prompt is attached, you can't add more text.",
    "discardChangesTitle": "Discard changes?",
    "discardChangesDescription": "You have unsaved edits to this prompt. If you close now, all changes will be lost.",
    "continueEditing": "Continue editing",
    "discardChanges": "Discard changes"
  },
  "es":{
    "placeholderNewMessage": "Escribe tu mensaje aquí...",
    "managePromptsTooltip": "Administrar Prompts",
    "promptVariablesTitle": "Configura las variables del prompt",
    "promptVariablePlaceholder": "Ingresa un valor para la variable",
    "confirm": "Confirmar",
    "transcriptionTooltip": "Transcripción",
    "maxFilesReached": "Máximo {max} archivos permitidos",
    "manageFilesTooltip": "Adjuntar archivos",
    "editInsertedPrompt": "Editar",
    "removeInsertedPrompt": "Quitar",
    "promptAttached": "Prompt adjunto",
    "collapsedPromptHint": "Mientras haya un prompt colapsado adjunto, no puedes agregar más texto.",
    "discardChangesTitle": "¿Descartar cambios?",
    "discardChangesDescription": "Tienes cambios sin guardar en este prompt. Si cierras ahora, todos los cambios se perderán.",
    "continueEditing": "Seguir editando",
    "discardChanges": "Descartar"
  }
}
</i18n>
