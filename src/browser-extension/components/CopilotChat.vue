<script lang="ts" setup>
import { ref, nextTick, watch, computed, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { IconMinus, IconMessage2Plus } from '@tabler/icons-vue'
import { ChatMessage } from '~/utils/tab-state'
import { Agent } from '~/utils/agent'
import ChatInput from '../../common/src/components/chat/ChatInput.vue'
import AgentChatMenu from '../../common/src/components/common/AgentChatMenu.vue'
import { AgentPrompt } from '../../common/src/utils/domain'
import { useToolAuthModal } from '@tero/common/utils/useToolAuthModal.js'

const props = defineProps<{
  messages: ChatMessage[],
  minimized?: boolean,
  agent: Agent,
  audioTranscriber: (blob: Blob) => Promise<string>,
  errorHandler: (error: any) => void
}>()
const emit = defineEmits<{
  (e: 'close'): void,
  (e: 'userMessage', text: string): void,
  (e: 'minimize'): void,
  (e: 'stopResponse'): void,
  (e: 'newChat'): void
}>()

const { t } = useI18n()

const messagesDiv = ref<HTMLDivElement>()
const inputText = ref('');
const chatInputRef = ref<InstanceType<typeof ChatInput>>()

const lastMessage = computed((): ChatMessage => props.messages[props.messages.length - 1])
const allPrompts = ref<AgentPrompt[]>([])
const starterPrompts = computed(() => allPrompts.value.filter(p => p.starter))

const findPrompts = async () => {
  if (!allPrompts.value.length) {
    allPrompts.value = await props.agent.getPrompts()
  }
  return allPrompts.value
}

const { showToolAuthModal, toolAuthType, toolId, submitAuth, closeAuth } = useToolAuthModal()

onMounted(async () => {
  await chatInputRef.value?.focus()
});

watch(props.messages, async () => {
  await adjustMessagesScroll()
})

const adjustMessagesScroll = async () => {
  await nextTick(() => {
    messagesDiv.value!.scrollTop = messagesDiv.value!.scrollHeight
  })
}
</script>

<template>
  <PageOverlay :minimized="minimized">
    <template v-slot:minimizedContent>
      <img :src="agent.logo" class="w-12 h-12 object-contain m-1" />
    </template>
    <template v-slot:headerActions>
      <button>
        <IconMinus v-if="!minimized" @click.stop="$emit('minimize')"/>
      </button>
      <BtnClose v-if="!minimized" @click="$emit('close')"/>
    </template>
    <template v-slot:content>
      <div class="h-full flex flex-col">
        <div class="flex flex-row border-b pb-2 mb-2 items-center">
          <div class="flex flex-row gap-2">
            <img :src="agent.logo" class="w-6 h-6" />
            <span class="text-base">{{ agent.manifest.name! }}</span>
          </div>
          <div class="flex-auto flex text-right items-center justify-end">
            <AgentChatMenu :items="[{
                label: t('newChatTooltip'),
                tablerIcon: IconMessage2Plus,
                command: () => emit('newChat')
              }]" />
          </div>
        </div>
        <ChatStarters v-if="messages.length === 0 && starterPrompts.length > 0" :prompts-starters="starterPrompts" :chat-input-ref="chatInputRef!" />
        <div class="h-full flex flex-col overflow-y-auto mb-4" ref="messagesDiv">
          <Message v-for="message in messages" :text="message.text" :is-user="message.isUser"
            :is-complete="message.isComplete" :is-success="message.isSuccess" :agent="agent"
            @prompt-create="chatInputRef?.createPromptFromMessage"/>
        </div>
        <ChatInput
          v-model="inputText"
          ref="chatInputRef"
          :chat="{
            supportsStopResponse: () => agent.supportsStopResponse(),
            findPrompts: async () => await findPrompts(),
            savePrompt: async (p: AgentPrompt) => await agent.savePrompt(p),
            deletePrompt: async (id: number) => await agent.deletePrompt(id),
            supportsFileUpload: () => false,
            supportsTranscriptions: () => agent.supportsTranscriptions(),
            transcribe: async (blob: Blob) => audioTranscriber(blob),
            handleError: errorHandler }"
          :is-answering="!!(lastMessage && !lastMessage.isComplete)"
          :enable-prompts="true"
          :shareable-prompts="false"
          @send="emit('userMessage', inputText)"
          @stop="emit('stopResponse')"/>
      </div>
    </template>
    <template v-slot:modalsContainer>
      <ToolAuthModal
        v-model:visible="showToolAuthModal"
        :tool-id="toolId"
        :auth-type="toolAuthType"
        @submit="submitAuth"
        @cancel="closeAuth"
      />
    </template>
  </PageOverlay>
</template>

<i18n lang="json">
  {
    "en": {
      "newChatTooltip": "New Chat"
    },
    "es": {
      "newChatTooltip": "Nuevo Chat"
    }
  }
</i18n>
