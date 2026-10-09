<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue';
import { useI18n } from 'vue-i18n';
import ChatInput from '../../../../common/src/components/chat/ChatInput.vue';
import { AgentTestcaseChatUiMessage } from './AgentTestcaseChatMessage.vue';
import { useAgentPromptStore } from '@/composables/useAgentPromptStore';

const props = defineProps<{
    selectedMessage: AgentTestcaseChatUiMessage | undefined
    modelValue: string
    handleError: (error: unknown) => void
    isEditing?: boolean
    agentId?: number
    enablePrompts?: boolean
}>()

const emit = defineEmits<{
    (e: 'update:modelValue', value: string): void
    (e: 'send'): void
    (e: 'cancel'): void
    (e: 'save'): void
}>()

const { t } = useI18n()
const chatInputRef = ref<InstanceType<typeof ChatInput>>()
const { agentsPromptStore, loadAgentPrompts } = useAgentPromptStore()

const inputText = computed({
    get: () => props.modelValue,
    set: (value: string) => emit('update:modelValue', value)
})

const focus = () => {
    chatInputRef.value?.focus()
}

onMounted(async () => {
    if (props.agentId) {
        await loadAgentPrompts(props.agentId)
    }
})

watch(() => props.agentId, async (newAgentId) => {
    if (newAgentId) {
        await loadAgentPrompts(newAgentId)
        await chatInputRef.value?.reloadPrompts()
    }
})

defineExpose({
    focus
})
</script>

<template>
    <div v-if="selectedMessage"
        class="flex flex-col gap-1 p-3 relative rounded-xl border focus-within:border-abstracta bg-surface shadow-sm"
        :class="selectedMessage ? selectedMessage?.isUser ? 'border-primary!' : 'border-info!' : ''">
        <span class="absolute top-[-.85rem] right-20 font-semibold rounded-full px-4 py-1 text-sm z-10"
            :class="{ 'bg-abstracta text-white': selectedMessage?.isUser, 'bg-info text-white': !selectedMessage?.isUser }">
            {{ selectedMessage?.isUser ? t('user') : t('agent') }}</span>
        <ChatInput
            v-model="inputText"
            ref="chatInputRef"
            :chat="{
                supportsStopResponse: () => false,
                findPrompts: async () => enablePrompts ? agentsPromptStore.prompts : [],
                savePrompt: async () => {},
                deletePrompt: async (id: number) => {},
                supportsFileUpload: () => false,
                supportsTranscriptions: () => false,
                transcribe: async (blob: Blob) => '',
                handleError: handleError
            }"
            :enable-prompts="enablePrompts"
            :insert-prompt-as-text="true"
            :readonly-prompts="true"
            :borderless="true"
            @send="isEditing ? $emit('save') : $emit('send')">
            <template v-if="isEditing" #rightActions>
                <div class="flex gap-3">
                    <SimpleButton @click="$emit('cancel')" shape="square" variant="secondary">
                        {{ t('cancelButton') }}
                    </SimpleButton>
                    <SimpleButton 
                        @click="$emit('save')" 
                        shape="square"
                        variant="primary"
                        :class="!selectedMessage?.isUser ? 'bg-info!' : ''"
                        :disabled="!modelValue.trim()">
                        {{ t('sendButton') }}
                    </SimpleButton>
                </div>
            </template>
        </ChatInput>
    </div>
</template>

<i18n lang="json">
{
    "en": {
        "user": "Message from the user",
        "agent": "Expected response",
        "cancelButton": "Cancel",
        "sendButton": "Save"
    },
    "es": {
        "user": "Mensaje del usuario",
        "agent": "Respuesta esperada",
        "cancelButton": "Cancelar",
        "sendButton": "Guardar"
    }
}
</i18n>

