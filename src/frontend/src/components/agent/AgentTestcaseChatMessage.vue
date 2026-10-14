<script lang="ts">
import { v4 as uuidv4 } from 'uuid';
import type { StatusUpdate } from '@tero/common/components/chat/ChatMessage.vue'

export class AgentTestcaseChatUiMessage{
    id?: number
    uuid: string
    text: string
    isUser: boolean
    isPlaceholder: boolean
    isStreaming?: boolean
    statusUpdates: StatusUpdate[] = []
    isStatusComplete: boolean = false

    constructor(text: string, isUser: boolean, isPlaceholder: boolean, id?: number, statusUpdates?: StatusUpdate[]){
        this.uuid = uuidv4()
        this.text = text
        this.isUser = isUser
        this.isPlaceholder = isPlaceholder
        this.id = id
        this.isStreaming = false
        this.statusUpdates = statusUpdates || []
        this.isStatusComplete = (statusUpdates?.length || 0) > 0
    }

    public addStatusUpdate(statusUpdate: StatusUpdate): void {
        this.statusUpdates.push(statusUpdate)
    }

    public completeStatus(): void {
        this.isStatusComplete = true
    }
}

</script>

<script setup lang="ts">
import { computed, ref } from 'vue'
import { IconEditCircle, IconTrash } from '@tabler/icons-vue';
import { escapeHtml } from 'markdown-it/lib/common/utils'
import { useI18n } from 'vue-i18n'
import { renderMarkDown } from '../../../../common/src/utils/formatter'
import { buildAgentResponseCopyHtml } from '../../../../common/src/utils/chatExport'
import CopyButton from '@tero/common/components/common/CopyButton.vue'

const props = defineProps<{
    message: AgentTestcaseChatUiMessage
    isSelected?: boolean
    actionsEnabled?: boolean
    selectable?: boolean
}>()

const emit = defineEmits<{
    (e: 'select', message: AgentTestcaseChatUiMessage): void
    (e: 'delete', message: AgentTestcaseChatUiMessage): void
}>()

const { t } = useI18n()

const messageElement = ref<HTMLElement | null>(null)

const canCopy = computed(() => !props.message.isUser && !props.message.isPlaceholder && !!props.message.text)

const renderedMessage = computed(() => {
    if (props.message.isPlaceholder) return props.message.text ? escapeHtml(props.message.text).replace(/\n/g, '<br/>') : ''
    return renderMarkDown(props.message.text, !props.message.isStreaming, t, messageElement)
})

const handleDeleteMessage = () => {
    emit('delete', props.message)
}

const getCopyHtml = () => buildAgentResponseCopyHtml(props.message.text, messageElement.value)

</script>

<template>
    <div class="p-2 py-3 formatted-text w-full" @click="selectable ? emit('select', message) : null">
        <div class="flex flex-col gap-2 max-w-full" :class="message.isUser ? 'items-end justify-end' : ''">
            <div v-if="message.isUser || message.isPlaceholder"
                class="flex gap-4 rounded-xl border-1 border-transparent"
                :class="{
                    'justify-self-end overflow-hidden bg-surface-muted max-w-3/4 p-4': message.isUser,
                    'p-4 w-full': !message.isUser,
                    'cursor-pointer': selectable,
                    '!border-primary border-pulse-primary': message.isUser && isSelected,
                    'p-4 !border-info border-pulse-info': !message.isUser && isSelected,
                    '!border-content-muted border-dashed': message.isPlaceholder && !message.isUser,
                }">
                <div class="overflow-x-auto w-full">
                    <div class="break-words"
                        :class="message.isPlaceholder ? 'text-content-muted' : 'text-content'"
                        v-html="message.text ? escapeHtml(message.text).replace(/\n/g, '<br/>') : ''"></div>
                </div>
            </div>
            <div v-else-if="message.text" class="flex gap-4 min-w-[0] w-full">
                <div ref="messageElement"
                    class="flex flex-col w-full leading-tight gap-2 overflow-x-auto text-content"
                    v-html="renderedMessage"></div>
            </div>
            <div v-if="canCopy || actionsEnabled" class="flex gap-2" :class="message.isUser ? '' : 'justify-end'">
                <CopyButton
                    v-if="canCopy"
                    :text="message.text"
                    :html="getCopyHtml"
                    v-tooltip.bottom="t('copyResponseButton')"
                />
                <template v-if="actionsEnabled">
                    <SimpleIcon interactive v-tooltip.bottom="t('editMessageButton')" @click.stop="$emit('select', message)"
                        :icon="IconEditCircle" />
                    <SimpleIcon interactive v-tooltip.bottom="t('deleteMessageButton')" @click.stop="handleDeleteMessage"
                        :icon="IconTrash" class="hover:text-error-alt" />
                </template>
            </div>
        </div>
    </div>
</template>

<style scoped>
.border-pulse-primary {
    animation: border-pulse 1.5s ease-in-out infinite;
    color: var(--color-primary);
}

.border-pulse-info {
    animation: border-pulse 1.5s ease-in-out infinite;
    color: var(--color-info);
}
</style>

<i18n lang="json">
{
    "en": {
        "editMessageButton": "Edit message",
        "deleteMessageButton": "Delete message",
        "copyResponseButton": "Copy response",
        "copyCodeButton": "Copy",
        "copiedMessage": "Copied!",
        "downloadCodeButton": "Download"
    },
    "es": {
        "editMessageButton": "Editar mensaje",
        "deleteMessageButton": "Eliminar mensaje",
        "copyResponseButton": "Copiar respuesta",
        "copyCodeButton": "Copiar",
        "copiedMessage": "Copiado!",
        "downloadCodeButton": "Descargar"
    }
}
</i18n>
