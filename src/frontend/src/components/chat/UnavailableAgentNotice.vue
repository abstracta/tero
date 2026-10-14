<script setup lang="ts">
import { ref } from 'vue'
import { useResizeObserver, useScroll } from '@vueuse/core'
import { useI18n } from 'vue-i18n'
import { IconBan, IconBook2 } from '@tabler/icons-vue'
import type { AgentPrompt } from '../../../../common/src/utils/domain'
import PromptContentDialog from '../../../../common/src/components/chat/PromptContentDialog.vue'
import GradientBottom from '../common/GradientBottom.vue'
import ReadOnlyPromptCard from './ReadOnlyPromptCard.vue'

withDefaults(defineProps<{ prompts?: AgentPrompt[] }>(), {
  prompts: () => []
})
const { t } = useI18n()
const showPrompts = ref(false)
const scrollContainer = ref<HTMLElement | null>(null)
const promptList = ref<HTMLElement | null>(null)
const { arrivedState, measure } = useScroll(scrollContainer)
useResizeObserver([scrollContainer, promptList], measure)
</script>

<template>
  <div class="border-t p-4">
    <div class="max-w-[837px] mx-auto flex items-center gap-3 rounded-xl bg-surface-muted p-4 text-content-muted">
      <IconBan class="shrink-0" />
      <div class="flex-1">
        <p class="font-medium text-content">{{ t('title') }}</p>
        <p class="text-sm">{{ t('description') }}</p>
      </div>
      <SimpleButton v-if="prompts.length" v-tooltip.top="t('viewPromptsTooltip')" :aria-label="t('viewPromptsTooltip')" @click="showPrompts = true">
        <IconBook2 />
      </SimpleButton>
    </div>
    <PromptContentDialog :visible="showPrompts" :title="t('myPrompts')" @close="showPrompts = false">
      <div class="relative flex-1 min-h-0">
        <div ref="scrollContainer" class="h-full overflow-y-auto">
          <div ref="promptList" class="flex flex-col gap-3 p-1 pr-4">
            <ReadOnlyPromptCard v-for="prompt in prompts" :key="prompt.id" :prompt="prompt" />
          </div>
        </div>
        <GradientBottom v-if="!arrivedState.bottom" class="absolute! pointer-events-none mr-4" />
      </div>
    </PromptContentDialog>
  </div>
</template>

<i18n lang="json">
{
  "en": {
    "title": "This agent is no longer available",
    "description": "Its visibility was changed. You can still review this conversation, but you can't send messages or start new chats.",
    "myPrompts": "My prompts",
    "viewPromptsTooltip": "View my prompts"
  },
  "es": {
    "title": "Este agente ya no está disponible",
    "description": "Se cambió su visibilidad. Podés revisar esta conversación, pero ya no podés enviar mensajes ni iniciar chats nuevos.",
    "myPrompts": "Mis prompts",
    "viewPromptsTooltip": "Ver mis prompts"
  }
}
</i18n>
