<script lang="ts" setup>
import { useTemplateRef } from 'vue'
import { useI18n } from 'vue-i18n'
import SidePanelRight from '@/components/common/SidePanelRight.vue'
import ChatStatusSteps from '@tero/common/components/chat/ChatStatusSteps.vue'
import CopyButton from '@tero/common/components/common/CopyButton.vue'
import type { StatusDetailRef, StatusUpdate } from '@tero/common/components/chat/ChatMessage.vue'
import { useSidePanel } from '@/composables/useSidePanel'
import StatusDetailPanel from './StatusDetailPanel.vue'

export type ThoughtProcessSource = {
  statusUpdates: StatusUpdate[]
  isStatusComplete: boolean
  id?: number
}

const props = defineProps<{
  source: ThoughtProcessSource
}>()

const { t } = useI18n()
const stepsRef = useTemplateRef<{ copyText: string }>('stepsRef')
const { pushPanel } = useSidePanel()

const handleOpenDetail = (detail: StatusDetailRef) => {
  pushPanel(StatusDetailPanel, {
    ...detail,
    messageId: props.source.id,
  })
}
</script>

<template>
  <SidePanelRight :title="t('title')">
    <template #content>
      <ChatStatusSteps
        ref="stepsRef"
        :status-updates="source.statusUpdates"
        :can-fetch-detail="source.id != null && source.isStatusComplete"
        @open-detail="handleOpenDetail"
      />
    </template>
    <template v-if="source.isStatusComplete" #footer>
      <div class="flex items-center justify-end py-4">
        <CopyButton :text="stepsRef?.copyText ?? ''" v-tooltip.bottom="t('copyButton')" />
      </div>
    </template>
  </SidePanelRight>
</template>

<i18n lang="json">
{
  "en": {
    "title": "Thought process",
    "copyButton": "Copy thought process"
  },
  "es": {
    "title": "Proceso de pensamiento",
    "copyButton": "Copiar proceso de pensamiento"
  }
}
</i18n>
