<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { IconChevronRight } from '@tabler/icons-vue'
import type { StatusUpdate } from './ChatMessage.vue'
import { formatStatus } from '../../utils/formatStatusAction'

const { t } = useI18n()

const props = defineProps<{
  statusUpdates: StatusUpdate[]
  isComplete: boolean
}>()

const emit = defineEmits<{
  (e: 'openDetail'): void
}>()

const showStatus = computed(() => {
  return props.statusUpdates.length > 0
})

const currentStatusText = computed(() => {
  if (!props.statusUpdates?.length) return ''
  return formatStatus(props.statusUpdates[props.statusUpdates.length - 1], t)
})

const thoughtTimeInSeconds = computed(() => {
  if (props.statusUpdates.length === 0) return 0
  const firstUpdate = props.statusUpdates[0]
  const lastUpdate = props.statusUpdates[props.statusUpdates.length - 1]
  if (!firstUpdate.timestamp || !lastUpdate.timestamp) return 0
  const diffMs = lastUpdate.timestamp.getTime() - firstUpdate.timestamp.getTime()
  return Math.round(diffMs / 1000)
})
</script>

<template>
  <div v-if="showStatus" class="status-container mb-2">
    <div class="overflow-hidden border-b">
      <button type="button" class="w-full flex items-center justify-between gap-2 py-2 min-w-0" @click.stop="emit('openDetail')">
        <div v-if="!isComplete" class="flex items-center gap-2 min-w-0 flex-1 text-left">
          <div class="w-3 h-3 p-1 border-2 border-t-transparent rounded-full animate-spin shrink-0"></div>
          <span class="text-sm text-content-muted truncate min-w-0" v-html="currentStatusText"></span>
        </div>
        <div v-else class="flex items-center gap-2 min-w-0 flex-1 text-left">
          <span class="text-sm font-medium text-content-muted truncate">
            {{ thoughtTimeInSeconds > 0 ? t('endMessage', { time: thoughtTimeInSeconds }) : t('thoughtProcessMessage') }}
          </span>
        </div>
        <IconChevronRight :size="20" class="text-content-muted shrink-0" />
      </button>
    </div>
  </div>
</template>

<i18n lang="json">
{
  "en": {
    "statusProcessing": "Processing",
    "statusPreModel": "Thinking",
    "planning": "Planning to run tools",
    "statusExecutingTool": "Executing {'<'}b>{tool}{'<'}/b>",
    "statusExecutedTool": "Tool {'<'}b>{tool}{'<'}/b> execution finished",
    "toolError": "Error in tool {'<'}b>{tool}{'<'}/b>",
    "errorReference": " (ref: {id})",
    "endMessage": "Thought for {time} seconds",
    "retrieving": "Retrieving chunks",
    "retrieved": "Chunks retrieved",
    "analyzing": "Analyzing chunks",
    "analyzed": "Chunks analyzed",
    "groundingResponse": "Validating response",
    "groundedResponse": "Response validated",
    "withParams": " with params {'<'}b>{params}{'<'}/b>",
    "thoughtProcessMessage": "Thought process",
    "delegatingToSubagent": "Delegating to {'<'}b>{agent}{'<'}/b>",
    "subagentFinished": "Subagent {'<'}b>{agent}{'<'}/b> finished"
  },
  "es": {
    "statusProcessing": "Procesando",
    "statusPreModel": "Pensando",
    "planning": "Planificando ejecutar herramientas",
    "statusExecutingTool": "Ejecutando {'<'}b>{tool}{'<'}/b>",
    "statusExecutedTool": "Ejecución de herramienta {'<'}b>{tool}{'<'}/b> finalizada",
    "toolError": "Error en la herramienta {'<'}b>{tool}{'<'}/b>",
    "errorReference": " (ref: {id})",
    "endMessage": "Pensó durante {time} segundos",
    "retrieving": "Recuperando secciones",
    "retrieved": "Secciones recuperadas",
    "analyzing": "Analizando secciones",
    "analyzed": "Secciones analizadas",
    "groundingResponse": "Validando respuesta",
    "groundedResponse": "Respuesta validada",
    "withParams": " con los siguientes parametros {'<'}b>{params}{'<'}/b>",
    "thoughtProcessMessage": "Proceso de pensamiento",
    "delegatingToSubagent": "Delegando a {'<'}b>{agent}{'<'}/b>",
    "subagentFinished": "Subagente {'<'}b>{agent}{'<'}/b> finalizó"
  }
}
</i18n>
