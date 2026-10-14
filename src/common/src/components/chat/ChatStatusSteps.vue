<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { IconPointFilled } from '@tabler/icons-vue'
import type { StatusResultItem, StatusDetailRef, StatusUpdate } from './ChatMessage.vue'
import { formatStatus, formatThoughtProcessText, resultContent, resultPreview } from '../../utils/formatStatusAction'

const { t } = useI18n()

const props = defineProps<{
  statusUpdates: StatusUpdate[]
  canFetchDetail?: boolean
}>()

const emit = defineEmits<{
  (e: 'openDetail', detail: StatusDetailRef): void
}>()

const copyText = computed(() => formatThoughtProcessText(props.statusUpdates, t))

const hasDetailPath = (item: string | StatusResultItem): item is StatusResultItem => {
  return typeof item === 'object' && item !== null && typeof item.path === 'string' && item.path.length > 0
}

const storedEventIndex = (index: number) =>
  props.statusUpdates[0]?.action === 'statusProcessing' ? index - 1 : index

const openDetail = (index: number, path: string, title: string, content: string) => {
  if (!props.canFetchDetail) return
  emit('openDetail', {
    path: `${storedEventIndex(index)}/${path}`,
    title,
    content,
  })
}

const emitDetail = (item: string | StatusResultItem, index: number) => {
  if (hasDetailPath(item) && item.path) {
    openDetail(index, item.path, item.title || item.fileName || props.statusUpdates[index]?.toolName || '', resultContent(item))
  }
}

defineExpose({
  get copyText() {
    return copyText.value
  }
})
</script>

<template>
  <div class="flex flex-col">
    <div
      v-for="(status, index) in statusUpdates"
      :key="index"
      class="flex items-stretch gap-1 text-sm animate-fade-in opacity-0"
      :style="{ animationDelay: `${index * 0.1}s` }"
    >
      <div class="flex flex-col items-center">
        <IconPointFilled class="text-content-muted" size="16" />
        <span v-if="index !== statusUpdates.length - 1" class="w-[2px] flex-1 bg-auxiliar-gray"></span>
      </div>
      <div class="min-w-0 pb-3 wrap-break-word">
        <div class="flex items-center gap-2 left-[-2px]">
          <span><span v-html="formatStatus(status, t)"></span><button
            v-if="status.action === 'executingTool' && status.argsPath"
            type="button"
            class="ml-1 text-sm underline underline-offset-2"
            :class="canFetchDetail ? 'text-content-muted hover:text-abstracta cursor-pointer' : 'text-content-muted/50 cursor-not-allowed'"
            :disabled="!canFetchDetail"
            v-tooltip.top="canFetchDetail ? undefined : t('seeMoreUnavailable')"
            @click.stop="openDetail(index, status.argsPath, status.toolName || '', typeof status.args === 'string' ? status.args : '')"
          >{{ t('seeMore') }}</button></span>
        </div>
        <div v-if="status.description && status.description.trim() !== '' && status.toolName !== 'task'">
          <span>{{ t('description') }}: </span> <span class="italic">{{ status.description }}</span>
        </div>
        <div v-if="status.result && (typeof status.result === 'string' ? status.result.trim() !== '' : false)" class="whitespace-pre-wrap">
          {{ t('result') }} {{ status.result }}
        </div>
        <div v-if="status.result && Array.isArray(status.result) && status.result.length > 0">
          <div v-for="(item, itemIndex) in status.result" :key="itemIndex" class="text-sm wrap-break-word">
            -<b>{{ resultPreview(item) }}</b><button
              v-if="hasDetailPath(item)"
              type="button"
              class="ml-1 text-sm underline underline-offset-2"
              :class="canFetchDetail ? 'text-content-muted hover:text-abstracta cursor-pointer' : 'text-content-muted/50 cursor-not-allowed'"
              :disabled="!canFetchDetail"
              v-tooltip.top="canFetchDetail ? undefined : t('seeMoreUnavailable')"
              @click.stop="emitDetail(item, index)"
            >{{ t('seeMore') }}</button>
          </div>
        </div>
      </div>
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
    "result": "Result:",
    "description": "Description",
    "retrieving": "Retrieving chunks",
    "retrieved": "Chunks retrieved",
    "analyzing": "Analyzing chunks",
    "analyzed": "Chunks analyzed",
    "groundingResponse": "Validating response",
    "groundedResponse": "Response validated",
    "withParams": " with params {'<'}b>{params}{'<'}/b>",
    "seeMore": "See more",
    "seeMoreUnavailable": "Available when the answer finishes",
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
    "result": "Resultado:",
    "description": "Descripción",
    "retrieving": "Recuperando secciones",
    "retrieved": "Secciones recuperadas",
    "analyzing": "Analizando secciones",
    "analyzed": "Secciones analizadas",
    "groundingResponse": "Validando respuesta",
    "groundedResponse": "Respuesta validada",
    "withParams": " con los siguientes parametros {'<'}b>{params}{'<'}/b>",
    "seeMore": "Ver más",
    "seeMoreUnavailable": "Disponible cuando termine la respuesta",
    "delegatingToSubagent": "Delegando a {'<'}b>{agent}{'<'}/b>",
    "subagentFinished": "Subagente {'<'}b>{agent}{'<'}/b> finalizó"
  }
}
</i18n>
