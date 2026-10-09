<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { IconInfoCircleFilled, IconPackageImport } from '@tabler/icons-vue'
import { buildToolConfigName, parseSubagentId } from '../../../../common/src/utils/toolConfig'
import type { AgentImportResult } from '@/services/api'

export interface WarningItem {
  title: string
  message: string
}

const props = defineProps<{
  result: AgentImportResult
}>()

defineEmits<{
  (e: 'review'): void
}>()

const { t } = useI18n()

const formatToolNames = (toolNames: string[]) =>
  toolNames.map((toolName) => buildToolConfigName(toolName)).join(', ')

const formatSubagentNames = (toolIds: string[]) =>
  toolIds.map((toolId) => {
    const id = parseSubagentId(toolId)
    return id == null ? toolId : t('subagentName', { id })
  }).join(', ')

const warningItems = computed(() => {
  const items: WarningItem[] = []
  const result = props.result

  if (result.unavailableModel) {
    items.push({
      title: t('modelReplacedTitle'),
      message: t('modelReplacedMessage', { model: result.unavailableModel, defaultModel: result.defaultModel })
    })
  }

  if (result.unavailableTools.length > 0) {
    items.push({
      title: t('unavailableToolsTitle'),
      message: t('unavailableToolsMessage', { tools: formatToolNames(result.unavailableTools) })
    })
  }

  if (result.unavailableSubagents.length > 0) {
    items.push({
      title: t('unavailableSubagentsTitle'),
      message: t('unavailableSubagentsMessage', { subagents: formatSubagentNames(result.unavailableSubagents) })
    })
  }

  if (result.toolsRequiringAuthentication.length > 0) {
    items.push({
      title: t('authToolsTitle'),
      message: t('authToolsMessage', { tools: formatToolNames(result.toolsRequiringAuthentication) })
    })
  }

  return items
})
</script>

<template>
  <div class="flex flex-col gap-4">
    <div class="flex flex-col gap-2 pt-6">
      <IconPackageImport size="80" stroke-width="1" />
      <span class="bold-span pt-4 text-lg">{{ t('importedWarningTitle') }}</span>
      <span>{{ t('importedWarningDescription') }}</span>
    </div>

    <div class="flex w-full flex-row gap-2 rounded-lg bg-surface-muted p-4 text-sm">
      <IconInfoCircleFilled size="24" class="text-content" />
      <div class="flex flex-col flex-1 gap-4 [&>*:not(:last-child)]:border-b [&>*:not(:last-child)]:pb-4">
        <div v-for="(item, index) in warningItems" :key="index" class="flex flex-col flex-1">
          <span class="bold-span">{{ item.title }}</span>
          <span v-html="item.message"></span>
        </div>
      </div>
    </div>

    <div class="mt-6 flex justify-end">
      <SimpleButton @click="$emit('review')" variant="primary" shape="square" class="whitespace-nowrap px-4">
        {{ t('reviewChanges') }}
      </SimpleButton>
    </div>
  </div>
</template>

<i18n lang="json">
{
  "en": {
    "importedWarningTitle": "Agent imported with changes",
    "importedWarningDescription": "Your agent was imported successfully, but some settings were adjusted for this environment. Review the items below before using it.",
    "modelReplacedTitle": "Model updated",
    "modelReplacedMessage": "\"{model}\" isn't available in this environment, so we replaced it with the default model: {'<'}span class='bold-span'> {defaultModel}{'<'}/span>.",
    "unavailableToolsTitle": "Tools not imported",
    "unavailableToolsMessage": "These tools aren't available in this environment and were skipped: {'<'}span class='bold-span'> {tools}{'<'}/span>.",
    "unavailableSubagentsTitle": "Subagents not imported",
    "unavailableSubagentsMessage": "These subagents aren't visible to you and were skipped: {'<'}span class='bold-span'> {subagents}{'<'}/span>.",
    "subagentName": "Subagent {id}",
    "authToolsTitle": "Tools that need configuration",
    "authToolsMessage": "These tools were imported, but need to be authenticated or configured before they can be used: {'<'}span class='bold-span'> {tools}{'<'}/span>.",
    "reviewChanges": "Review changes"
  },
  "es": {
    "importedWarningTitle": "Agente importado con cambios",
    "importedWarningDescription": "La configuración del agente que estás editando ha sido reemplazada con la nueva configuración.",
    "modelReplacedTitle": "Modelo actualizado",
    "modelReplacedMessage": "\"{model}\" no está disponible en este entorno, así que lo reemplazamos con el modelo por defecto: {'<'}span class='bold-span'> {defaultModel}{'<'}/span>.",
    "unavailableToolsTitle": "Herramientas no importadas",
    "unavailableToolsMessage": "Estas herramientas no están disponibles en este entorno y fueron omitidas: {'<'}span class='bold-span'> {tools}{'<'}/span>.",
    "unavailableSubagentsTitle": "Subagentes no importados",
    "unavailableSubagentsMessage": "Estos subagentes no son visibles para ti y fueron omitidos: {'<'}span class='bold-span'> {subagents}{'<'}/span>.",
    "subagentName": "Subagente {id}",
    "authToolsTitle": "Herramientas que necesitan configuración",
    "authToolsMessage": "Estas herramientas fueron importadas, pero necesitan ser autenticadas o configuradas antes de poder usarse: {'<'}span class='bold-span'> {tools}{'<'}/span>.",
    "reviewChanges": "Revisar cambios"
  }
}
</i18n>
