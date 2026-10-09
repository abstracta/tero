import { ref, watch, type MaybeRefOrGetter, toValue } from 'vue'
import { useI18n } from 'vue-i18n'
import { ApiService, HttpError, type AgentToolConfig } from '@/services/api'
import { buildToolConfigName, isSubagentToolId, parseSubagentId } from '@tero/common/utils/toolConfig.js'

type LabelEnricher = (configs: AgentToolConfig[], api: ApiService, t: (key: string) => string) => Promise<Record<string, string>>

const enrichSubagentLabels: LabelEnricher = async (configs, api, t) => {
  const labels: Record<string, string> = {}
  await Promise.all(configs.map(async (config) => {
    const id = parseSubagentId(config.toolId)
    if (id == null) {
      return
    }
    try {
      const agent = await api.findAgentById(id)
      if (agent.name) {
        labels[config.toolId] = agent.name
      }
    } catch (error) {
      if (error instanceof HttpError && error.status === 404) {
        labels[config.toolId] = t('unavailableSubagent')
      }
    }
  }))
  return labels
}

const enrichers: LabelEnricher[] = [enrichSubagentLabels]

export const useToolConfigLabels = (toolConfigs: MaybeRefOrGetter<AgentToolConfig[]>) => {
  const api = new ApiService()
  const { t, mergeLocaleMessage } = useI18n({ useScope: 'global' })
  const customLabels = ref<Record<string, string>>({})

  mergeLocaleMessage('en', {
    unavailableSubagent: 'Unavailable subagent',
  })
  mergeLocaleMessage('es', {
    unavailableSubagent: 'Subagente no disponible',
  })

  watch(
    () => toValue(toolConfigs),
    async (configs) => {
      const maps = await Promise.all(enrichers.map((enrich) => enrich(configs, api, t)))
      customLabels.value = Object.assign({}, ...maps)
    },
    { immediate: true },
  )

  const toolConfigLabel = (toolId: string) => {
    const custom = customLabels.value[toolId]
    if (custom) {
      return custom
    }
    if (isSubagentToolId(toolId)) {
      return ''
    }
    return buildToolConfigName(toolId)
  }

  return { toolConfigLabel }
}
