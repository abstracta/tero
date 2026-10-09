<script setup lang="ts">
import { computed, nextTick, onMounted, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { IconEditCircle } from '@tabler/icons-vue'
import { Agent, ApiService, GLOBAL_TEAM_ID, MY_TEAM_ID } from '@/services/api'
import { useErrorHandler } from '@/composables/useErrorHandler'
import { useDebounce } from '@/composables/useDebounce'
import { useAgentStore } from '@/composables/useAgentStore'

const props = defineProps<{
  modelValue?: number | null
  excludeAgentIds: number[]
  parentTeamId?: number | null
  disabled?: boolean
  inputId?: string
}>()

const emit = defineEmits<{
  (e: 'update:modelValue', value: number | null): void
}>()

const { t } = useI18n()
const api = new ApiService()
const { handleError } = useErrorHandler()
const { configureAgent } = useAgentStore()

const pageLimit = 30
const defaultAgents = ref<Agent[]>([])
const agents = ref<Agent[]>([])
const selectedAgentCache = ref<Agent | null>(null)
const searchQuery = ref('')
const loading = ref(false)
const searchInputRef = ref<{ focus: () => void } | null>(null)

const model = computed({
  get: () => props.modelValue ?? null,
  set: (value: number | null) => {
    emit('update:modelValue', value)
  },
})

const isAddableSubagent = (agent: Agent) => {
  if (props.parentTeamId == null || props.parentTeamId === MY_TEAM_ID) {
    return true
  }
  const teamId = agent.team?.id
  return teamId === props.parentTeamId || teamId === GLOBAL_TEAM_ID
}

const addableAgents = (list: Agent[]) => list.filter(isAddableSubagent)

const options = computed(() => {
  const excluded = new Set(props.excludeAgentIds)
  if (props.modelValue) {
    excluded.delete(props.modelValue)
  }
  const mapped = agents.value
    .filter((agent) => !excluded.has(agent.id))
    .filter((agent) => agent.id === props.modelValue || isAddableSubagent(agent))
    .map((agent) => ({
      id: agent.id,
      name: agent.name,
      description: agent.description?.trim() || '',
      agent,
    }))
  if (searchQuery.value.trim()) {
    return mapped
  }
  return mapped.sort((a, b) => (a.name ?? '').localeCompare(b.name ?? ''))
})

const selectedAgent = computed(() => {
  if (model.value == null) {
    return null
  }
  return agents.value.find((agent) => agent.id === model.value)
    ?? (selectedAgentCache.value?.id === model.value ? selectedAgentCache.value : null)
})

const loadSelectedAgent = async (id: number) => {
  if (agents.value.some((agent) => agent.id === id)) {
    selectedAgentCache.value = agents.value.find((agent) => agent.id === id) ?? selectedAgentCache.value
    return
  }
  const cached =
    (selectedAgentCache.value?.id === id ? selectedAgentCache.value : null)
    ?? defaultAgents.value.find((agent) => agent.id === id)
  if (cached) {
    selectedAgentCache.value = cached
    agents.value = [...agents.value, cached]
    return
  }
  try {
    const agent = await api.findAgentById(id)
    selectedAgentCache.value = agent
    agents.value = [...agents.value, agent]
  } catch {
    // Keep options without the selected agent if it cannot be loaded.
  }
}

const applyAgents = async (list: Agent[]) => {
  agents.value = list
  if (props.modelValue && !searchQuery.value.trim()) {
    await loadSelectedAgent(props.modelValue)
  }
}

const loadDefaultAgents = async () => {
  loading.value = true
  try {
    const [own, newest] = await Promise.all([api.findOwnAgents(pageLimit), api.findNewestAgents(pageLimit)])
    const byId = new Map<number, Agent>()
    for (const agent of [...own, ...newest]) {
      byId.set(agent.id, agent)
    }
    defaultAgents.value = addableAgents([...byId.values()]).slice(0, pageLimit)
    await applyAgents(defaultAgents.value)
  } catch (error) {
    handleError(error)
  } finally {
    loading.value = false
  }
}

const matchingDefaultAgents = (q: string) => {
  const query = q.toLowerCase()
  return defaultAgents.value.filter((agent) =>
    agent.name?.toLowerCase().includes(query) || agent.description?.toLowerCase().includes(query)
  )
}

const searchAgents = async () => {
  const q = searchQuery.value.trim()
  if (!q) {
    await applyAgents(defaultAgents.value)
    return
  }
  try {
    const results = await api.findAgentsByText(q, pageLimit)
    if (searchQuery.value.trim() !== q) {
      return
    }
    const byId = new Map<number, Agent>()
    for (const agent of [...matchingDefaultAgents(q), ...results]) {
      byId.set(agent.id, agent)
    }
    await applyAgents(addableAgents([...byId.values()]).slice(0, pageLimit))
  } catch (error) {
    handleError(error)
  }
}

const debouncedSearch = useDebounce(searchAgents)

const onSearch = (value: string | number | undefined) => {
  searchQuery.value = String(value ?? '')
  const q = searchQuery.value.trim()
  if (!q) {
    applyAgents(defaultAgents.value)
    return
  }
  applyAgents(matchingDefaultAgents(q))
  debouncedSearch()
}

const onOverlayShow = () => {
  nextTick(() => searchInputRef.value?.focus())
}

const onOverlayHide = async () => {
  searchQuery.value = ''
  await applyAgents(defaultAgents.value)
}

onMounted(loadDefaultAgents)

watch(() => props.modelValue, async (id) => {
  if (id) {
    await loadSelectedAgent(id)
  }
})
</script>

<template>
  <div class="flex items-center gap-2 my-1">
    <Select
      v-model="model"
      :input-id="inputId"
      :options="options"
      option-label="name"
      option-value="id"
      :placeholder="t('placeholder')"
      :disabled="disabled || loading"
      class="flex-1 min-w-0"
      :class="{ '!bg-surface-muted !cursor-not-allowed !opacity-60': disabled }"
      @show="onOverlayShow"
      @hide="onOverlayHide"
    >
      <template #header>
        <div class="w-full py-2 px-4">
          <InteractiveInput
            ref="searchInputRef"
            autofocus
            :model-value="searchQuery"
            @update:model-value="onSearch"
            :placeholder="t('searchPlaceholder')"
            start-icon="IconSearch"
            class="w-full text-sm"
          />
        </div>
      </template>
      <template #value="{ value, placeholder }">
        <div v-if="value != null" class="flex items-center gap-2 min-h-8">
          <div v-if="selectedAgent" class="relative shrink-0">
            <AgentAvatar :agent="selectedAgent" :show-shared-status="true" />
          </div>
          <span>{{ selectedAgent?.name ?? (loading ? '' : t('unavailableSubagent')) }}</span>
        </div>
        <span v-else>{{ placeholder }}</span>
      </template>
      <template #option="{ option }">
        <div class="flex items-center gap-2 py-1 max-w-full">
          <div class="relative shrink-0">
            <AgentAvatar :agent="option.agent" :show-shared-status="true" />
          </div>
          <div class="flex flex-col gap-0.5 min-w-0">
            <span>{{ option.name }}</span>
            <span
              v-if="option.description"
              class="text-sm text-content-muted line-clamp-1"
            >
              {{ option.description }}
            </span>
          </div>
        </div>
      </template>
    </Select>
    <SimpleButton
      v-if="selectedAgent?.canEdit"
      class="shrink-0"
      v-tooltip.bottom="t('editAgentTooltip')"
      @click="configureAgent(selectedAgent.id)"
    >
      <IconEditCircle />
    </SimpleButton>
  </div>
</template>

<i18n lang="json">
{
  "en": {
    "placeholder": "Select an agent",
    "searchPlaceholder": "Search agents...",
    "editAgentTooltip": "Edit agent",
    "unavailableSubagent": "Unavailable subagent"
  },
  "es": {
    "placeholder": "Seleccioná un agente",
    "searchPlaceholder": "Buscar agentes...",
    "editAgentTooltip": "Editar agente",
    "unavailableSubagent": "Subagente no disponible"
  }
}
</i18n>
