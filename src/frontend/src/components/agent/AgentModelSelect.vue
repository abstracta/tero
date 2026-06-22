<script lang="ts" setup>
import { computed, ref, watch, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { LlmModel, LlmModelVendor, llmModelVendorDisplayName, LlmModelType, LlmTemperature, ReasoningEffort } from '@/services/api'
import VendorLogo from '@/components/common/VendorLogo.vue'
import LlmModelSettings from './LlmModelSettings.vue'
import GroupedSelectPanel, { type GroupedSelectPanelOptionGroup } from '@/components/common/GroupedSelectPanel.vue'

const props = defineProps<{
  modelId?: string
  models: LlmModel[]
  temperature?: LlmTemperature
  reasoningEffort?: ReasoningEffort
  size?: 'normal' | 'large'
}>()

const emit = defineEmits<{
  (e: 'update:modelId', value: string): void
  (e: 'update:temperature', value: LlmTemperature): void
  (e: 'update:reasoningEffort', value: ReasoningEffort): void
  (e: 'change'): void
}>()

const { t } = useI18n()

const showAllModels = ref(false)
const searchQuery = ref('')

const selectedModel = computed(() => props.models.find((m) => m.id === props.modelId))

const updateShowAllModels = () => {
  if (selectedModel.value && !selectedModel.value.isBasic) {
    showAllModels.value = true
  } else if (!selectedModel.value) {
    showAllModels.value = false
  }
}

watch(() => props.modelId, () => {
  updateShowAllModels()
})

onMounted(() => {
  updateShowAllModels()
})

const modelType = computed(() => selectedModel.value?.modelType)

const groupByVendor = (models: LlmModel[]): Map<LlmModelVendor, LlmModel[]> => {
  const map = new Map<LlmModelVendor, LlmModel[]>()
  for (const model of models) {
    const list = map.get(model.modelVendor)
    if (list) list.push(model)
    else map.set(model.modelVendor, [model])
  }
  return map
}

const groupedModelOptions = computed<GroupedSelectPanelOptionGroup[]>(() => {
  let list: LlmModel[]
  if (showAllModels.value) {
    list = props.models
  } else {
    const basic = props.models.filter((m) => m.isBasic)
    const selected = selectedModel.value
    list = selected && !selected.isBasic && !basic.some((m) => m.id === selected.id) ? [...basic, selected] : basic
  }

  const q = searchQuery.value.trim().toLowerCase()
  if (q) {
    list = list.filter(({ modelVendor, name }) => [modelVendor?.toString(), name].some((v) => v?.toLowerCase().includes(q)))
  }
  return [...groupByVendor(list).entries()].map(([vendor, models]) => ({
    id: vendor,
    name: vendor,
    children: models.map((model) => ({
      id: model.id,
      name: model.name,
      description: model.description,
      costMultiplier: model.costMultiplier
    }))
  }))
})

const groupedSelectPanelRef = ref<InstanceType<typeof GroupedSelectPanel>>()

const onShowAllModels = () => {
  groupedSelectPanelRef.value?.onShowDropdown()
}

const onModelChange = (value: string) => {
  emit('update:modelId', value)
  emit('change')
  groupedSelectPanelRef.value?.onShowDropdown()
}

const onSearch = (value: string) => {
  searchQuery.value = value
  if (value?.trim()) {
    showAllModels.value = true
  }
}

const containerRef = ref<HTMLDivElement>()
const buttonRef = ref<HTMLButtonElement>()
const selectedOptionId = computed(() => selectedModel.value?.id ?? null)
const visibleOptions = computed(() => groupedModelOptions.value.filter((option) => option.children!.length > 0))
const baseCostModelName = computed(() => props.models.find((m) => m.isBaseCostModel)?.name)
</script>

<template>
  <div class="flex flex-row gap-5 relative" ref="containerRef">
    <div class="form-field flex-1 gap-2">
      <label :class="size === 'large' ? '!text-sm' : '!text-base'" for="model">{{ t('modelLabel') }}</label>
      <button ref="buttonRef" class="flex items-center w-full border rounded-xl p-2 justify-between focus:border-abstracta h-12" type="button" @click="onShowAllModels">
        <span class="flex flex-row items-center gap-2">
          <VendorLogo :vendor="selectedModel!.modelVendor"/>
          <span class="font-medium">{{ selectedModel?.name }}</span>
        </span>
        <IconCaretDown class="fill-abstracta border-none" />
      </button>
      <span class="text-sm text-content-muted">{{ t('modelDescription') }}</span>
      <GroupedSelectPanel ref="groupedSelectPanelRef" :search-placeholder="t('searchPlaceholder')" :container="containerRef" :anchor="buttonRef" :show-load-more="!showAllModels && models.some((m) => !m.isBasic)" @search="onSearch" @load-more="showAllModels = true">
        <template #content>
          <div v-if="visibleOptions.length > 0" class="flex flex-row items-center gap-2">
            <div class="flex flex-col items-stretch w-full">
              <div v-for="(option, index) in visibleOptions" :key="option.id" class="py-3" :class="[!(showAllModels && index === visibleOptions.length - 1) && 'border-b-2 border-dotted']">
                <div class="flex flex-row items-start gap-3 w-full">
                  <div class="w-36 shrink-0 pt-2 flex items-center gap-2 pl-2">
                    <VendorLogo v-if="option.id" :vendor="option.id as LlmModelVendor" />
                    <span class="text-lg font-semibold">{{ llmModelVendorDisplayName[option.id as LlmModelVendor] }}</span>
                  </div>
                  <div class="flex-1 min-w-0 flex flex-col gap-2">
                    <div
                      v-for="optionItem in option.children"
                      :key="optionItem.id"
                      class="flex flex-col gap-2 hover:bg-surface-muted rounded-2xl p-2 px-4 cursor-pointer"
                      :class="{ 'bg-surface-muted': selectedOptionId === optionItem.id }"
                      @click="onModelChange(optionItem.id)"
                    >
                      <div class="flex items-center justify-between">
                        <span v-if="optionItem.name" class="font-semibold">{{ optionItem.name }}</span>
                        <span v-if="optionItem.costMultiplier != null" v-tooltip="t('costRatioTooltip', { baseModel: baseCostModelName })" class="text-xs flex items-center gap-0.5 border p-0.5 pr-2 rounded-xl bg-surface font-semibold">
                          <IconCurrencyDollar class="w-4 h-4 text-content-muted" />
                          x{{ Number.isInteger(optionItem.costMultiplier) ? optionItem.costMultiplier : Number(optionItem.costMultiplier).toFixed(1) }}
                        </span>
                      </div>
                      <span>{{ optionItem.description }}</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
          <div v-else class="text-sm text-content-muted text-center py-2 min-h-[30vh] flex items-center justify-center">
            {{ t('noOptionsPlaceholder') }}
          </div>
        </template>
      </GroupedSelectPanel>
    </div>
    <div class="form-field gap-2 w-1/2">
      <LlmModelSettings
        :model-type="modelType"
        :temperature="temperature"
        :reasoning-effort="reasoningEffort"
        :size="size"
        @update:temperature="(value) => emit('update:temperature', value)"
        @update:reasoning-effort="(value) => emit('update:reasoningEffort', value)"
        @change="emit('change')"
      />
      <span v-if="modelType === LlmModelType.CHAT" class="text-sm text-content-muted">{{ t('modelSettingsDescriptionTemperature') }}</span>
      <span v-else class="text-sm text-content-muted">{{ t('modelSettingsDescriptionReasoningEffort') }}</span>
    </div>
  </div>
</template>

<i18n lang="json">
{
  "en": {
    "modelLabel": "Model",
    "searchPlaceholder": "Search model...",
    "noOptionsPlaceholder": "No models found",
    "modelDescription": "Choose the model this agent will use to respond. Different models may vary in quality, speed, and cost.",
    "modelSettingsDescriptionTemperature": "Choose how the agent balances precision and creativity.",
    "modelSettingsDescriptionReasoningEffort": "Choose how much the agent analyzes before answering.",
    "costRatioTooltip": "Compared to {baseModel}. A higher multiplier means higher estimated budget usage."
  },
  "es": {
    "modelLabel": "Modelo",
    "searchPlaceholder": "Buscar modelo...",
    "noOptionsPlaceholder": "No se han encontrado modelos",
    "modelDescription": "Elige el modelo que este agente usará para responder. Los modelos pueden variar en calidad, velocidad y costo.",
    "modelSettingsDescriptionTemperature": "Elige cómo el agente balancea precisión y creatividad.",
    "modelSettingsDescriptionReasoningEffort": "Elige cuánto razona el agente antes de responder.",
    "costRatioTooltip": "Comparado con {baseModel}. Un multiplicador más alto significa un mayor uso estimado del presupuesto."
  }
}
</i18n>
