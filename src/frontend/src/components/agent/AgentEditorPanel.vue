<script lang="ts" setup>
import { onMounted, ref, computed, watch } from 'vue'
import { useRoute, onBeforeRouteUpdate, useRouter } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { Agent, AgentType, ApiService, HttpError, LlmModel, AgentToolConfig, AutomaticAgentField, Team, TestCase, TestSuiteRun, GLOBAL_TEAM_ID, Role, TeamRoleStatus, findManifest, type AgentImportResult } from '@/services/api'
import { IconPencil, IconListDetails, IconLock, IconDeviceFloppy, IconPackageExport, IconPackageImport } from '@tabler/icons-vue'
import { useErrorHandler } from '@/composables/useErrorHandler'
import { useAgentStore } from '@/composables/useAgentStore'
import { useAgentPromptStore } from '@/composables/useAgentPromptStore'
import { useTestCaseStore } from '@/composables/useTestCaseStore'
import { loadUserProfile } from '@/composables/useUserProfile'
import { AgentPrompt, UploadedFile } from '../../../../common/src/utils/domain'
import { AgentTestcaseChatUiMessage } from './AgentTestcaseChatMessage.vue'
import AgentCollapsibleSection from './AgentCollapsibleSection.vue'
import moment from 'moment'

const props = defineProps<{
  selectedThreadId: number
  loadingTests?: boolean
  editingTestcase?: boolean
  testCases?: TestCase[]
  isComparingResultWithTestSpec?: boolean
  testSpecMessages?: AgentTestcaseChatUiMessage[]
  initialTab?: string
}>()

const emit = defineEmits<{
  (e: 'showTestCaseEditor', show: boolean): void
  (e: 'editingTestcase', editing: boolean): void
  (e: 'runTests'): void
  (e: 'runSingleTest', testCaseId: number): void
  (e: 'importAgent' ): void
  (e: 'selectExecution', execution: TestSuiteRun): void
}>()

const { t } = useI18n()
const { handleError } = useErrorHandler()
const { setCurrentAgent } = useAgentStore()
const { testCasesStore } = useTestCaseStore()

const api = new ApiService()
const route = useRoute()
const router = useRouter()

const agent = ref<Agent>();
const menu = ref();
const invalidAttrs = ref<string>('')
const backendAgent = ref<Agent>()
const isSaving = ref(false)
const models = ref<LlmModel[]>([])
const toolConfigs = ref<AgentToolConfig[]>([])
const isGenerating = ref({
  name: false,
  description: false,
  systemPrompt: false
})
const nameMaxLength = 30
const descriptionMaxLength = 100
const showShareConfirmation = ref(false)
const starters = ref<AgentPrompt[]>([])
const { agentsPromptStore, loadAgentPrompts, removePrompt, setPrompts } = useAgentPromptStore()
const publishPrompts = ref(false)
const privatePromptsCount = computed(() => agentsPromptStore.prompts.filter(p => !p.shared).length)
const isLoading = ref(true)
const teams = ref<Team[]>([])
const defaultTeams = ref<Team[]>([new Team(0, t('private'))])
const selectedTeam = ref<number | null>(null)
const activeTab = ref<string>(props.initialTab ?? '0')
const showImportAgent = ref(false)
const importResult = ref<AgentImportResult | null>(null)
const importError = ref<string | null>(null)
const showPastExecutions = ref(false)
const showGenerateDialog = ref(false)
const selectedField = ref<AutomaticAgentField | null>(null)
const savingAdvancedSettings = ref(false)
const collapsed = ref({
  basicInfo: false,
  behavior: false,
  advancedSettings: true,
})
const RECURSION_LIMIT_OPTIONS = [20, 40, 60, 80, 100]
const AGENT_TYPE_OPTIONS = [
  { label: t('agentTypeBasicAgent'), value: AgentType.REACT_AGENT },
  { label: t('agentTypeAdvancedAgent'), value: AgentType.DEEP_AGENT }
]
const lastSavedAt = ref<Date | null>(null)
const isGlobalOwner = ref(false)

const isOlderThanOneDay = computed(() => {
  return moment().diff(moment.utc(lastSavedAt.value).local(), 'days') >= 1
})

const formattedSavedTime = computed(() => {
  const saved = moment.utc(lastSavedAt.value).local()
  if (isOlderThanOneDay.value) {
    return saved.format('MMM D, YYYY')
  }
  return saved.format('h:mm A')
})

const loadAgentData = async (agentIdStr: string) => {
  const agentId = parseInt(agentIdStr)
  if (isNaN(agentId)) return

  try {
    agent.value = await api.findAgentById(agentId)
    toolConfigs.value = await api.findAgentToolConfigs(agentId)
    backendAgent.value = { ...agent.value }
    selectedTeam.value = agent.value?.team?.id ?? 0;
    lastSavedAt.value = agent.value?.lastUpdate ? moment.utc(agent.value.lastUpdate).toDate() : null
    setCurrentAgent(agent.value)
    await loadPromptStarters(agentId)
  } catch (e) {
    handleError(e)
  } finally {
    isLoading.value = false
  }
}

const loadPromptStarters = async (agentId: number) => {
  await loadAgentPrompts(agentId)
  starters.value = agentsPromptStore.prompts.filter(p => p.starter) || []
}

onMounted(async () => {
  try {
    models.value = await api.findModels()
    const manifest = await findManifest()
    const user = await loadUserProfile()
    isGlobalOwner.value = user!.teams.some(t =>
      t.role === Role.TEAM_OWNER && t.id === GLOBAL_TEAM_ID && t.status === TeamRoleStatus.ACCEPTED
    )

    if (!manifest.disablePublishGlobal || isGlobalOwner.value) {
      defaultTeams.value.push(new Team(GLOBAL_TEAM_ID, t('global')))
    }

    teams.value = [...defaultTeams.value, ...user!.teams.filter(t => !defaultTeams.value.some(dt => dt.id === t.id))];

    if (route.params.agentId) {
      await loadAgentData(route.params.agentId as string)
    }
  }
  finally {
    isLoading.value = false
  }
})


onBeforeRouteUpdate(async (to) => {
  if (to.params.agentId) {
    await loadAgentData(to.params.agentId as string)
  }
})

const onClose = async () => {
  await router.push(`/chat/${props.selectedThreadId}`)
}

const checkAgentFields = () => {
  if (!agent.value) {
    return ''
  }

  const invalidAttrs = []
  if (!agent.value.name && isSelectedPublicTeam.value) {
    invalidAttrs.push(t('shareMissingName'))
  } else if (agent.value.name && agent.value.name.length > nameMaxLength) {
    invalidAttrs.push(t('shareNameTooLong'))
  }

  if (!agent.value.description && isSelectedPublicTeam.value) {
    invalidAttrs.push(t('shareMissingDescription'))
  } else if (agent.value.description && agent.value.description.length > descriptionMaxLength) {
    invalidAttrs.push(t('shareDescriptionTooLong'))
  }


  if (invalidAttrs.length == 0)
    return ''

  const lastVal = invalidAttrs.pop()
  return invalidAttrs.join(', ') + (invalidAttrs.length > 0 ? ' ' + t('shareMissingAnd') + ' ' : '') + lastVal
}

const findTeam = (teamId: number) => {
  return teams.value.find(t => t.id === teamId)
}

const compareAgents = (a: Agent, b: Agent) => {
  return JSON.stringify({ ...a, team: findTeam(a.team?.id || 0) }) === JSON.stringify({ ...b, team: findTeam(b.team?.id || 0) })
}

const updateAgent = async () => {
  if (!agent.value)
    return

  invalidAttrs.value = checkAgentFields()
  if (invalidAttrs.value) {
    return
  }

  if (!agent.value.name?.trim()) {
    agent.value.name = `Agent #${agent.value.id}`
  }

  if (!compareAgents({ ...agent.value, team: findTeam(selectedTeam.value!) }, backendAgent.value!)) {
    isSaving.value = true
    try {
      await api.updateAgent({ ...agent.value!, publishPrompts: publishPrompts.value, teamId: selectedTeam.value || null })
      const updatedAgent = { ...agent.value!, team: findTeam(selectedTeam.value!) }
      agent.value = updatedAgent;
      setCurrentAgent(updatedAgent);
      backendAgent.value = { ...updatedAgent } as Agent
      if (publishPrompts.value) {
        await setPrompts(agentsPromptStore.prompts.map(p => ({ ...p, shared: true })))
      }
      lastSavedAt.value = moment.utc().toDate()
      setTimeout(() => {
        isSaving.value = false
      }, 2000)
    } catch (e) {
      handleError(e)
      isSaving.value = false
    }
  }
}

const onChangeTeam = async (team: number | null) => {
  if (team !== agent.value?.team?.id) {
    publishPrompts.value = false
    invalidAttrs.value = checkAgentFields();
    showShareConfirmation.value = true;
  }
}

const onConfirmChangeTeam = async () => {
  await updateAgent()
  showShareConfirmation.value = false
}

const onCancelChangeTeam = async () => {
  selectedTeam.value = agent.value?.team?.id ?? 0;
  showShareConfirmation.value = false
}

const generateAgentField = async (field: AutomaticAgentField, callback: (v: string) => void, generatingCallback: (v: boolean) => void) => {
  try {
    generatingCallback(true)
    const response = await api.generateAgentField(agent.value!.id, field)
    callback(response)
    await updateAgent()
  } catch (e) {
    handleError(e)
  } finally {
    generatingCallback(false)
  }
}

const handleStarterDelete = async (starterId: number) => {
  try {
    await removePrompt(agent.value!.id, starterId)
    await loadPromptStarters(agent.value!.id)
  } catch (e) {
    handleError(e)
  }
}

const handleReloadStarters = async () => {
  try {
    await loadPromptStarters(agent.value!.id)
  } catch (e) {
    handleError(e)
  }
}

const onUpdateToolConfigs = async () => {
  toolConfigs.value = await api.findAgentToolConfigs(agent.value!.id)
}

const onTestCaseDeleted = () => {
  if (testCasesStore.testCases.length === 0 && !props.editingTestcase) {
    emit('editingTestcase', true)
  }
}

watch(activeTab, (newVal) => {
  emit('showTestCaseEditor', newVal === '1')
  emit('editingTestcase', newVal === '1')
})

watch(
  () => props.initialTab,
  (newTab) => {
    if (newTab && newTab !== activeTab.value) {
      activeTab.value = newTab
    }
  }
)

const isSelectedPublicTeam = computed(() => selectedTeam.value != null && selectedTeam.value > 0)

const hasPublishableTeams = computed(() => teams.value.some(t => t.id > 0))

const shareDialogTranslationKey = computed(() => {
  if (invalidAttrs.value) {
    return isSelectedPublicTeam.value ? 'shareInvalidAttrs' : 'unshareInvalidAttrs'
  }

  if (isSelectedPublicTeam.value) {
    if (agent.value?.team?.id) {
      return 'changeTeamConfirmationMessage'
    }
    return selectedTeam.value === 1 ? 'shareConfirmationMessageGlobal' : 'shareConfirmationMessageTeam'
  }

  return 'unshareConfirmationMessage'
})

const shareDialogTranslationParams = computed(() => ({
  invalidAttrs: invalidAttrs.value,
  team: findTeam(selectedTeam.value!)?.name
}))

const exportAgent = async () => {
  try {
    const response = await api.exportAgent(agent.value!.id)
    const url = window.URL.createObjectURL(response)
    const link = document.createElement('a')
    link.href = url
    link.download = response.name
    link.click()
    window.URL.revokeObjectURL(url)
  } catch (error) {
    handleError(error)
  }
}

const onImportAgent = async (file: UploadedFile) => {
  try {
    importResult.value = null
    importError.value = null
    const result = await api.importAgent(agent.value!.id, file.file!)
    await loadAgentData(String(agent.value!.id))
    emit('importAgent')
    importResult.value = result
  } catch (error) {
    if (error instanceof HttpError && error.status === 400) {
      const parsedImportError = JSON.parse(error.body)?.detail as string | undefined
      if (parsedImportError) {
        importResult.value = null
        importError.value = parsedImportError
        return
      }
    }
    showImportAgent.value = false
    handleError(error)
  }
}

const onUpdateAdvancedSettings = async (value: number) => {
  agent.value!.recursionLimit = value
  savingAdvancedSettings.value = true
  try {
    await updateAgent()
  } catch (error) {
    handleError(error)
  } finally {
    savingAdvancedSettings.value = false
  }
}

const onUpdateProtectedSettings = async (value: boolean) => {
  if (!agent.value) return
  agent.value.isProtected = value
  await updateAgent()
}

const onUpdateAgentType = async (value: AgentType) => {
  if (!agent.value) return
  agent.value.agentType = value
  await updateAgent()
}

const onSelectExecution = (execution: TestSuiteRun) => {
  emit('selectExecution', execution)
  showPastExecutions.value = false
}

const onGenerateDialog = async (field: AutomaticAgentField) => {
  showGenerateDialog.value = true
  selectedField.value = field
}

const onGenerate = async () => {
  showGenerateDialog.value = false
  switch (selectedField.value) {
    case AutomaticAgentField.NAME:
      await generateAgentField(selectedField.value, (v) => agent.value!.name = v, (v) => isGenerating.value.name = v)
      break
    case AutomaticAgentField.DESCRIPTION:
      await generateAgentField(selectedField.value, (v) => agent.value!.description = v, (v) => isGenerating.value.description = v)
      break
    case AutomaticAgentField.SYSTEM_PROMPT:
      await generateAgentField(selectedField.value, (v) => agent.value!.systemPrompt = v, (v) => isGenerating.value.systemPrompt = v)
      break
  }
  selectedField.value = null
}
</script>

<template>
  <Tabs v-model:value="activeTab" class="h-full flex flex-col pb-6">
    <FlexCard header-height="auto" header-class="!py-0" class="flex flex-col h-full">
      <template #header>
        <div v-if="isComparingResultWithTestSpec && testSpecMessages" class="flex flex-row gap-2 items-center min-h-[73px]">
          <IconListDetails />
          {{ t('testSpec') }}
        </div>
        <div v-else class="flex flex-row justify-between items-center">
          <div class="flex flex-row items-center gap-3 mt-1">
            <SimpleButton @click="onClose">
              <IconArrowLeft />
            </SimpleButton>
            <TabList class="!p-0 !pt-3">
              <Tab value="0">
                <div class="flex gap-2">
                  <IconPencil size="20" />
                  {{ t('editAgentTabTitle') }}
                </div>
              </Tab>
              <Tab value="1">
                <div class="flex gap-2">
                  <IconBug size="20" />
                  {{ t('testsTabTitle') }}
                </div>
              </Tab>
            </TabList>
          </div>
          <AgentChatMenu
            v-if="activeTab === '0'"
            ref="menu"
            :can-view-agent-info="true"
            @menu-toggle="(event: Event) => menu?.toggle(event)"
            :items="[
              {label: t('exportAgent'), tablerIcon: IconPackageExport, command: () => exportAgent()},
              {label: t('importAgent'), tablerIcon: IconPackageImport, command: () => { importResult = null; importError = null; showImportAgent = true }}
            ]"/>
        </div>
      </template>
      <div v-if="isComparingResultWithTestSpec && testSpecMessages" class="flex-1 overflow-y-auto p-4">
        <div v-if="testSpecMessages.length > 0" class="flex flex-col gap-2">
          <div v-for="message in testSpecMessages" :key="message.uuid">
            <AgentTestcaseChatMessage :message="message"
              :actions-enabled="false"
              :is-selected="false"
              :selectable="false" />
          </div>
        </div>
      </div>
      <TabPanels v-else class="flex-1 !p-0.5 overflow-hidden">
        <TabPanel value="0" class="h-full overflow-y-auto">
          <AgentEditorPanelSkeleton v-if="isLoading" />
          <div class="flex flex-col gap-3 px-4 py-2 mb-4" v-if="agent && !isLoading">
            <AgentCollapsibleSection
              :title="t('basicInfoSectionTitle')"
              :collapsed="collapsed.basicInfo"
              @update:collapsed="collapsed.basicInfo = $event"
            >
              <div class="flex flex-col gap-7">
                <div class="flex flex-row justify-between">
                  <div class="form-field justify-center">
                    <AgentIconEditor v-model:icon="agent.icon" @change="updateAgent" />
                  </div>
                  <div class="form-field flex gap-2 w-7/12">
                    <label class="!text-sm" for="visibility">{{ t('visibilityLabel') }}</label>
                    <UserTeamsSelect id="visibility" v-model="selectedTeam" class="w-full" :default-teams="defaultTeams" :default-selected-team="selectedTeam"
                      @change="onChangeTeam" :disabled="!hasPublishableTeams" size="large"/>
                  </div>
                </div>
                <div class="form-field relative gap-1">
                  <label class="!text-sm" for="name">{{ t('nameLabel') }}</label>
                  <InteractiveInput id="name" v-model="agent.name" :maxlength="nameMaxLength"
                    :required="isSelectedPublicTeam" @blur="updateAgent" :placeholder="t('namePlaceholder')"
                    @end-icon-click="onGenerateDialog(AutomaticAgentField.NAME)" end-icon="IconWand" :loading="isGenerating.name" size="large" />
                  <span class="text-sm text-content-muted">{{ t('nameDescription')}}</span>
                </div>
                <div class="form-field relative gap-1">
                  <label class="!text-sm" for="description">{{ t('descriptionLabel') }}</label>
                  <InteractiveInput id="description" v-model="agent.description" :maxlength="descriptionMaxLength"
                    :required="isSelectedPublicTeam" @blur="updateAgent" :placeholder="t('descriptionPlaceholder')"
                    @end-icon-click="onGenerateDialog(AutomaticAgentField.DESCRIPTION)" end-icon="IconWand" :loading="isGenerating.description" size="large" />
                  <span class="text-sm text-content-muted">{{ t('descriptionDescription')}}</span>
                </div>
              </div>
            </AgentCollapsibleSection>
            <AgentCollapsibleSection
              :title="t('behaviorSectionTitle')"
              :collapsed="collapsed.behavior"
              @update:collapsed="collapsed.behavior = $event"
            >
              <template #header>
                <div class="flex items-center gap-4">
                  <h4 class="!text-sm font-semibold uppercase tracking-[0.2em]">
                    {{ t('behaviorSectionTitle') }}
                  </h4>
                  <div v-if="agent.isProtected" class="flex items-center gap-1.5">
                    <SimpleIcon :icon="IconLock" :size="14" class="bg-content text-surface rounded-md p-1" />
                    <span class="text-sm font-semibold">{{ t('protectedSettingsBadge') }}</span>
                  </div>
                </div>
              </template>
              <div class="flex flex-col gap-7">
                <AgentModelSelect
                  v-model:model-id="agent.modelId"
                  :models="models"
                  v-model:temperature="agent.temperature"
                  v-model:reasoning-effort="agent.reasoningEffort"
                  size="large"
                  @change="updateAgent"
                />
                <div class="form-field relative gap-0.5">
                  <label class="!text-sm mb-1.5" for="systemPrompt">{{ t('systemPromptLabel') }}</label>
                  <InteractiveInput id="systemPrompt" v-model="agent.systemPrompt" @blur="updateAgent" :rows="10"
                    :placeholder="t('systemPromptPlaceholder')" end-icon="IconWand" :loading="isGenerating.systemPrompt"
                    @end-icon-click="onGenerateDialog(AutomaticAgentField.SYSTEM_PROMPT)" />
                  <span class="text-sm text-content-muted">{{ t('systemPromptDescription') }}</span>
                </div>
                <div class="form-field">
                  <AgentToolConfigsEditor :agent-id="agent.id" :tool-configs="toolConfigs" @update="onUpdateToolConfigs" />
                </div>
                <div class="form-field relative">
                  <AgentConversationStarters :starters="starters" @delete="handleStarterDelete" :agent="agent"
                    @reload="handleReloadStarters" />
                </div>
              </div>
            </AgentCollapsibleSection>
            <AgentCollapsibleSection
              :title="t('advancedSettings')"
              :collapsed="collapsed.advancedSettings"
              @update:collapsed="collapsed.advancedSettings = $event"
            >
              <template #header>
                <div class="flex items-center gap-2">
                  <h4 class="!text-sm font-semibold uppercase tracking-[0.2em]">
                    {{ t('advancedSettings') }}
                  </h4>
                  <span class="uppercase text-sm text-content-muted tracking-[0.2em]">({{ t('optional') }})</span>
                </div>
              </template>
              <div v-if="isGlobalOwner" class="form-field flex flex-col gap-2 mb-6">
                <div class="flex items-center gap-3">
                  <ToggleSwitch
                  id="protectedConfiguration"
                  :model-value="agent.isProtected"
                  @update:model-value="onUpdateProtectedSettings"
                  />
                  <label for="protectedConfiguration">{{ t('protectedConfiguration') }}</label>
                </div>
                <p class="text-content-muted text-sm tool-message mb-2 whitespace-pre-line">{{ t('protectedConfigurationHelp') }}</p>
              </div>
              <div class="form-field flex flex-col gap-2 mb-6">
                <label for="agentType">{{ t('agentType') }}</label>
                <SelectButton
                  id="agentType"
                  :model-value="agent.agentType"
                  :options="AGENT_TYPE_OPTIONS"
                  option-label="label"
                  option-value="value"
                  :allow-empty="false"
                  @update:model-value="onUpdateAgentType"
                  class="w-fit"
                />
                <p class="text-content-muted text-sm tool-message whitespace-pre-line">{{ t('agentTypeHelp') }}</p>
              </div>
              <div class="form-field flex flex-col gap-1">
                <label for="recursionLimit">{{ t('recursionLimit') }}</label>
                <SelectButton
                  id="recursionLimit"
                  :model-value="agent.recursionLimit"
                  :options="RECURSION_LIMIT_OPTIONS.map(v => ({ label: String(v), value: v }))"
                  option-label="label"
                  option-value="value"
                  :allow-empty="false"
                  @update:model-value="onUpdateAdvancedSettings"
                  class="w-fit"
                />
                <p class="text-content-muted text-sm tool-message whitespace-pre-line">{{ t('recursionLimitHelpLine1') }}<br><strong>{{ t('recursionLimitHelpLine2') }}</strong></p>
              </div>
            </AgentCollapsibleSection>
          </div>
        </TabPanel>
        <TabPanel value="1" v-if="!loadingTests" class="flex flex-col w-full h-full">
          <AgentTestcases
            v-if="editingTestcase"
            :agent="agent"
            @run-test-case="(testCaseThreadId: number) => $emit('runSingleTest', testCaseThreadId)"
            @run-tests="$emit('runTests')"
            @show-past-executions="showPastExecutions = true"
            @test-case-deleted="onTestCaseDeleted"
          />
          <AgentTestcaseResults v-else
            @close-results="$emit('editingTestcase', true)"
          />
        </TabPanel>
      </TabPanels>
    </FlexCard>
    <div v-if="agent && !isLoading && activeTab === '0'" class="sticky bottom-0 z-10 flex items-center justify-center gap-2 px-4 py-3 text-sm border-t bg-content dark:bg-surface-muted text-surface dark:text-content rounded-b-2xl">
      <IconDeviceFloppy :size="18" :class="{ 'animate-pulse': isSaving }" />
      <span class="font-semibold">{{ isSaving ? t('savingChanges') : t('autoSaveMessage') }}</span>
      <template v-if="!isSaving && formattedSavedTime">
        <span class="text-surface dark:text-content">|</span>
        <span class="text-surface-muted dark:text-content-muted">{{ t(isOlderThanOneDay ? 'savedOn' : 'savedAt', { time: formattedSavedTime }) }}</span>
      </template>
    </div>
  </Tabs>

  <Dialog v-model:visible="showShareConfirmation"
    :header="t(isSelectedPublicTeam ? agent?.team?.id ? 'changeTeamConfirmationTitle' : 'shareConfirmationTitle' : 'unshareConfirmationTitle', { oldTeam: agent?.team?.name, newTeam: findTeam(selectedTeam!)?.name })"
    :modal="true" :draggable="false" :resizable="false" :closable="false" class="max-w-200">
    <div class="flex flex-col gap-2">
      <div class="flex flex-row gap-2 items-start whitespace-pre-line">
        <IconAlertTriangleFilled color="var(--color-warn)" v-if="invalidAttrs" />
        {{ t(shareDialogTranslationKey, shareDialogTranslationParams) }}
      </div>
      <div v-if="!invalidAttrs && isSelectedPublicTeam && privatePromptsCount > 0"
        class="flex items-center gap-2 py-4 mt-3 border-t-1">
        <div class="flex flex-row gap-2 items-start">
          <ToggleSwitch v-model="publishPrompts" />
          <div class="flex flex-col">
            <label for="publish-prompts"> {{ t('publishPrompts', { count: privatePromptsCount }) }} </label>
            <label class="text-sm text-content-muted">{{ t('publishPromptsDescription') }}</label>
          </div>
        </div>
      </div>
      <div class="flex flex-row gap-2 justify-end">
        <SimpleButton @click="onCancelChangeTeam" shape="square" variant="secondary">{{ t('cancel') }}</SimpleButton>
        <SimpleButton @click="onConfirmChangeTeam" v-if="!invalidAttrs" variant="primary" shape="square">{{ t(
          invalidAttrs ? 'backToEdit' : isSelectedPublicTeam ? agent?.team?.id ? 'changeTeam' : 'publish' : 'unpublish')
        }}</SimpleButton>
      </div>
    </div>
  </Dialog>
  <AgentImportDialog v-model:visible="showImportAgent" :result="importResult" :import-error="importError" @import="onImportAgent" />
  <AgentGenerateDialog v-model:visible="showGenerateDialog" :field="selectedField!" @generate="onGenerate" />
  <AgentPastExecutionsDialog v-if="agent" v-model:visible="showPastExecutions" :agent-id="agent.id" @select-execution="onSelectExecution" />
</template>

<i18n lang="json">
{
  "en": {
    "nameLabel": "Name",
    "basicInfoSectionTitle": "Basic info",
    "behaviorSectionTitle": "Behavior",
    "protectedSettingsBadge": "Protected settings",
    "descriptionLabel": "Description",
    "systemPromptLabel": "Instructions",
    "namePlaceholder": "e.g. Support assistant",
    "descriptionPlaceholder": "What does this agent do?",
    "nameDescription": "This is the name people will see when using the agent.",
    "descriptionDescription": "Briefly describe what this agent can help with. People will see this before using it.",
    "systemPromptPlaceholder": "Write the instructions for this agent",
    "systemPromptDescription": "Define the agent’s behavior, response style, and decision-making guidelines, including how it should leverage available tools and interpret context.",
    "private": "Private",
    "optional": "Optional",
    "shareConfirmationTitle": "Do you want to make this agent public?",
    "shareInvalidAttrs": "To share the agent you need to specify {invalidAttrs}.",
    "unshareInvalidAttrs": "To make this agent private you need to specify {invalidAttrs}.",
    "shareMissingName": "a name",
    "shareNameTooLong": "a shorter name",
    "shareMissingAnd": "and",
    "shareMissingDescription": "a description",
    "shareDescriptionTooLong": "a shorter description",
    "shareConfirmationMessageGlobal": "When you make an agent public it will be visible to everyone in the home page of Tero, so we can all benefit from what you have created.\n\nAdditionally, all future modifications to the agent will be immediately available to the rest of the users.",
    "shareConfirmationMessageTeam": "When you make an agent public it will be visible to everyone in the {team} team, so they can benefit from what you have created.\n\nAdditionally, all future modifications to the agent will be immediately available to the rest of the users.",
    "unshareConfirmationTitle": "Do you want to make this agent private?",
    "unshareConfirmationMessage": "When you make an agent private it will no longer be visible, but the users that have already used it will keep being able to use it.\n\nAdditionally, future modifications to the agent will still be available to these users.",
    "unpublish": "Make private",
    "publish": "Publish",
    "backToEdit": "Back to edit",
    "cancel": "Cancel",
    "publishPrompts": "Do you want to publish your {count} private prompts?",
    "publishPromptsDescription": "If you want to review your prompts go to the chats prompts panel",
    "visibilityLabel": "Who can see it?",
    "global": "Global",
    "changeTeamConfirmationTitle": "Change the team of this agent from {oldTeam} to {newTeam}?",
    "changeTeamConfirmationMessage": "When you change the team of an agent only the users in the new team will be able to use it.",
    "changeTeam": "Change team",
    "editAgentTabTitle": "Edit",
    "testsTabTitle": "Tests",
    "exportAgent": "Export",
    "importAgent": "Import",
    "testSpec": "Test case specification",
    "advancedSettings": "Advanced settings",
    "protectedConfiguration": "Protected configuration",
    "protectedConfigurationHelp": "When enabled, others can’t view, clone, or modify this agent’s configuration.",
    "protected": "Protected",
    "agentType": "Agent mode",
    "agentTypeBasicAgent": "Basic",
    "agentTypeAdvancedAgent": "Advanced",
    "agentTypeHelp": "Basic — faster responses for chat, drafting, and simple tasks. Best default for most agents.\nAdvanced — for complex multi-step work: offloads large context to files, summarizes long conversations, and tracks subtasks; useful when the agent relies heavily on tools or long-running workflows.",
    "recursionLimit": "Thought process steps limit",
    "recursionLimitHelpLine1": "Higher values increase budget usage and response time.",
    "recursionLimitHelpLine2": "Before increasing, optimize your agent to use fewer steps.",
    "autoSaveMessage": "Changes are saved automatically",
    "savingChanges": "Saving changes...",
    "savedAt": "Saved at {time}",
    "savedOn": "Saved on {time}"
  },
  "es": {
    "nameLabel": "Nombre",
    "basicInfoSectionTitle": "Información básica",
    "behaviorSectionTitle": "Comportamiento",
    "protectedSettingsBadge": "Configuración protegida",
    "descriptionLabel": "Descripción",
    "systemPromptLabel": "Instrucciones",
    "namePlaceholder": "ej: Asistente de soporte",
    "nameDescription": "Este es el nombre que las personas verán cuando usen el agente.",
    "descriptionPlaceholder": "¿Qué hace este agente?",
    "descriptionDescription": "Describe brevemente en qué puede ayudar este agente. Las personas lo verán antes de usarlo.",
    "systemPromptPlaceholder": "Escribe las instrucciones de este agente",
    "systemPromptDescription": "Define el comportamiento del agente, su estilo de respuesta y sus criterios de toma de decisiones, incluyendo cómo debe aprovechar las herramientas disponibles e interpretar el contexto.",
    "private": "Privado",
    "optional": "Opcional",
    "shareConfirmationTitle": "¿Quieres hacer este agente público?",
    "shareConfirmationMessageGlobal": "Cuando haces un agente público, este será visible en la página de inicio de Tero para que todos podamos beneficiarnos de lo que has creado.\n\nAdemás, todas las modificaciones futuras al agente estarán disponibles inmediatamente para el resto de sus usuarios.",
    "shareConfirmationMessageTeam": "Cuando haces un agente público, este será visible para los miembros del equipo {team}, para que ellos puedan beneficiarse de lo que has creado.\n\nAdemás, todas las modificaciones futuras al agente estarán disponibles inmediatamente para el resto de sus usuarios.",
    "shareInvalidAttrs": "Para compartir el agente, primero necesitas especificar {invalidAttrs}.",
    "unshareInvalidAttrs": "Para hacer este agente privado, primero necesitas especificar {invalidAttrs}.",
    "shareMissingName": "un nombre",
    "shareNameTooLong": "un nombre más corto",
    "shareMissingAnd": "y",
    "shareMissingDescription": "una descripción",
    "shareDescriptionTooLong": "una descripción más corta",
    "unshareConfirmationTitle": "¿Quieres hacer este agente privado?",
    "unshareConfirmationMessage": "Cuando haces un agente privado ya no será visible, pero los usuarios que ya lo han utilizado seguirán siendo capaces de usarlo.\n\nAdemás, las modificaciones futuras al agente seguirán estando disponibles para estos usuarios.",
    "unpublish": "Hacer privado",
    "publish": "Publicar",
    "backToEdit": "Volver a editar",
    "cancel": "Cancelar",
    "publishPrompts": "¿Quieres publicar tus {count} prompts privados?",
    "publishPromptsDescription": "Si quieres revisar tus prompts ve al panel de prompts en el chat",
    "visibilityLabel": "¿Quién puede verlo?",
    "global": "Global",
    "changeTeamConfirmationTitle": "¿Quieres cambiar el equipo de este agente de {oldTeam} a {newTeam}?",
    "changeTeamConfirmationMessage": "Cuando cambias el equipo de un agente, solo lo podrán usar los usuarios del nuevo equipo.",
    "changeTeam": "Cambiar equipo",
    "editAgentTabTitle": "Editar",
    "testsTabTitle": "Tests",
    "exportAgent": "Exportar",
    "importAgent": "Importar",
    "testSpec": "Especificación del test case",
    "advancedSettings": "Configuración avanzada",
    "protectedConfiguration": "Configuración protegida",
    "protectedConfigurationHelp": "Cuando está habilitada, otros no pueden ver, clonar ni modificar la configuración de este agente.",
    "protected": "Protegido",
    "agentType": "Modo del agente",
    "agentTypeBasicAgent": "Básico",
    "agentTypeAdvancedAgent": "Avanzado",
    "agentTypeHelp": "Básico — respuestas más rápidas para chat, redacción y tareas simples. Mejor opción por defecto para la mayoría de los agentes.\nAvanzado — para trabajo complejo en varios pasos: transfiere contexto extenso a archivos, resume conversaciones largas y rastrea subtareas; útil cuando el agente usa muchas herramientas o flujos prolongados.",
    "recursionLimit": "Límite de pasos del proceso de pensamiento",
    "recursionLimitHelpLine1": "Valores más altos aumentan el uso de presupuesto y el tiempo de respuesta.",
    "recursionLimitHelpLine2": "Antes de aumentar, optimiza tu agente para usar menos pasos.",
    "autoSaveMessage": "Los cambios se guardan automáticamente",
    "savingChanges": "Guardando cambios...",
    "savedAt": "Guardado a las {time}",
    "savedOn": "Guardado el {time}"
  }
}
</i18n>

<style scoped>
@import '@/assets/styles.css';

:deep(.p-inputnumber) .p-inputtext {
  @apply pr-9
}

:deep(.p-inputnumber.loading) .p-inputtext {
  @apply animate-glowing
}

</style>
