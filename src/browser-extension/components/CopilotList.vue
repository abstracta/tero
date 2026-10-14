<script lang="ts" setup>
import { onBeforeMount, ref, computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { useToast } from 'vue-toastification'
import { IconPlus, IconLoader } from '@tabler/icons-vue'
import { findAllAgents, removeAgent, updateAgents, removeAgentsByHubUrl } from '~/utils/agent-repository'
import { Agent, AgentType, AgentSource } from '~/utils/agent'
import { AuthService } from '~/utils/auth'

interface HubGroup {
  url: string
  name: string
  agents: Agent[]
  isExpanded: boolean
}

defineEmits<{
  (e: 'close'): void
  (e: 'activateAgent', agentId: string): void
}>();
const { t } = useI18n()
const toast = useToast()
const agents = ref<Agent[]>()
const showAddCopilot = ref(false)
const deletingIndex = ref(-1)
const isLoading = ref(false)
const hubStates = ref<Map<string, boolean>>(new Map())
const deletingHubUrl = ref<string | null>(null)

const groupedAgents = computed(() => {
  if (!agents.value) return { hubs: new Map<string, HubGroup>(), standalone: [] as Agent[] }
  
  const hubs = new Map<string, HubGroup>()
  const standalone: Agent[] = []
  
  agents.value.forEach(agent => {
    if (agent.type === AgentType.TeroAgent) {
      const hubUrl = agent.url
      if (!hubs.has(hubUrl)) {
        const isExpanded = hubStates.value.get(hubUrl) ?? true
        const hubGroup: HubGroup = {
          url: hubUrl,
          name: getHubName(hubUrl),
          agents: [],
          isExpanded
        }
        hubs.set(hubUrl, hubGroup)
      }
      hubs.get(hubUrl)!.agents.push(agent)
    } else {
      standalone.push(agent)
    }
  })
  
  return { hubs, standalone }
})

const getHubName = (url: string): string => {
  try {
    return new URL(url).hostname
  } catch {
    return url
  }
}

onBeforeMount(async () => {
  agents.value = await findAllAgents()

  // this logic avoid initial empty list of agents in development environment due to asynchronous loading of agents 
  // and this component being mounted before agents are loaded
  if (agents.value?.length === 0 && import.meta.env.DEV) {
    loadDevAgents()
  }
});

const loadDevAgents = () => {
  const startTime = Date.now()
  const timeoutMillis = 30000
  const pollInterval = setInterval(async () => {
    try {
      agents.value = await findAllAgents()
      if (agents.value.length > 0 || (Date.now() - startTime) >= timeoutMillis) {
        clearInterval(pollInterval)
      }
    } catch (e) {
      console.error('Error loading agents', e)
      clearInterval(pollInterval)
    }
  }, 500)
}

const removeCopilot = (index: number) => {
  deletingIndex.value = index
};

const closeDeletionConfirmation = () => {
  deletingIndex.value = -1
};

const confirmRemoval = async () => {
  let agent = agents.value![deletingIndex.value]
  await removeAgent(agent.manifest.id!)
  await agent.tearDown()
  agents.value!.splice(deletingIndex.value, 1)
  closeDeletionConfirmation()
};

const onCopilotAdded = (agent: Agent) => {
  agents.value!.push(agent)
  showAddCopilot.value = false
};

const cleanupAgents = async (agentsToCleanup: Agent[]) => {
  for (const agent of agentsToCleanup) {
    await agent.tearDown()
  }
}

const clearHubAuth = async (hubAgents: Agent[]) => {
  const auth = hubAgents[0]?.manifest.auth
  if (auth) {
    await new AuthService(auth).clearAuth()
  }
}

const refreshHub = async (hubUrl: string) => {
  isLoading.value = true
  let existingAgents: Agent[] = []
  try {
    existingAgents = await findAllAgents()
    const hubAgents = existingAgents.filter(a => a.type === AgentType.TeroAgent && a.url === hubUrl)
    await cleanupAgents(hubAgents)
    await removeAgentsByHubUrl(hubUrl)
    await AgentSource.loadAgentsFromUrl(hubUrl)
    agents.value = await findAllAgents()
  } catch (e: any) {
    agents.value = existingAgents
    await updateAgents(existingAgents)
    toast.error(t('refreshHubError'))
  } finally {
    isLoading.value = false
  }
};

const removeHub = (hubUrl: string) => {
  deletingHubUrl.value = hubUrl
};

const closeHubDeletionConfirmation = () => {
  deletingHubUrl.value = null
};

const confirmHubRemoval = async () => {
  if (!deletingHubUrl.value) return
  try {
    const existingAgents = await findAllAgents()
    const hubAgents = existingAgents.filter(a => a.type === AgentType.TeroAgent && a.url === deletingHubUrl.value)
    await cleanupAgents(hubAgents)
    await removeAgentsByHubUrl(deletingHubUrl.value)
    await clearHubAuth(hubAgents)
    agents.value = await findAllAgents()
    closeHubDeletionConfirmation()
  } catch (e: any) {
    agents.value = await findAllAgents()
    toast.error(t('removeHubError'))
  }
};

const refreshStandaloneAgent = async (agentId: string) => {
  const existingAgents = await findAllAgents()
  const existingAgent = existingAgents.find(a => a.manifest.id === agentId)
  if (!existingAgent || existingAgent.type !== AgentType.StandaloneAgent) return
  await refreshHub(existingAgent.url)
};

const removeStandaloneAgent = (agentId: string) => {
  const index = agents.value!.findIndex(a => a.manifest.id === agentId)
  removeCopilot(index)
};

const setHubState = (hubUrl: string, value: boolean) => {
  hubStates.value.set(hubUrl, value)
}

const getHubState = (hubUrl: string): boolean => {
  return hubStates.value.get(hubUrl) ?? true
}
</script>

<template>
  <PageOverlay>
    <template v-slot:headerActions>
      <BtnClose @click="$emit('close')" />
    </template>
    <template v-slot:content>
      <div v-if="isLoading" class="flex justify-center items-center w-full h-full">
        <IconLoader class="animate-spin text-violet-600" />
      </div>
      <div v-if="!isLoading" class="space-y-2">
        <div v-if="groupedAgents.hubs.size === 0 && groupedAgents.standalone.length === 0" class="flex justify-center items-center w-full h-full py-8">
          <span class="text-content-muted text-center whitespace-pre-line">{{ t('noAgents') }}</span>
        </div>
        <HubSection 
          v-for="[hubUrl, hub] in groupedAgents.hubs" 
          :key="hubUrl" 
          :title="hub.name"
          :default-expanded="hub.isExpanded"
          @update:expanded="(value: boolean) => setHubState(hubUrl, value)">
          <template #actions>
            <HubActionsMenu :hub-url="hubUrl" @refresh="refreshHub(hubUrl)" @remove="removeHub(hubUrl)" />
          </template>
          <div v-for="agent in hub.agents" :key="agent.manifest.id" class="flex flex-row py-3">
            <CopilotItem :agent="agent" @activate="$emit('activateAgent', $event)" />
          </div>
        </HubSection>
        <HubSection
          v-if="groupedAgents.standalone.length > 0"
          :title="t('standalone')"
          :default-expanded="getHubState('standalone')"
          @update:expanded="(value: boolean) => setHubState('standalone', value)">
          <div v-for="agent in groupedAgents.standalone" :key="agent.manifest.id" class="flex flex-row py-3">
            <CopilotItem :agent="agent" @activate="$emit('activateAgent', $event)" />
            <HubActionsMenu type="agent" @refresh="refreshStandaloneAgent(agent.manifest.id!)" @remove="removeStandaloneAgent(agent.manifest.id!)" />
          </div>
        </HubSection>
      </div>
    </template>
    <template v-slot:modalsContainer>
      <ModalForm :title="t('removeTitle')" :show="deletingIndex >= 0" @close="closeDeletionConfirmation" @save="confirmRemoval" :button-text="t('removeButton')">
        {{
          t('removeConfirmation', {
            agentName: agents ? agents[deletingIndex].manifest.name : ''
          })
        }}
      </ModalForm>
      <ModalForm :title="t('removeHubTitle')" :show="deletingHubUrl !== null" @close="closeHubDeletionConfirmation" @save="confirmHubRemoval" :button-text="t('removeButton')">
        {{
          t('removeHubConfirmation', {
            hubName: deletingHubUrl ? getHubName(deletingHubUrl) : ''
          })
        }}
      </ModalForm>
      <AddCopilotModal :show="showAddCopilot" @close="showAddCopilot = false" @saved="onCopilotAdded" />
    </template>
    <template v-slot:footer>
      <div v-if="!isLoading" class="flex flex-row p-3">
         <SimpleButton @click="showAddCopilot = true" shape="square" size="small" class="gap-2 px-3">
          <IconPlus size="16" />
          <span>{{ t('addTitle') }}</span>
        </SimpleButton>
      </div>
    </template>
  </PageOverlay>
</template>

<i18n lang="json">
{
  "en": {
    "addTitle": "Add",
    "refreshHubError": "Error refreshing the Tero instance",
    "removeHubError": "Error removing the Tero instance",
    "removeTitle": "Remove copilot",
    "removeButton": "Remove",
    "removeConfirmation": "Are you sure you want to remove the copilot {agentName}?",
    "refreshAgentError": "Error refreshing agent",
    "standalone": "Standalone",
    "noAgents": "No agents available\nClick Add to add a new agent",
    "removeHubTitle": "Remove Tero instance",
    "removeHubConfirmation": "Are you sure you want to remove the Tero instance {hubName} and all its agents?"
  },
  "es" : {
    "addTitle": "Agregar",
    "refreshHubError": "Error al actualizar la instancia de Tero",
    "removeHubError": "Error al eliminar la instancia de Tero",
    "removeTitle": "Quitar copiloto",
    "removeButton": "Quitar",
    "removeConfirmation": "¿Estás seguro de quitar el copiloto {agentName}?",
    "refreshAgentError": "Error al actualizar agente",
    "standalone": "Independientes",
    "noAgents": "No hay agentes disponibles\nHaz clic en Agregar para agregar un nuevo agente",
    "removeHubTitle": "Quitar instancia de Tero",
    "removeHubConfirmation": "¿Estás seguro de quitar la instancia de Tero {hubName} y todos sus agentes?"
  }
}
</i18n>
