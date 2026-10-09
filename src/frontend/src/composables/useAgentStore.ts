import { reactive } from 'vue'
import { useRouter } from 'vue-router'
import { ApiService, Agent } from '@/services/api'

export const agentsPageSize = 20

const agentsStore = reactive({
  agents: [] as Agent[],
  currentAgent: {} as Agent,
  hasMoreAgents: false,
  agentsRevision: 0,
  // one extra agent is loaded per page to know if there's more, and duplicates are skipped on append since a recent chat can move an agent to a newer page
  setAgents(page: Agent[], append = false) {
    const agents = page.slice(0, agentsPageSize)
    if (append) this.agents.push(...agents.filter((agent) => !this.agents.some((a) => a.id === agent.id)))
    else this.agents = agents
    this.hasMoreAgents = page.length > agentsPageSize
  },
  addAgent(agent: Agent) {
    this.agents.unshift(agent)
  },
  removeAgent(agentId: number) {
    this.agents = this.agents.filter((a) => a.id !== agentId)
  },
  updateAgent(agentId: number) {
    // agents might not be loaded yet if we are editing an agent and didn't go through the main page
    if (!this.agents.length) {
      return
    }
    const agent = this.agents.find((a) => a.id === agentId)!
    // if the agent has been removed then just don't update it
    if (!agent) {
      return
    }
    // we remove it and re add it so it is shown at top of list
    this.removeAgent(agentId)
    this.addAgent(agent)
  },
  setCurrentAgent(agent: Agent) {
    this.currentAgent = agent
  }
})

export function useAgentStore() {
  const api = new ApiService()
  const router = useRouter()

  async function newAgent() {
    const agent = await api.newAgent()
    agentsStore.addAgent(agent)
    router.push(`/agents/${agent.id}`)
  }

  async function configureAgent(agentId: number) {
    await router.push(`/agents/${agentId}`)
  }

  async function addAgent(agent: Agent) {
    if (!agentsStore.agents.find((a) => a.id === agent.id)) {
      await api.addUserAgent(agent.id)
      // sidebar watches this and reloads, so a pin shows up with its chats in either view, including when the list was empty
      agentsStore.agentsRevision++
    }
  }

  async function removeAgent(agentId: number) {
    await api.removeUserAgent(agentId)
    agentsStore.removeAgent(agentId)
  }

  async function loadAgents(append = false) {
    agentsStore.setAgents(await api.findUserAgents('', agentsPageSize + 1, append ? agentsStore.agents.length : 0), append)
  }

  function updateAgent(agentId: number) {
    agentsStore.updateAgent(agentId)
  }

  function setCurrentAgent(agent: Agent) {
    agentsStore.setCurrentAgent(agent)
  }

  async function addDefaultAgent() {
    const agent = await api.findDefaultAgent();
    await addAgent(agent);
    return agent;
  }

  async function cloneAgent(agentId: number) {
    const agent = await api.cloneAgent(agentId);
    await router.push(`/agents/${agent.id}`)
  }

  return { agentsStore, loadAgents, addAgent, removeAgent, updateAgent, newAgent, configureAgent, setCurrentAgent, addDefaultAgent, cloneAgent }
}
