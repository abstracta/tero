import { ref, computed } from 'vue';

const groupedViewStorageKey = 'tero.ui.sidebar.groupChatsByAgent';
const collapsedGroupedAgentsStorageKey = 'tero.ui.sidebar.collapsedGroupedAgents';

const readInitialGroupedView = () => localStorage.getItem(groupedViewStorageKey) !== 'false';
const readInitialCollapsedGroupedAgents = () => {
  try {
    return new Set<number>(JSON.parse(localStorage.getItem(collapsedGroupedAgentsStorageKey) ?? '[]'));
  } catch {
    return new Set<number>();
  }
};

const isSidebarCollapsed = ref(false);
const groupedViewEnabled = ref(readInitialGroupedView());
const collapsedGroupedAgentIds = ref(readInitialCollapsedGroupedAgents());

export function useSidebar() {
  const toggleSidebar = () => {
    isSidebarCollapsed.value = !isSidebarCollapsed.value;
  };

  const setGroupedViewEnabled = (enabled: boolean) => {
    groupedViewEnabled.value = enabled;
    localStorage.setItem(groupedViewStorageKey, String(enabled));
  };

  const toggleGroupedAgent = (agentId: number) => {
    const collapsedIds = new Set(collapsedGroupedAgentIds.value);
    if (collapsedIds.has(agentId)) collapsedIds.delete(agentId);
    else collapsedIds.add(agentId);
    collapsedGroupedAgentIds.value = collapsedIds;
    localStorage.setItem(collapsedGroupedAgentsStorageKey, JSON.stringify([...collapsedIds]));
  };

  return {
    isSidebarCollapsed,
    toggleSidebar,
    isGroupedViewEnabled: computed(() => groupedViewEnabled.value),
    setGroupedViewEnabled,
    collapsedGroupedAgentIds: computed(() => collapsedGroupedAgentIds.value),
    toggleGroupedAgent,
  };
}
