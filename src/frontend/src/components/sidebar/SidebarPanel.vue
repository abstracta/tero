<script lang="ts" setup>
  import { onMounted, ref, computed, watch } from 'vue';
  import { useIntersectionObserver } from '@vueuse/core';
  import { IconChevronDown, IconChevronUp, IconPlus, IconSearch, IconLogout, IconLayoutSidebarLeftExpand, IconLayoutSidebarLeftCollapse } from '@tabler/icons-vue';
  import { useRouter } from 'vue-router';
  import { useI18n } from 'vue-i18n';
  import { useChatStore } from '@/composables/useChatStore';
  import { useAgentStore } from '@/composables/useAgentStore';
  import { useErrorHandler } from '@/composables/useErrorHandler';
  import SidebarSearch from './SidebarSearch.vue';
  import { type Thread, type Agent, ApiService } from '@/services/api';
  import { useSidebar } from '@/composables/useSidebar';
  
  const { t } = useI18n();
  const router = useRouter();
  
  const { chatsStore, loadGroupedChats, refreshSidebar } = useChatStore();
  const { agentsStore, loadAgents, newAgent } = useAgentStore();
  const { handleError } = useErrorHandler();
  const { isSidebarCollapsed, isGroupedViewEnabled, toggleSidebar } = useSidebar();
  const api = new ApiService();

  const showingSearchInput = ref(false);
  const isLoading = ref(true);
  const searchQuery = ref('');
  const searchResults = ref<{ agents: Agent[], chats: Thread[] }>({ agents: [], chats: [] });
  const sidebarSearchRef = ref<InstanceType<typeof SidebarSearch> | null>(null);
  const sidebarScrollRef = ref<HTMLElement | null>(null);
  const agentListEndRef = ref<HTMLElement | null>(null);
  const agentsCollapsed = ref(false);
  const chatsCollapsed = ref(false);
  const defaultAgentName = ref<string>('');
  
  const filterModeIsActive = computed(() => showingSearchInput.value && searchQuery.value )
  const showingSearchResults = computed(() => filterModeIsActive.value && !sidebarSearchRef.value?.isSearching);
  const displayedAgents = computed(() => showingSearchResults.value ? searchResults.value.agents : agentsStore.agents);
  const displayedChats = computed(() => showingSearchResults.value ? searchResults.value.chats : chatsStore.chats);
  const agentGroups = computed(() => {
    if (!showingSearchResults.value) {
      return agentsStore.agents.map((agent) => {
        const group = chatsStore.chatsByAgent[agent.id];
        return { agent, chats: group?.chats ?? [], hasMoreChats: group?.hasMoreChats ?? false };
      });
    }
    const groups = new Map(searchResults.value.agents.map((agent) => [agent.id, { agent, chats: [] as Thread[], isSaved: true }]));
    for (const chat of searchResults.value.chats) {
      if (!groups.has(chat.agent.id)) groups.set(chat.agent.id, { agent: chat.agent, chats: [], isSaved: false });
      groups.get(chat.agent.id)!.chats.push(chat);
    }
    return [...groups.values()];
  });

  onMounted(async () => {
    try {
      const [defaultAgent] = await Promise.all([api.findDefaultAgent(), refreshSidebar()]);
      defaultAgentName.value = defaultAgent.name!;
    } catch (error) {
      handleError(error);
    } finally {
      isLoading.value = false;
    }
  });

  watch(isGroupedViewEnabled, () => refreshSidebar().catch(handleError));
  watch(() => agentsStore.agentsRevision, () => refreshSidebar().catch(handleError));

  let isLoadingMoreAgents = false;
  const loadMoreAgents = async () => {
    if (isLoadingMoreAgents) return;
    isLoadingMoreAgents = true;
    try {
      await (isGroupedViewEnabled.value ? loadGroupedChats(true) : loadAgents(true));
    } catch (error) {
      handleError(error);
    } finally {
      isLoadingMoreAgents = false;
    }
  };

  useIntersectionObserver(agentListEndRef, ([entry]) => {
    if (entry?.isIntersecting) void loadMoreAgents();
  }, { root: sidebarScrollRef, rootMargin: '0px 0px 160px 0px' });

  const handleSearchClick = () => {
    if (filterModeIsActive.value) {
      sidebarSearchRef.value?.clearSearch();
      return;
    }
    showingSearchInput.value = !showingSearchInput.value;
  };
  
  const onNewAgent = async () => {
    try {
      await newAgent();
    } catch (e) {
      handleError(e);
    }
  }
  
  const newChat = () => {
    router.push('/');
  }
  
  const logout = () => {
    router.push('/logout');
  }
  
  watch(isSidebarCollapsed, (newVal) => {
    if (newVal) {
      agentsCollapsed.value = false;
    }
  })
  </script>
  
  <template>
    <SidebarSkeleton v-if="isLoading" :is-collapsed="isSidebarCollapsed" />
    <FlexCard v-else id="sidebar" :class="['transition-[width] duration-[250ms] ease-out', isSidebarCollapsed ? '!w-16 sm:!w-20 px-1 sm:px-2' : 'w-xs px-2 sm:px-3']">
      <template #header>
        <div class="flex items-center" :class="isSidebarCollapsed ? 'flex-col' : 'justify-between'">
          <SimpleButton :class="{ 'mr-2 sm:mr-3': !isSidebarCollapsed }" @click="toggleSidebar" style="transition: all 0.3s ease;">
            <IconLayoutSidebarLeftExpand v-if="isSidebarCollapsed" />
            <IconLayoutSidebarLeftCollapse v-else />
          </SimpleButton>
          <div v-if="!isSidebarCollapsed" class="flex items-center gap-2">
            <SimpleButton shape="rounded" class="whitespace-nowrap" @click="onNewAgent">
              <IconPlus />
              <span class="whitespace-nowrap">
                {{ t('createAgent') }}
              </span>
            </SimpleButton>
            <SimpleButton
              v-tooltip.bottom="t(showingSearchInput ? 'cancelSearch' : 'startSearch')"
              @click="handleSearchClick">
              <IconSearch />
            </SimpleButton>
          </div>
        </div>
      </template>
  
      <div ref="sidebarScrollRef" class="flex-1 overflow-y-auto">
        <div
          class="grid transition-[grid-template-rows] duration-[250ms] ease-out"
          :class="showingSearchInput && !isSidebarCollapsed ? 'grid-rows-[1fr]' : 'grid-rows-[0fr]'"
          :inert="!showingSearchInput || isSidebarCollapsed">
          <div class="overflow-hidden min-h-0">
            <SidebarSearch
              v-model:showingSearchInput="showingSearchInput"
              v-model:searchQuery="searchQuery"
              @search-results="searchResults = $event"
              ref="sidebarSearchRef"/>
          </div>
        </div>
  
        <div class="border-b border-surface-muted py-1.5 mb-1">
          <SidebarDashboardItem />
          <SidebarDiscoverItem />
        </div>

        <template v-if="isGroupedViewEnabled && !isSidebarCollapsed">
          <div class="py-1.5">
            <SidebarAgent v-for="group in agentGroups" :key="group.agent.id" v-bind="group" variant="grouped" />
            <!-- key changes on every page load so this element is fresh, and the observer above notices it again even if it didn't need to move -->
            <div v-if="!filterModeIsActive && agentsStore.hasMoreAgents" :key="agentsStore.agents.length" ref="agentListEndRef" class="h-px" />
            <div v-if="agentGroups.length === 0" class="justify-left p-2 text-content-muted">
              {{ t(filterModeIsActive ? 'noGroupedResults' : 'noAgents') }}
            </div>
          </div>
        </template>
  
        <div v-if="!isSidebarCollapsed && !isGroupedViewEnabled" class="sticky top-0 bg-surface z-10 p-2">
          <div class="flex items-center gap-1 text-sm">
            <button @click="agentsCollapsed = !agentsCollapsed">
              <component :is="agentsCollapsed ? IconChevronDown : IconChevronUp" class="w-5 h-5" />
            </button>
            <span class="text-content-muted">{{ t('agents') }}</span>
          </div>
        </div>
  
        <div v-if="!isGroupedViewEnabled || isSidebarCollapsed" class="grid transition-[grid-template-rows] duration-[250ms] ease-out" :class="agentsCollapsed ? 'grid-rows-[0fr]' : 'grid-rows-[1fr]'">
          <div class="overflow-hidden min-h-0">
            <SidebarAgent v-for="agent in displayedAgents" :key="agent.id" :agent="agent" />
            <div v-if="!agentsCollapsed && !filterModeIsActive && agentsStore.hasMoreAgents" :key="agentsStore.agents.length" ref="agentListEndRef" class="h-px" />
            <div v-if="displayedAgents.length === 0" class="justify-left p-2 text-content-muted">
              {{ t(filterModeIsActive ? 'noAgentsFound' : 'noAgents')}}
            </div>
          </div>
        </div>
  
        <div v-if="!isSidebarCollapsed && !isGroupedViewEnabled" class="mt-5">
          <div class="sticky top-0 bg-surface z-10">
            <div class="flex justify-between items-center p-2">
              <div class="flex items-center gap-1 text-sm">
                <button @click="chatsCollapsed = !chatsCollapsed">
                  <component :is="chatsCollapsed ? IconChevronDown : IconChevronUp" class="w-5 h-5" />
                </button>
                <span class="text-content-muted whitespace-nowrap">{{ t('chats') }}</span>
              </div>
              <SimpleButton
                variant="light"
                size="small"
                @click="newChat"
                v-tooltip.bottom="t('newChatTooltip', { name: defaultAgentName })"
                class="hover:text-primary! hover:bg-transparent! !bg-transparent text-sm font-semibold"
              >
                <IconPlus class="w-5 h-5"/>
                <p class="underline underline-offset-2 whitespace-nowrap">{{ t('newChat') }}</p>
              </SimpleButton>
            </div>
          </div>
  
          <div class="grid transition-[grid-template-rows] duration-[250ms] ease-out" :class="chatsCollapsed ? 'grid-rows-[0fr]' : 'grid-rows-[1fr]'">
            <div class="overflow-hidden min-h-0">
              <SidebarChat v-for="chat in displayedChats" :key="chat.id" :chat="chat" />
              <div v-if="displayedChats.length === 0" class="justify-left p-2 text-content-muted">
                {{ t(filterModeIsActive ? 'noChatsFound' : 'noChats') }}
              </div>
            </div>
          </div>
  
        </div>
      </div>
  
      <div class="sticky bottom-0 border-t">
        <div v-if="!isSidebarCollapsed" class="px-2 sm:px-3 pt-3">
          <ConsumedBudget />
        </div>
        <SidebarUserMenu v-if="!isSidebarCollapsed" />
        <div v-else class="flex justify-center py-3">
          <SimpleButton @click="logout" v-tooltip.bottom="t('logout')">
            <IconLogout />
          </SimpleButton>
        </div>
      </div>
    </FlexCard>
  </template>
  
  <i18n lang="json">
    {
      "en": {
        "createAgent": "Create agent",
        "agents": "Agents",
        "newChatTooltip": "Start new chat with {name}",
        "chats": "My chats",
        "logout": "Log out",
        "noChats": "Nothing to show here yet",
        "noChatsFound": "No chats found matching your search",
        "noAgentsFound": "No agents found matching your search",
        "noGroupedResults": "No agents or chats found matching your search",
        "noAgents": "No agents available",
        "startSearch": "Click to start search",
        "cancelSearch": "Click to cancel search",
        "loadMore": "Load more",
        "newChat": "New chat"
      },
      "es": {
        "createAgent": "Crear agente",
        "agents": "Agentes",
        "newChatTooltip": "Iniciar nuevo chat con {name}",
        "chats": "Mis chats",
        "logout": "Cerrar sesión",
        "noChats": "No hay nada para mostrar",
        "noChatsFound": "Ningún chat coincide con tu búsqueda",
        "noAgentsFound": "Ningún agente coincide con tu búsqueda",
        "noGroupedResults": "Ningún agente o chat coincide con tu búsqueda",
        "noAgents": "No hay agentes disponibles",
        "startSearch": "Haz clic para empezar a buscar",
        "cancelSearch": "Haz clic para cancelar la búsqueda",
        "loadMore": "Cargar más",
        "newChat": "Nuevo chat"
      }
    }
  </i18n>
