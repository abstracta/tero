<script lang="ts" setup>
import { onMounted, ref, computed } from 'vue';
import { IconChevronDown, IconChevronUp } from '@tabler/icons-vue';
import { useRouter } from 'vue-router';
import { useI18n } from 'vue-i18n';
import { useChatStore } from '@/composables/useChatStore';
import { useAgentStore } from '@/composables/useAgentStore';
import { useErrorHandler } from '@/composables/useErrorHandler';
import SidebarSkeleton from './SidebarSkeleton.vue';
import SidebarSearch from './SidebarSearch.vue';
import type { Thread, Agent } from '@/services/api';
import { useSidebar } from '@/composables/useSidebar';
import { ApiService } from '@/services/api';

const { t } = useI18n();
const router = useRouter();

const { chatsStore, loadChats } = useChatStore();
const { agentsStore, loadAgents } = useAgentStore();
const { newAgent } = useAgentStore();
const { handleError } = useErrorHandler();
const { isSidebarCollapsed } = useSidebar();

const showingSearchInput = ref(false);
const isLoading = ref(true);
const searchQuery = ref('');
const searchResults = ref<{ agents: Agent[], chats: Thread[] }>({ agents: [], chats: [] });
const sidebarSearchRef = ref<InstanceType<typeof SidebarSearch> | null>(null);
const displayedAgents = computed(() => filterModeIsActive.value && !sidebarSearchRef.value?.isSearching ? searchResults.value.agents : agentsStore.agents);
const displayedChats = computed(() => filterModeIsActive.value && !sidebarSearchRef.value?.isSearching ? searchResults.value.chats : chatsStore.chats);
const agentsCollapsed = ref(false);
const chatsCollapsed = ref(false);

const onNewAgent = async () => {
  try {
    await newAgent();
  } catch (e) {
    handleError(e);
  }
}

const toggleSidebar = () => {
  isSidebarCollapsed.value = !isSidebarCollapsed.value;
};

const logout = () => {
  router.push('/logout');
}

const filterModeIsActive = computed(() => {
  return showingSearchInput.value && searchQuery.value;
})

const handleSearchClick = () => {
  if (filterModeIsActive.value) {
    sidebarSearchRef.value?.clearSearch();
  }
  showingSearchInput.value = !showingSearchInput.value;
};

const newChat = () => {
  router.push('/');
}

const defaultAgentName = ref<string>('');
const api = new ApiService();

const newChatTooltip = computed(() => t('startNewChatWith', { name: defaultAgentName.value }));

onMounted(async () => {
  try {
    isLoading.value = true;
    await loadAgents();
    await loadChats();
    const agent = await api.findDefaultAgent();
    if (agent && agent.name) defaultAgentName.value = agent.name;
  } catch (error) {
    handleError(error);
  } finally {
    isLoading.value = false;
  }
})
</script>

<template>
  <SidebarSkeleton v-if="isLoading" :is-collapsed="isSidebarCollapsed" />
  <FlexCard v-else id="sidebar" :class="[isSidebarCollapsed ? '!w-16 sm:!w-20 px-1 sm:px-2' : 'w-xs px-2 sm:px-3']">
    <template #header>
      <div class="flex items-center" :class="isSidebarCollapsed ? 'flex-col' : 'justify-between'">
        <SimpleButton :class="{ 'mr-2 sm:mr-3': !isSidebarCollapsed }" @click="toggleSidebar" style="transition: all 0.3s ease;">
          <iconLayoutSidebarRightCollapse v-if="isSidebarCollapsed" />
          <iconLayoutSidebarLeftCollapse v-else />
        </SimpleButton>
        <div v-if="!isSidebarCollapsed" class="flex items-center gap-2">
          <SimpleButton shape="rounded" @click="onNewAgent">
            <IconPlus />
            <span>
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

    <div class="flex-1 overflow-y-auto">
      <SidebarSearch
        v-if="showingSearchInput"
        v-model:showingSearchInput="showingSearchInput"
        v-model:searchQuery="searchQuery"
        @search-results="searchResults = $event"
        ref="sidebarSearchRef"/>

      <div class="border-b border-pale py-1.5 mb-1">
        <SidebarDashboardItem />
        <SidebarDiscoverItem />
      </div>

      <div class="sticky top-0 bg-white z-10">
        <div class="flex justify-between items-center p-2">
          <h3 class="flex items-center gap-2">
            {{ t('agents') }}
            <button class="p-1" @click="agentsCollapsed = !agentsCollapsed">
              <component :is="agentsCollapsed ? IconChevronDown : IconChevronUp" class="w-5 h-5" />
            </button>
          </h3>
        </div>
      </div>

      <div v-if="!agentsCollapsed">
        <SidebarAgent v-for="agent in displayedAgents" :key="agent.id" :agent="agent" />
        <div v-if="displayedAgents.length === 0" class="justify-left p-2 text-light-gray">
          {{ t(filterModeIsActive ? 'noAgentsFound' : 'noAgents')}}
        </div>
      </div>

      <div v-if="!isSidebarCollapsed" class="mt-5">
        <div class="sticky top-0 bg-white z-10">
            <div class="flex justify-between items-center p-2">
              <h3 class="flex items-center gap-2">
                {{ t('chats') }}
                <button class="p-1" @click="chatsCollapsed = !chatsCollapsed">
                  <component :is="chatsCollapsed ? IconChevronDown : IconChevronUp" class="w-5 h-5" />
                </button>
              </h3>
              <SimpleButton
                size="small"
                @click="newChat"
                v-tooltip.bottom="newChatTooltip"
                class="hover:text-primary! hover:bg-transparent! border-0! ring-0! outline-0! shadow-none! gap-0.5!"
              >
                <IconPlus size="12"  class="font-medium"/>
                <p class="underline underline-offset-2 font-medium">{{ t('newChat') }}</p>
              </SimpleButton>
            </div>
          </div>

        <div v-if="!chatsCollapsed">
          <SidebarChat v-for="chat in displayedChats" :key="chat.id" :chat="chat" />
          <div v-if="displayedChats.length === 0" class="justify-left p-2 text-light-gray">
            {{ t(filterModeIsActive ? 'noChatsFound' : 'noChats') }}
          </div>
        </div>
        
      </div>
    </div>

    <div class="sticky bottom-0 border-t border-auxiliar-gray">
      <div v-if="!isSidebarCollapsed" class="px-2 sm:px-3 pt-3">
        <ConsumedBudget />
      </div>
      <div :class="isSidebarCollapsed ? 'flex justify-center' : ''" class="py-3">
        <SimpleButton @click="logout" :size="isSidebarCollapsed ? 'default' : 'large'" v-tooltip.bottom="t(isSidebarCollapsed ? 'logout' : '')" class="w-full">
          <IconLogout />
          <span class="mx-1" v-if="!isSidebarCollapsed">{{ t('logout') }}</span>
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
      "startNewChatWith": "Start new chat with {name}",
      "expandAgents": "Expand agents",
      "collapseAgents": "Collapse agents",
      "chats": "My chats",
      "logout": "Log out",
      "noChats": "Nothing to show here yet",
      "noChatsFound": "No chats found matching your search",
      "noAgentsFound": "No agents found matching your search",
      "noAgents": "No agents available",
      "startSearch": "Click to start search",
      "cancelSearch": "Click to cancel search",
      "loadMore": "Load more",
      "newChat": "New chat"
    },
    "es": {
      "createAgent": "Crear agente",
      "agents": "Agentes",
      "startNewChatWith": "Iniciar nuevo chat con {name}",
      "expandAgents": "Expandir agentes",
      "collapseAgents": "Contraer agentes",
      "chats": "Mis chats",
      "logout": "Cerrar sesión",
      "noChats": "No hay nada para mostrar",
      "noChatsFound": "Ningún chat coincide con tu búsqueda",
      "noAgentsFound": "Ningún agente coincide con tu búsqueda",
      "noAgents": "No hay agentes disponibles",
      "startSearch": "Haz clic para empezar a buscar",
      "cancelSearch": "Haz clic para cancelar la búsqueda",
      "loadMore": "Cargar más",
      "newChat": "Nuevo chat"
    }

  }
</i18n>