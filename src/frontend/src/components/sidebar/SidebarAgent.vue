<script setup lang="ts">
import { ref, computed, nextTick } from 'vue';
import { useI18n } from 'vue-i18n';
import { useAgentStore } from '@/composables/useAgentStore';
import { useErrorHandler } from '@/composables/useErrorHandler';
import { useChatStore, chatsPerPage } from '@/composables/useChatStore';
import { AnimationEffect } from '../../../../common/src/utils/animations';
import { ApiService, type Agent, type Thread } from '@/services/api';
import { useSidebar } from '@/composables/useSidebar';
import { IconMessage2Plus, IconEditCircle, IconTrashX, IconCopyPlus, IconInfoCircle, IconCaretDownFilled } from '@tabler/icons-vue';

type SidebarAgentVariant = 'legacy' | 'grouped';

const props = withDefaults(defineProps<{
  agent: Agent
  chats?: Thread[]
  isSaved?: boolean
  hasMoreChats?: boolean
  variant?: SidebarAgentVariant
}>(), {
  chats: () => [],
  isSaved: true,
  hasMoreChats: false,
  variant: 'legacy',
});

const { isSidebarCollapsed, collapsedGroupedAgentIds, toggleGroupedAgent } = useSidebar();
const { t } = useI18n();
const { newChat, openChat, loadMoreChats } = useChatStore();
const { removeAgent, cloneAgent } = useAgentStore();
const { handleError } = useErrorHandler();
const { configureAgent } = useAgentStore();
const showConfirmation = ref(false);
const showAgentInfoModal = ref(false);
const isExpanded = computed(() => !collapsedGroupedAgentIds.value.has(props.agent.id));
const menu = ref();
const pageToAnimate = ref<number | null>(null);
const isLoadingMoreChats = ref(false);
const chatPages = computed(() => Array.from({ length: Math.ceil(props.chats.length / chatsPerPage) }, (_, index) =>
  props.chats.slice(index * chatsPerPage, (index + 1) * chatsPerPage)));
const api = new ApiService();

const isGrouped = computed(() => props.variant === 'grouped');
const showCaret = computed(() => isGrouped.value && !isSidebarCollapsed.value);
const canViewAgentInfo = computed(() => props.agent.canEdit || !props.agent.isProtected);

const agentName = computed(() => props.agent.name || t('unavailableAgent'));

const onNewChat = async () => {
  try {
    await newChat(props.agent);
  } catch (e) {
    handleError(e);
  }
};

const openLatestChat = async () => {
  try {
    const [latestChat] = await api.findLastChats(1, false, props.agent.id);
    if (latestChat) await openChat(latestChat.id);
  } catch (e) {
    handleError(e);
  }
};

const handleRemoveAgent = () => {
  showConfirmation.value = true;
}

const onRemoveAgent = async () => {
  try {
    await removeAgent(props.agent.id);
    showConfirmation.value = false;
  } catch (e) {
    handleError(e);
  }
}

const handleEditAgent = async () => {
  configureAgent(props.agent.id);
}

const handleCloneAgent = async () => {
  try {
    await cloneAgent(props.agent.id);
  } catch (error) {
    handleError(error);
  }
}

const handleShowAgentInfo = () => {
  showAgentInfoModal.value = !showAgentInfoModal.value;
};

const onLoadMoreChats = async () => {
  isLoadingMoreChats.value = true;
  pageToAnimate.value = chatPages.value.length;
  try {
    await loadMoreChats(props.agent.id);
  } catch (e) {
    handleError(e);
  } finally {
    isLoadingMoreChats.value = false;
  }
  // wait for the new page to render before resetting, so it plays its entrance animation once
  await nextTick();
  pageToAnimate.value = null;
};

const handleAgentClick = () => {
  if (isGrouped.value) {
    toggleGroupedAgent(props.agent.id);
    return;
  }
  if (props.agent.readOnly) openLatestChat();
  else onNewChat();
};

</script>

<template>
  <SidebarItem
    v-if="!showConfirmation"
    :is-collapsed="isSidebarCollapsed"
    @click="handleAgentClick"
    @menu-toggle="(event: Event) => menu?.toggle(event)"
    :class="isSidebarCollapsed ? 'justify-center' : ''">
    <div
      class="flex items-center"
      :class="isSidebarCollapsed ? 'justify-center w-full' : 'gap-2'"
      v-tooltip.bottom="{value: agentName, showDelay: 2000}">
      <div class="flex-shrink-0 flex items-center relative" :class="{ 'group-hover:[&>:not(:last-child)]:invisible': showCaret }">
        <AgentAvatar :agent="agent" :show-shared-status="true" />
        <IconCaretDownFilled
          v-if="showCaret"
          class="absolute inset-0 m-auto w-5 h-5 text-content-muted invisible group-hover:visible transition-transform"
          :class="{ '-rotate-90': !isExpanded }" />
      </div>
      <span v-if="!isSidebarCollapsed" class="line-clamp-1 break-all min-w-0 flex-1">
        {{ agentName }}
      </span>
    </div>
    <template #actions v-if="!isSidebarCollapsed">
      <SimpleButton
        v-if="isGrouped && !agent.readOnly"
        variant="muted"
        size="small"
        class="!p-0 !bg-transparent !text-content-muted !shadow-none !ring-0 !outline-none opacity-0 w-0 overflow-hidden transition-[width,opacity] group-hover:w-7 group-hover:opacity-100"
        :aria-label="t('newChatTooltip')"
        v-tooltip.bottom="{value: t('newChatTooltip'), showDelay: 1000, pt: {root: {style: {marginTop: '0.5rem'}}}}"
        @click.stop="onNewChat">
        <IconMessage2Plus class="w-5 h-5" />
      </SimpleButton>
      <AgentChatMenu
        ref="menu"
        :agent-team="agent.team?.name"
        :is-collapsed="isSidebarCollapsed"
        :can-view-agent-info="canViewAgentInfo"
        :class="[menu?.isMenuOpen ? 'opacity-100 w-auto' : 'opacity-0 w-0 group-hover:opacity-100 group-hover:w-auto']"
        :items="[
          ...(!isGrouped && !agent.readOnly ? [{
            label: t('newChatTooltip'),
            tablerIcon: IconMessage2Plus,
            command: onNewChat
          }] : []),
          ...(!agent.readOnly ? [{
            label: t('agentInfoTooltip'),
            tablerIcon: IconInfoCircle,
            command: handleShowAgentInfo
          }] : []),
          ...(agent.canEdit ? [{
            label: t('editAgentTooltip'),
            tablerIcon: IconEditCircle,
            command: handleEditAgent
          }] : []),
          ...(canViewAgentInfo && !agent.readOnly ? [{
            label: t('cloneAgentTooltip'),
            tablerIcon: IconCopyPlus,
            command: handleCloneAgent
          }] : []),
          ...(isSaved ? [{
            label: t('removeAgentTooltip'),
            tablerIcon: IconTrashX,
            command: handleRemoveAgent
          }] : [])]"
      />
    </template>
  </SidebarItem>

  <div
    v-if="isGrouped && !showConfirmation"
    class="grid transition-[grid-template-rows] duration-[250ms] ease-out"
    :class="isExpanded ? 'grid-rows-[1fr]' : 'grid-rows-[0fr]'">
    <div class="overflow-hidden min-h-0">
      <div class="pl-10">
        <Animate
          v-for="(page, pageIndex) in chatPages"
          :key="pageIndex"
          :effect="AnimationEffect.EXPAND_VERTICAL"
          :animate-on-mount="pageIndex === pageToAnimate">
          <SidebarChat v-for="chat in page" :key="chat.id" :chat="chat" nested />
        </Animate>
        <button
          v-if="hasMoreChats"
          class="text-content-muted hover:text-content ml-2 mt-1 py-1 text-sm font-normal"
          :disabled="isLoadingMoreChats"
          @click="onLoadMoreChats">
          {{ t('showMore') }}
        </button>
      </div>
    </div>
  </div>

  <Animate v-if="showConfirmation" :effect="AnimationEffect.QUICK_SLIDE_DOWN">
    <ItemConfirmation
      :tooltip="t('removeAgentConfirmation')"
      @confirm="onRemoveAgent"
      @cancel="() => showConfirmation = false"
    />
  </Animate>
  <DiscoverAgentInfo :agent="agent" :showModal="showAgentInfoModal" @close="handleShowAgentInfo" />
</template>

<i18n>
  {
    "en": {
      "removeAgentTooltip": "Remove agent",
      "removeAgentConfirmation": "Remove agent?",
      "newChatTooltip": "New chat",
      "editAgentTooltip": "Edit agent",
      "cloneAgentTooltip": "Clone agent",
      "agentInfoTooltip": "View details",
      "showMore": "Show more",
      "unavailableAgent": "Unavailable agent"
    },
    "es": {
      "removeAgentTooltip": "Quitar agente",
      "removeAgentConfirmation": "¿Quitar agente?",
      "newChatTooltip": "Nuevo chat",
      "editAgentTooltip": "Editar agente",
      "cloneAgentTooltip": "Clonar agente",
      "agentInfoTooltip": "Ver detalles",
      "showMore": "Ver más",
      "unavailableAgent": "Agente no disponible"
    }
  }
</i18n>
