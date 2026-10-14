<script setup lang="ts">
import { ref, computed } from 'vue';
import { useI18n } from 'vue-i18n';
import { useChatStore } from '@/composables/useChatStore';
import { useErrorHandler } from '@/composables/useErrorHandler';
import { AnimationEffect } from '../../../../common/src/utils/animations';
import { useSidebar } from '@/composables/useSidebar';
import type { Thread } from '@/services/api';
import moment from 'moment';
import { IconPencil, IconTrash } from '@tabler/icons-vue';

const { t, locale } = useI18n();
const { openChat, deleteChat, updateChat } = useChatStore();
const { handleError } = useErrorHandler();
const { isSidebarCollapsed } = useSidebar();

const props = defineProps<{
  chat: Thread,
  nested?: boolean,
}>();

const showConfirmation = ref(false);
const showEditName = ref(false);
const menu = ref();

const lastMessageUpdate = computed(() => {
  moment.locale(locale.value);
  return moment.utc(props.chat.lastMessage ? props.chat.lastMessage : props.chat.creation).local().format('DD MMM YYYY, HH:mm A');
});
const agentName = computed(() => props.chat.agent.name || t('unavailableAgent'));

const handleDeleteChat = () => {
  showConfirmation.value = true;
}
const handleUpdateChatName = () => {
  showEditName.value = true;
}

const onDeleteChat = async () => {
  try {
    await deleteChat(props.chat);
  } catch (e) {
    handleError(e);
  }
};

const onUpdateChatName = async (name: string) => {
  try {
    const chat = {...props.chat}
    chat.name = name;
    await updateChat(chat, false);
    showEditName.value = false;
  } catch (e) {
    handleError(e);
  }
}
</script>

<template>
  <SidebarItem
    v-if="!showConfirmation && !showEditName"
    :is-collapsed="isSidebarCollapsed"
    @click="openChat(chat.id)"
    v-tooltip.bottom="{value: `@${agentName}` + '\n' + lastMessageUpdate, showDelay: 1000}"
    @menu-toggle="(event: Event) => menu?.toggle(event)">
    <p
      class="line-clamp-1 break-all"
      :class="[
        { hidden: isSidebarCollapsed },
        props.nested ? 'text-content opacity-80 text-sm font-normal' : ''
      ]">
      {{ chat.name }}
    </p>
    <template #actions>
      <AgentChatMenu
        ref="menu"
        :is-collapsed="isSidebarCollapsed"
        :can-view-agent-info="true"
        :items="[
        {
          label: t('editChatNameTooltip'),
          tablerIcon: IconPencil,
          command: handleUpdateChatName
        },{
          label: t('deleteChatTooltip'),
          tablerIcon: IconTrash,
          command: handleDeleteChat,
        }
        ]"
        :class="[menu?.isMenuOpen ? 'opactity-100 w-auto' : 'opacity-0 w-0 group-hover:opacity-100 group-hover:w-auto']" />
    </template>
  </SidebarItem>
  <Animate v-if="showConfirmation" :effect="AnimationEffect.QUICK_SLIDE_DOWN">
    <ItemConfirmation
      :tooltip="t('deleteChatConfirmation')"
      @confirm="onDeleteChat"
      @cancel="() => showConfirmation = false"
    />
  </Animate>
  <Animate v-if="showEditName" :effect="AnimationEffect.QUICK_SLIDE_DOWN">
    <SidebarEditor :value="chat.name" :on-save="onUpdateChatName"
      :on-cancel="() => showEditName = false" />
  </Animate>
</template>

<i18n>
  {
    "en": {
      "deleteChatTooltip": "Delete chat",
      "deleteChatConfirmation": "Delete chat?",
      "editChatNameTooltip": "Rename chat",
      "unavailableAgent": "Unavailable agent"
    },
    "es": {
      "deleteChatTooltip": "Eliminar chat",
      "deleteChatConfirmation": "¿Eliminar chat?",
      "editChatNameTooltip": "Renombrar chat",
      "unavailableAgent": "Agente no disponible"
    }
  }
</i18n>
