<script setup lang="ts">
import { computed, ref } from 'vue';
import { useI18n } from 'vue-i18n';
import { Agent, Thread } from '@/services/api';
import { useAgentStore } from '@/composables/useAgentStore';
import { useChatExport } from '@/composables/useChatExport';
import { IconMessage2Plus, IconEditCircle, IconHistory, IconTrash, IconInfoCircle, IconCopyPlus, IconDownload } from '@tabler/icons-vue';
import { useErrorHandler } from '@/composables/useErrorHandler';

const { configureAgent, cloneAgent } = useAgentStore();
const { handleError } = useErrorHandler();
const { exportChatAsJson, exportChatAsMarkdown } = useChatExport();

const props = defineProps<{
    agent: Agent,
    chat: Thread,
    editingAgent?: boolean,
    messages?: any[]
}>()
const emit = defineEmits<{
    (e: 'showPastChats'): void
    (e: 'deleteChat', chat: Thread): void
    (e: 'newChat'): void
}>();


const { t } = useI18n();
const showDeleteConfirmation = ref(false);
const menuIsActive = ref(false);
const showAgentInfoModal = ref(false);

const agentName = computed(() => props.agent.name ? props.agent.name : t('newAgentName', { agentId: props.agent.id }));

const handlePreviousChats = () => {
  emit('showPastChats');
};

const handleDeleteChat = (chat:Thread)=>{
  emit('deleteChat', chat)
  showDeleteConfirmation.value = false;
}

const handleEditAgent = async () => {
  configureAgent(props.agent.id);
}

const toggleActive = () => {
  menuIsActive.value = !menuIsActive.value;
}

const closeMenu = () => {
  menuIsActive.value = false;
}

const handleShowAgentInfo = () => {
  showAgentInfoModal.value = !showAgentInfoModal.value;
};

const handleCloneAgent = async () => {
  try {
    await cloneAgent(props.agent.id);
  } catch (error) {
    handleError(error);
  }
}


const handleExportAsJson = () => {
  exportChatAsJson(props.chat, props.chat.name, agentName.value, props.messages ?? []);
  closeMenu();
}

const handleExportAsMarkdown = () => {
  exportChatAsMarkdown(props.chat, props.chat.name, agentName.value, props.messages ?? []);
  closeMenu();
}

</script>
<template>
  <div class="flex items-center gap-2 px-3 py-1.5 text-light-gray border border-auxiliar-gray rounded-2xl" :class="[{ '!bg-abstracta !text-white': menuIsActive }]">
    <AgentAvatar :agent="agent" :desaturated="!menuIsActive">
      <IconSparkles class="text-white" fill="currentColor" />
    </AgentAvatar>
    <span
      class="truncate max-w-[200px] cursor-default"
      v-tooltip.bottom="{value: agentName, showDelay: 1000}">
      {{ agentName }}
    </span>
    <div class="ml-2">
      <AgentChatMenu
        :agent-team="agent.team?.name"
        :is-collapsed="true"
        :active="menuIsActive"
        @toggle-active="toggleActive"
        @close-menu="closeMenu"
        :items="[
          {
            label: t('newChatTooltip'),
            tablerIcon: IconMessage2Plus,
            command: () => emit('newChat')
          },
          ...(!editingAgent ? [{
            label: t('agentInfoTooltip'),
            tablerIcon: IconInfoCircle,
            command: handleShowAgentInfo
          }] : []),
          ...(!editingAgent && agent.canEdit ? [{
            label: t('editAgentTooltip'),
            tablerIcon: IconEditCircle,
            command: handleEditAgent
          }] : []),
          {
            label: t('cloneAgentTooltip'),
            tablerIcon: IconCopyPlus,
            command: handleCloneAgent
          },
          { separator: true },
          {
            label: t('exportChatAsJsonTooltip'),
            tablerIcon: IconDownload,
            command: handleExportAsJson
          },
          {
            label: t('exportChatAsMarkdownTooltip'),
            tablerIcon: IconDownload,
            command: handleExportAsMarkdown
          },
          { separator: true },
          {
            label: t('previousChatsTooltip'),
            tablerIcon: IconHistory,
            command: handlePreviousChats
          },
          {
            label: t('deleteChatTooltip'),
            tablerIcon: IconTrash,
            command: ()=> showDeleteConfirmation = true
          }]" />
    </div>
  </div>
  <Dialog v-model:visible="showDeleteConfirmation" :header="t('deleteConfirmTitle')" :modal="true" :draggable="false"
    :resizable="false" :closable="false" class="max-w-150">
    <div class="flex flex-col gap-5">
      <div class="flex flex-row gap-2 items-start whitespace-pre-line">
        {{ t('deleteConfirmDescription') }}
      </div>
      <div class="flex flex-row gap-2 justify-end">
        <SimpleButton @click="showDeleteConfirmation = false" shape="square" variant="secondary">{{ t('cancelDeleteButton') }}</SimpleButton>
        <SimpleButton @click="handleDeleteChat(chat)" variant="error" shape="square">{{
          t('deleteButton') }}
        </SimpleButton>
      </div>
    </div>
  </Dialog>
  <DiscoverAgentInfo v-if="!editingAgent" :agent-id="agent.id" :show-modal="showAgentInfoModal" @close="showAgentInfoModal = false" />
</template>
<i18n>
  {
    "en": {
      "deleteChatTooltip": "Delete chat",
      "previousChatsTooltip": "Previous chats",
      "deleteConfirmTitle": "Delete chat",
      "deleteConfirmDescription": "Are you sure you want to delete this chat? This action cannot be undone.",
      "cancelDeleteButton": "Cancel",
      "deleteButton": "Delete",
      "newChatTooltip": "New chat",
      "editAgentTooltip": "Edit agent",
      "agentInfoTooltip": "View details",
      "cloneAgentTooltip": "Clone agent",
      "exportChatAsJsonTooltip": "Export as JSON",
      "exportChatAsMarkdownTooltip": "Export as Markdown"
    },
    "es": {
      "deleteChatTooltip": "Eliminar chat",
      "previousChatsTooltip": "Chats anteriores",
      "deleteConfirmTitle": "Eliminar chat",
      "deleteConfirmDescription": "¿Estás seguro de que deseas eliminar este chat? Esta acción no se puede deshacer.",
      "cancelDeleteButton": "Cancelar",
      "deleteButton": "Eliminar",
      "newChatTooltip": "Nuevo chat",
      "editAgentTooltip": "Editar agente",
      "agentInfoTooltip": "Ver detalles",
      "cloneAgentTooltip": "Clonar agente",
      "exportChatAsJsonTooltip": "Exportar como JSON",
      "exportChatAsMarkdownTooltip": "Exportar como Markdown",
          }
  }
</i18n>
