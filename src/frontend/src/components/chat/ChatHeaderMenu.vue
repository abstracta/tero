<script setup lang="ts">
import { computed, ref } from 'vue';
import { useI18n } from 'vue-i18n';
import { Agent, Thread } from '@/services/api';
import { useAgentStore } from '@/composables/useAgentStore';
import { IconMessage2Plus, IconEditCircle, IconHistory, IconTrash, IconInfoCircle, IconCopyPlus, IconBug, IconLoader2, IconPackageExport } from '@tabler/icons-vue';
import { useErrorHandler } from '@/composables/useErrorHandler';

const { configureAgent, cloneAgent } = useAgentStore();
const { handleError } = useErrorHandler();

const props = defineProps<{
    agent: Agent,
    chat: Thread,
    editingAgent?: boolean,
    testCaseLoading?: boolean,
    hasMessages?: boolean
}>()
const emit = defineEmits<{
    (e: 'showPastChats'): void
    (e: 'deleteChat', chat: Thread): void
    (e: 'newChat'): void
    (e: 'createTestCase'): void
    (e: 'exportChat'): void
}>();


const { t } = useI18n();
const showDeleteConfirmation = ref(false);
const showCreateTestCaseConfirmation = ref(false);
const menuIsActive = ref(false);
const showAgentInfoModal = ref(false);

const agentName = computed(() => props.agent.name || t(props.agent.readOnly ? 'unavailableAgent' : 'newAgentName', { agentId: props.agent.id }));

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

const handleCreateTestCase = () => {
  emit('createTestCase');
}

const canViewAgentInfo = computed(() => props.agent.canEdit || !props.agent.isProtected);

</script>
<template>
  <div class="flex items-center gap-2 px-3 py-1.5 text-content-muted border rounded-2xl dark:bg-surface-muted dark:border-none" :class="[{ '!bg-abstracta !text-white': menuIsActive }]">
    <AgentAvatar :agent="agent" :desaturated="!menuIsActive" :show-shared-status="agent.readOnly">
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
        :can-view-agent-info="canViewAgentInfo"
        :is-collapsed="true"
        :active="menuIsActive"
        @toggle-active="toggleActive"
        @close-menu="closeMenu"
        :items="[
          ...(!agent.readOnly ? [{
            label: t('newChatTooltip'),
            tablerIcon: IconMessage2Plus,
            command: () => emit('newChat')
          }] : []),
          {
            label: t('previousChatsTooltip'),
            tablerIcon: IconHistory,
            command: handlePreviousChats
          },
          ...(agent.canEdit && hasMessages ? [{
            label: t('createTestCaseTooltip'),
            tablerIcon: IconBug,
            command: () => showCreateTestCaseConfirmation = true
          }] : []),
          ...(hasMessages ? [{
            label: t('exportChatTooltip'),
            tablerIcon: IconPackageExport,
            command: () => emit('exportChat')
          }] : []),
          {
            label: t('deleteChatTooltip'),
            tablerIcon: IconTrash,
            command: ()=> showDeleteConfirmation = true
          },
          ...(!agent.readOnly ? [{ separator: true }] : []),
          ...(!editingAgent && !agent.readOnly ? [{
            label: t('agentInfoTooltip'),
            tablerIcon: IconInfoCircle,
            command: handleShowAgentInfo
          }] : []),
          ...(!editingAgent && agent.canEdit ? [{
            label: t('editAgentTooltip'),
            tablerIcon: IconEditCircle,
            command: handleEditAgent
          }] : []),
          ...(canViewAgentInfo && !agent.readOnly ? [{
            label: t('cloneAgentTooltip'),
            tablerIcon: IconCopyPlus,
            command: handleCloneAgent
          }] : []),
         ]" />
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
  <DiscoverAgentInfo v-if="!editingAgent" :agent="agent" :show-modal="showAgentInfoModal" @close="showAgentInfoModal = false" />
  <Dialog v-model:visible="showCreateTestCaseConfirmation" :header="t('createTestCaseConfirmTitle')" :modal="true" :draggable="false"
    :resizable="false" :closable="false" class="max-w-150">
    <div class="flex flex-col gap-5">
      <div class="flex flex-row gap-2 items-start whitespace-pre-line">
        {{ t('createTestCaseConfirmDescription') }}
      </div>
      <div class="flex flex-row gap-2 justify-end">
        <SimpleButton @click="showCreateTestCaseConfirmation = false" shape="square" variant="secondary" :disabled="testCaseLoading">{{ t('cancelButton') }}</SimpleButton>
        <SimpleButton @click="handleCreateTestCase" variant="primary" shape="square" :disabled="testCaseLoading">
          <IconLoader2 v-if="testCaseLoading" class="animate-spin" />
          <span v-else>{{ t('createButton') }}</span>
        </SimpleButton>
      </div>
    </div>
  </Dialog>
</template>
<i18n lang="json">
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
      "exportChatTooltip": "Export chat",
      "createTestCaseTooltip": "Create test",
      "createTestCaseConfirmTitle": "Create test from chat",
      "createTestCaseConfirmDescription": "This will create a new test case with the currently visible conversation. You will be redirected to the test case after creation.",
      "cancelButton": "Cancel",
      "createButton": "Create",
      "unavailableAgent": "Unavailable agent"
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
      "exportChatTooltip": "Exportar chat",
      "createTestCaseTooltip": "Crear test",
      "createTestCaseConfirmTitle": "Crear test a partir de chat",
      "createTestCaseConfirmDescription": "Esto creará un nuevo caso de prueba con la conversación actualmente visible. Serás redirigido al caso de prueba después de la creación.",
      "cancelButton": "Cancelar",
      "createButton": "Crear",
      "unavailableAgent": "Agente no disponible"
    }
  }
</i18n>
