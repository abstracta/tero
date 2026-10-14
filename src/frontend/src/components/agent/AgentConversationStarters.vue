<script setup lang="ts">
import { Agent } from '@/services/api'
import { useI18n } from 'vue-i18n'
import { ref } from 'vue'
import { IconEditCircle, IconTrash } from '@tabler/icons-vue';
import { useAgentPromptStore } from '@/composables/useAgentPromptStore';
import { AgentPrompt } from '../../../../common/src/utils/domain';
import { useErrorHandler } from '@/composables/useErrorHandler';

const props = withDefaults(defineProps<{
  starters: AgentPrompt[],
  agent: Agent
}>(), {
  starters: () => []
})
const emit = defineEmits<{
  (e: 'delete', id: number): void
  (e: 'reload'): void
}>()

const { t } = useI18n()
const { updatePrompt, newPrompt } = useAgentPromptStore();
const { handleError } = useErrorHandler();

const editingStarter = ref<AgentPrompt>();
const deletingStarterId = ref<number>();

const confirmDelete = (starter: AgentPrompt) => {
  deletingStarterId.value = starter.id!
}

const cancelDelete = () => {
  deletingStarterId.value = undefined
}

const deleteStarter = () => {
  emit('delete', deletingStarterId.value!)
}

const editStarter = (starter: AgentPrompt) => {
  editingStarter.value = starter;
}

const createStarter = () => {
  editingStarter.value = new AgentPrompt(undefined, undefined, undefined, true, true, true)
}

const closePromptEditor = () => {
  editingStarter.value = undefined
}

const saveStarter = async (p: AgentPrompt) => {
  try {
    if (p.id) {
      await updatePrompt(props.agent.id, p.id, p)
    } else {
      await newPrompt(props.agent.id, p)
    }
    emit('reload')
  } catch (e) {
    handleError(e)
  }
}
</script>

<template>
  <div class="flex items-center justify-between w-full mb-2">
    <label class="!text-sm"> {{ t('conversationStarter') }} </label>
    <SimpleButton
      v-tooltip.top="starters.length >= 4 ? t('maxStartersTooltip') : undefined"
      size="small"
      shape="square"
      class="px-3"
      @click="createStarter"
      :disabled="starters.length >= 4">
        <IconPlus /> {{ t('addConversationStarter') }}
    </SimpleButton>
  </div>
  <div class="flex flex-col gap-2 mb-2">
    <div v-if="starters.length === 0" class="text-sm text-content-muted">
      {{ t('noStarters') }}
    </div>
    <div v-else class="flex flex-col gap-2">
      <div v-for="starter in starters" :key="starter.id" class="rounded-xl border-1 overflow-hidden">
            <ItemConfirmation
              v-if="deletingStarterId === starter.id"
              class="border-none shadow-none !py-0"
              :tooltip="t('deleteStarterConfirmation')"
              @confirm="deleteStarter"
              @cancel="cancelDelete"
            />
          <div class="flex justify-between items-center py-3 px-4" v-else>
            <span class="flex-1 border-r pr-2">{{ starter.name }}</span>
            <div class="flex justify-center items-center pl-2 gap-2">
              <div class="flex justify-center items-center">
                <SimpleIcon interactive @click="editStarter(starter)" v-tooltip.bottom="t('editStarter')" :icon="IconEditCircle"/>
              </div>
              <div class="flex justify-center items-center">
                <SimpleIcon interactive @click="confirmDelete(starter)" v-tooltip.bottom="t('deleteStarter')" :icon="IconTrash" class="hover:text-error-alt"/>
              </div>
            </div>
          </div>
      </div>
    </div>
  </div>
  <PromptEditor
    v-if="editingStarter"
    :starter="true"
    :editing-prompt="editingStarter"
    :shared-agent="agent.team !== undefined"
    :prompt-saver="saveStarter"
    :error-handler="handleError"
    @close="closePromptEditor"
  />
</template>

<i18n>
  {
    "en": {
      "noStarters": "Add example prompts to help users quickly start conversations and understand how to interact with the agent.",
      "deleteStarterConfirmation": "Delete starter?",
      "conversationStarter": "Conversation starters",
      "addConversationStarter": "New starter",
      "maxStartersTooltip": "You have already created the maximum of 4 starters",
      "editStarter": "Edit conversation starter",
      "deleteStarter": "Delete conversation starter"
    },
    "es": {
      "noStarters": "Agrega ejemplos de prompts para ayudar a los usuarios a iniciar conversaciones rápidamente y entender cómo interactuar con el agente.",
      "deleteStarterConfirmation": "¿Eliminar iniciador?",
      "conversationStarter": "Iniciadores de chats",
      "addConversationStarter": "Nuevo iniciador",
      "maxStartersTooltip": "Ya has creado el máximo de 4 iniciadores",
      "editStarter": "Editar iniciador",
      "deleteStarter": "Eliminar iniciador"
    },
  }
</i18n>
