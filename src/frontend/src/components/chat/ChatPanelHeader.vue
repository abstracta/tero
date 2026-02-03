<script lang="ts" setup>
import { useI18n } from 'vue-i18n';
import { Agent, Thread } from '@/services/api';
import { AnimationEffect } from '../../../../common/src/utils/animations';
import { useChatStore } from '@/composables/useChatStore';
import { useErrorHandler } from '@/composables/useErrorHandler';

const { deleteChat } = useChatStore();
const { handleError } = useErrorHandler();
const { t } = useI18n();

defineProps<{chat: Thread, agent: Agent, messages?: any[], editingAgent?: boolean}>()
const emit = defineEmits(['newChat', 'showPastChats']);

const handleChatDelete = async (chat: Thread)=>{
  try {
    await deleteChat(chat.id);
    emit('newChat')
  } catch (e) {
    handleError(e);
  }
}
</script>

<template>
  <div class="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2 sm:gap-0">
    <div class="flex flex-col sm:flex-row items-start sm:items-center gap-2 sm:gap-3 w-full">
      <div class="flex items-center gap-2">
        <Animate :effect="AnimationEffect.FADE_IN">
          <ChatHeaderMenu :agent="agent" :chat="chat" :messages="messages" :editing-agent="editingAgent" @show-past-chats="emit('showPastChats')"
            @delete-chat="handleChatDelete" @new-chat="emit('newChat')"/>
        </Animate>
      </div>
      <div class="flex items-center gap-2">
        <SimpleButton v-if="chat.name" v-tooltip.bottom="t('newChatTooltip')" @click="emit('newChat')">
          <IconMessage />
        </SimpleButton>
        <span class="text-light-gray max-w-[400px]">{{ chat.name }}</span>
      </div>
      <div class="flex flex-grow justify-end">
        <Notifications />
      </div>
    </div>
  </div>
</template>

<i18n>
{
  "en": {
    "newChatTooltip": "New Chat"
  },
  "es": {
    "newChatTooltip": "Nuevo Chat"
  }
}
</i18n>
