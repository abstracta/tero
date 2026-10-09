<script lang="ts" setup>
import { useI18n } from 'vue-i18n';
import { Agent, Thread } from '@/services/api';
import { AnimationEffect } from '../../../../common/src/utils/animations';
import { useChatStore } from '@/composables/useChatStore';
import { useErrorHandler } from '@/composables/useErrorHandler';

const { deleteChat } = useChatStore();
const { handleError } = useErrorHandler();
const { t } = useI18n();

defineProps<{chat: Thread, agent: Agent, editingAgent?: boolean, testCaseLoading?: boolean, hasMessages?: boolean}>()
const emit = defineEmits(['newChat', 'showPastChats', 'createTestCase', 'exportChat']);

const handleChatDelete = async (chat: Thread)=>{
  try {
    await deleteChat(chat);
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
          <ChatHeaderMenu :agent="agent" :chat="chat" :editing-agent="editingAgent" :test-case-loading="testCaseLoading" :has-messages="hasMessages" @show-past-chats="emit('showPastChats')"
            @delete-chat="handleChatDelete" @new-chat="emit('newChat')" @create-test-case="emit('createTestCase')" @export-chat="emit('exportChat')"/>
        </Animate>
      </div>
      <div class="flex items-center gap-2 min-w-0 flex-1">
        <SimpleButton v-if="chat.name && !agent.readOnly" v-tooltip.bottom="t('newChatTooltip')" class="shrink-0" @click="emit('newChat')">
          <IconMessage />
        </SimpleButton>
        <span class="text-content-muted line-clamp-2 min-w-0 wrap-break-word">{{ chat.name }}</span>
      </div>
      <div class="flex shrink-0">
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
