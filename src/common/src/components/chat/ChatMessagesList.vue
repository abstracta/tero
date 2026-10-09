<script setup lang="ts">
import type { ChatUiMessage } from './ChatMessage.vue';
import { UserFeedback, UploadedFile } from "../../utils/domain";

defineProps<{
  messages: ChatUiMessage[]
  actionsEnabled: boolean
  depth: number
  showMessageIndexes: Record<number, number>
  isEditingAgent: boolean
  readOnly?: boolean
  beforeSend?: () => Promise<boolean>
  feedbackLoadingMessageId?: number
  maxFileSizeMb?: number
}>()
const emit = defineEmits<{
  (e: 'promptCreate', text: string): void
  (e: 'editMessage', messageId: number, text: string, files: UploadedFile[], collapsedName?: string): void
  (e: 'selectMessageBranch', depth: number, index: number): void
  (e: 'feedbackChange', messageId: number, feedback?: UserFeedback): void
  (e: 'viewFile', file: UploadedFile): void
  (e: 'openDetail', message: ChatUiMessage): void
}>()

</script>

<template>
  <div v-for="(message, index) in messages" :key="message.uuid">
    <ChatMessage :message="message" v-if="showMessageIndexes[depth] == index"
      :selectedIndex="showMessageIndexes[depth]"
      :siblingsCount="messages.length"
      :actionsEnabled="actionsEnabled"
      :isLastMessage="message.children.length == 0"
      :isEditingAgent="isEditingAgent"
      :readOnly="readOnly"
      :beforeSend="beforeSend"
      :isFeedbackLoading="feedbackLoadingMessageId === message.id"
      :maxFileSizeMb="maxFileSizeMb"
      @prompt-create="emit('promptCreate', $event)"
      @edit-message="(text: string, files: UploadedFile[], collapsedName?: string) => emit('editMessage', message.isUser ? message.id! : message.parent!.id!, text, files, collapsedName)"
      @select-message-branch="(index:number) => emit('selectMessageBranch', depth, index)"
      @feedback-change="(feedback?: UserFeedback) => emit('feedbackChange', message.id!, feedback)"
      @view-file="emit('viewFile', $event)"
      @open-detail="emit('openDetail', $event)"/>
  </div>
  <!-- Recursive rendering for children -->
  <div v-for="(message, idx) in messages" :key="message.uuid">
    <ChatMessagesList :messages="message.children" :actionsEnabled="actionsEnabled" :depth="depth + 1"
    :isEditingAgent="isEditingAgent"
      :readOnly="readOnly"
      :beforeSend="beforeSend"
      :showMessageIndexes="showMessageIndexes"
      :feedbackLoadingMessageId="feedbackLoadingMessageId"
      :maxFileSizeMb="maxFileSizeMb"
      v-if="showMessageIndexes[depth] == idx"
      @prompt-create="emit('promptCreate', $event)"
      @edit-message="(messageId, text, files, collapsedName) => emit('editMessage', messageId, text, files, collapsedName)"
      @select-message-branch="(depth:number, index:number) => emit('selectMessageBranch', depth, index)"
      @feedback-change="(messageId:number, feedback?: UserFeedback) => emit('feedbackChange', messageId, feedback)"
      @view-file="emit('viewFile', $event)"
      @open-detail="emit('openDetail', $event)"/>
  </div>
</template>
