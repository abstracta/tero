<script setup lang="ts">
import type { ChatUiMessage } from './ChatMessage.vue';
import { UserFeedback, UploadedFile } from "../../utils/domain";

defineProps<{
  messages: ChatUiMessage[]
  actionsEnabled: boolean
  depth: number
  showMessageIndexes: Record<number, number>
  isEditingAgent: boolean
  feedbackLoadingMessageId?: number
}>()
const emit = defineEmits<{
  (e: 'promptCreate', text: string): void
  (e: 'editMessage', messageId: number, text: string, files: UploadedFile[]): void
  (e: 'selectMessageBranch', depth: number, index: number): void
  (e: 'feedbackChange', messageId: number, feedback?: UserFeedback): void
  (e: 'viewFile', file: UploadedFile): void
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
      :isFeedbackLoading="feedbackLoadingMessageId === message.id"
      @prompt-create="emit('promptCreate', $event)"
      @edit-message="(text: string, files: UploadedFile[]) => emit('editMessage', message.isUser ? message.id! : message.parent!.id!, text, files)"
      @select-message-branch="(index:number) => emit('selectMessageBranch', depth, index)"
      @feedback-change="(feedback?: UserFeedback) => emit('feedbackChange', message.id!, feedback)"
      @view-file="emit('viewFile', $event)"/>
  </div>
  <!-- Recursive rendering for children -->
  <div v-for="(message, idx) in messages" :key="message.uuid">
    <ChatMessagesList :messages="message.children" :actionsEnabled="actionsEnabled" :depth="depth + 1"
    :isEditingAgent="isEditingAgent"
      :showMessageIndexes="showMessageIndexes"
      :feedbackLoadingMessageId="feedbackLoadingMessageId"
      v-if="showMessageIndexes[depth] == idx"
      @prompt-create="emit('promptCreate', $event)"
      @edit-message="(messageId, text, files) => emit('editMessage', messageId, text, files)"
      @select-message-branch="(depth:number, index:number) => emit('selectMessageBranch', depth, index)"
      @feedback-change="(messageId:number, feedback?: UserFeedback) => emit('feedbackChange', messageId, feedback)"
      @view-file="emit('viewFile', $event)"/>
  </div>
</template>
