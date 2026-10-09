<script setup lang="ts">
import { Dialog } from 'primevue'
import { IconBook2, IconX } from '@tabler/icons-vue'
import SimpleButton from '../common/SimpleButton.vue'

defineProps<{
  visible: boolean
  title?: string
}>()

const emit = defineEmits<{
  (e: 'close'): void
}>()
</script>

<template>
  <Dialog :visible="visible" :modal="true" :draggable="false" :resizable="false" :closable="false" :dismissable-mask="true" class="w-220" :showHeader="false" @update:visible="emit('close')">
    <div class="flex flex-col h-[85vh]! gap-4">
      <div class="border-b py-4 flex items-center justify-between">
        <div class="flex gap-2 items-center min-w-0">
          <slot name="icon">
            <IconBook2 />
          </slot>
          <span class="text-lg truncate">{{ title }}</span>
        </div>
        <SimpleButton type="button" @click="emit('close')">
          <IconX />
        </SimpleButton>
      </div>
      <div class="flex flex-col flex-1 min-h-0 overflow-y-auto w-full gap-3 p-1">
        <slot />
      </div>
    </div>
  </Dialog>
</template>
