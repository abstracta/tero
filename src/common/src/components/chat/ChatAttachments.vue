<script setup lang="ts">
import { truncateFileName } from '../../utils/file'
import { IconPhoto, IconFileText, IconX, IconEye } from '@tabler/icons-vue'
import { UploadedFile } from '../../utils/domain'

withDefaults(defineProps<{
  variant: 'message' | 'input'
  attachedFiles?: UploadedFile[]
}>(), { variant: 'input', attachedFiles: () => []})
const emit = defineEmits<{
  (e: 'removeFile', index: number): void
  (e: 'viewFile', file: UploadedFile): void
}>()
</script>

<template>
  <div v-if="attachedFiles.length || $slots.leading" class="flex flex-wrap items-center" :class="variant === 'input' ? 'px-2 pb-4' : ''">
    <slot name="leading" />
    <div v-for="(file, idx) in attachedFiles" :key="idx"
        class="inline-flex items-center whitespace-nowrap rounded-xl bg-surface border px-3 py-1.5 mr-2">
        <IconPhoto v-if="file.contentType.startsWith('image/')" class="mr-2" />
        <IconFileText v-else class="mr-2" />
        <span class="mr-2" v-tooltip.bottom="{value: file.name, showDelay: 1000}">{{ truncateFileName(file.name) }}</span>
        <button v-if="variant === 'input'" @click="emit('removeFile', idx)" class="ml-1 text-content hover:text-error-alt">
          <IconX />
        </button>
        <SimpleIcon interactive v-else-if="file.id" :icon="IconEye" @click="emit('viewFile', file)" />
        <IconLoader v-else class="animate-spin"/>
      </div>
  </div>
</template>
