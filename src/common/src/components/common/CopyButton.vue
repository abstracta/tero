<script lang="ts" setup>
import { ref } from 'vue'
import { IconCopy, IconCheck } from '@tabler/icons-vue'
import SimpleIcon from './SimpleIcon.vue'

const props = defineProps<{
  text?: string
  html?: string | (() => string | Promise<string>)
  compact?: boolean
}>()
const emit = defineEmits<{
  (e: 'error', error: unknown): void
}>()
const copied = ref(false)

const resolveHtml = async (): Promise<string | undefined> => {
  if (typeof props.html === 'function') return await props.html()
  return props.html
}

const copy = async (event: MouseEvent) => {
  event.stopPropagation()
  try {
    const html = await resolveHtml()
    const plain = props.text || html || ''
    if (html) {
      await navigator.clipboard.write([
        new ClipboardItem({
          'text/html': new Blob([html], { type: 'text/html' }),
          'text/plain': new Blob([plain], { type: 'text/plain' }),
        }),
      ])
    } else if (plain) {
      await navigator.clipboard.writeText(plain)
    }
    copied.value = true
    setTimeout(() => copied.value = false, 2000)
  } catch (error) {
    emit('error', error)
  }
}
</script>

<template>
  <button v-if="compact" class="flex items-center cursor-pointer" @click="copy">
    <IconCopy v-if="!copied" size="18" class="text-content-muted hover:text-abstracta cursor-pointer" />
    <IconCheck v-else size="18" class="text-primary" />
  </button>
  <SimpleIcon v-else interactive :icon="copied ? IconCheck : IconCopy" @click="copy" />
</template>
