<script setup lang="ts">
import { ref, useId } from 'vue'
import { useResizeObserver } from '@vueuse/core'
import { useI18n } from 'vue-i18n'
import { IconChevronDown, IconChevronUp } from '@tabler/icons-vue'
import type { AgentPrompt } from '../../../../common/src/utils/domain'

defineProps<{ prompt: AgentPrompt }>()

const { t } = useI18n()
const contentId = useId()
const expanded = ref(false)
const content = ref<HTMLElement | null>(null)
const truncated = ref(false)
useResizeObserver(content, () => {
  if (content.value && !expanded.value) {
    truncated.value = content.value.scrollHeight > content.value.clientHeight
  }
})
</script>

<template>
  <article class="flex flex-col gap-3 rounded-xl border border-auxiliar-gray bg-surface p-4 text-sm">
    <h3 class="font-medium leading-6" :class="expanded ? 'break-words' : 'truncate'" :title="prompt.name">{{ prompt.name }}</h3>
    <p :id="contentId" ref="content" class="whitespace-pre-wrap break-words leading-6" :class="{ 'line-clamp-3': !expanded }">{{ prompt.content }}</p>
    <div v-if="truncated || expanded" class="flex items-center">
      <button
        type="button"
        class="flex items-center gap-1 rounded-md px-2 py-1 text-content-muted hover:bg-surface-muted hover:text-content cursor-pointer"
        :aria-expanded="expanded"
        :aria-controls="contentId"
        @click="expanded = !expanded"
      >
        {{ t(expanded ? 'showLess' : 'showMore') }}
        <IconChevronUp v-if="expanded" :size="16" />
        <IconChevronDown v-else :size="16" />
      </button>
    </div>
  </article>
</template>

<i18n lang="json">
{
  "en": {
    "showMore": "Show more",
    "showLess": "Show less"
  },
  "es": {
    "showMore": "Ver más",
    "showLess": "Ver menos"
  }
}
</i18n>
