<script lang="ts" setup>
import { computed, onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import hljs from 'highlight.js'
import SidePanelRight from '@/components/common/SidePanelRight.vue'
import CopyButton from '@tero/common/components/common/CopyButton.vue'
import { useErrorHandler } from '@/composables/useErrorHandler'
import { ApiService } from '@/services/api'

const props = defineProps<{
  content: string
  title: string
  path?: string
  messageId?: number
}>()

const { t } = useI18n()
const { handleError } = useErrorHandler()
const api = new ApiService()
const content = ref(props.content)
const loading = ref(false)
const loadFailed = ref(false)

const formatted = computed(() => {
  try {
    const text = JSON.stringify(JSON.parse(content.value), null, 2)
    return { text, html: hljs.highlight(text, { language: 'json' }).value }
  } catch {
    return null
  }
})

onMounted(async () => {
  if (!props.path || props.messageId == null) return
  loading.value = true
  try {
    const detail = await api.findThreadMessageDetail(props.messageId, props.path)
    content.value = detail.value
  } catch (error) {
    loadFailed.value = true
    handleError(error)
  } finally {
    loading.value = false
  }
})
</script>

<template>
  <SidePanelRight :title="title">
    <template #content>
      <div v-if="loading" class="flex justify-center py-8">
        <div class="w-4 h-4 border-2 border-t-transparent rounded-full animate-spin"></div>
      </div>
      <p v-else-if="loadFailed" class="text-sm text-content-muted">{{ t('loadError') }}</p>
      <pre v-else-if="formatted" class="text-sm whitespace-pre-wrap wrap-break-word"><code class="block p-2 bg-surface-muted border rounded-lg" v-html="formatted.html" /></pre>
      <p v-else class="text-sm whitespace-pre-wrap wrap-break-word">{{ content }}</p>
    </template>
    <template v-if="!loading && !loadFailed" #footer>
      <div class="flex items-center justify-end py-4">
        <CopyButton :text="formatted?.text ?? content" v-tooltip.bottom="t('copyButton')" />
      </div>
    </template>
  </SidePanelRight>
</template>

<i18n lang="json">
{
  "en": {
    "copyButton": "Copy",
    "loadError": "Couldn't load the full text"
  },
  "es": {
    "copyButton": "Copiar",
    "loadError": "No se pudo cargar el texto completo."
  }
}
</i18n>
