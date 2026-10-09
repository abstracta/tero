<script setup lang="ts">
import { computed, ref, watch, onMounted } from 'vue'
import { Dialog } from 'primevue'
import { IconX } from '@tabler/icons-vue'
import { UploadedFile } from '../../../../common/src/utils/domain'
import { ErrorMessage } from '../../../../common/src/components/common/ErrorBox.vue'
import AgentImportDialogInitial from './AgentImportDialogInitial.vue'
import AgentImportDialogWarning from './AgentImportDialogWarning.vue'
import AgentImportDialogError from './AgentImportDialogError.vue'
import type { AgentImportResult } from '@/services/api'
import { findManifest } from '@/services/api'

const visible = defineModel<boolean>("visible")
const props = defineProps<{
  result?: AgentImportResult | null
  importError?: string | null
}>()
const emit = defineEmits<{
  (e: 'import', file: UploadedFile): void
}>()

const uploadedFiles = ref<UploadedFile[]>([])
const uploadedFilesError = ref<ErrorMessage>({ title: '', message: '' })
const importing = ref(false)
const dismissedImportError = ref(false)
const maxFileSizeMb = ref<number>()

onMounted(async () => {
  const manifest = await findManifest()
  maxFileSizeMb.value = manifest.maxFileSizeMb
})

const hasWarnings = (result: AgentImportResult) =>
  result.unavailableTools.length > 0 ||
  result.unavailableSubagents.length > 0 ||
  result.toolsRequiringAuthentication.length > 0 ||
  !!result.unavailableModel ||
  !!result.defaultModel

type DialogState = 'initial' | 'warning' | 'error'

const dialogState = computed<DialogState>(() => {
  if (!importing.value && props.importError && !dismissedImportError.value) return 'error'
  if (!importing.value && props.result && hasWarnings(props.result)) return 'warning'
  return 'initial'
})

const removeFile = () => {
  uploadedFiles.value = []
  dismissedImportError.value = true
}

const onImport = () => {
  if (!uploadedFiles.value[0]) return
  importing.value = true
  setTimeout(() => {
    emit('import', uploadedFiles.value[0])
  }, 3000)
}

watch(visible, () => {
  uploadedFiles.value = []
  uploadedFilesError.value = { title: '', message: '' }
  importing.value = false
  dismissedImportError.value = false
})

watch(() => props.result, (result) => {
  if (!result || !importing.value) return
  importing.value = false
  if (!hasWarnings(result)) {
    visible.value = false
  }
})

watch(() => props.importError, (error) => {
  dismissedImportError.value = false
  if (!importing.value) return
  if (!error) return
  importing.value = false
})
</script>

<template>
  <Dialog v-model:visible="visible" :modal="true" :draggable="false" :resizable="false" :closable="false" class="basic-dialog">
    <SimpleButton v-if="!importing" @click="visible = false" class="absolute right-4 top-4">
      <IconX />
    </SimpleButton>
    <div class="w-158 min-h-28 p-4 pb-6">
      <AgentImportDialogError
        v-if="dialogState === 'error'"
        :error-key="props.importError"
        :file-name="uploadedFiles[0]?.name"
        @close="visible = false"
        @remove-file="removeFile"
      />
      <AgentImportDialogWarning
        v-else-if="dialogState === 'warning'"
        :result="props.result!"
        @review="visible = false"
      />
      <AgentImportDialogInitial
        v-else
        :importing="importing"
        :uploaded-files="uploadedFiles"
        :uploaded-files-error="uploadedFilesError"
        :max-file-size-mb="maxFileSizeMb"
        @close="visible = false"
        @import="onImport"
        @remove-file="removeFile"
        @files-change="uploadedFiles = $event"
        @file-error="uploadedFilesError = $event"
      />
    </div>
  </Dialog>
</template>
