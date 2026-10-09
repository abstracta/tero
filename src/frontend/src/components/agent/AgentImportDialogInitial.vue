<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { IconBox, IconInfoCircleFilled, IconLoader2, IconPackageImport, IconTrashX } from '@tabler/icons-vue'
import { UploadedFile } from '../../../../common/src/utils/domain'
import { truncateFileName } from '../../../../common/src/utils/file'
import { ErrorMessage } from '../../../../common/src/components/common/ErrorBox.vue'
import FileInput from '../../../../common/src/components/common/FileInput.vue'

const props = defineProps<{
  importing: boolean
  uploadedFiles: UploadedFile[]
  uploadedFilesError: ErrorMessage
  maxFileSizeMb?: number
}>()

defineEmits<{
  (e: 'close'): void
  (e: 'import'): void
  (e: 'remove-file'): void
  (e: 'files-change', files: UploadedFile[]): void
  (e: 'file-error', error: ErrorMessage): void
}>()

const { t } = useI18n()

const description = computed(() => {
  if (props.importing) return t('importingDescription')
  if (props.uploadedFiles.length > 0) return t('fileSelectedDescription')
  return t('idlePlaceholder')
})
</script>

<template>
  <div class="flex flex-col gap-4">
    <div class="flex min-h-[54px] items-center border-b pb-4">
      <div class="flex items-center gap-2">
        <IconLoader2 v-if="importing" class="animate-spin" />
        <IconPackageImport v-else />
        <span class="bold-span">{{ importing ? t('importingTitle') : t('importAgentTitle') }}</span>
      </div>
    </div>

    <span :class="{ 'whitespace-pre-line': importing }">{{ description }}</span>

    <div v-if="!importing" class="mb-2 flex w-full flex-row gap-2 rounded-lg bg-surface-muted p-4 px-2 text-sm bold-span">
      <IconInfoCircleFilled size="24" class="text-content" />
      <span class="flex-1" v-html="t('importWarningText')"></span>
    </div>

    <div v-if="uploadedFiles.length > 0" class="flex flex-row items-center justify-between rounded-lg border p-2">
      <div class="flex flex-row items-center gap-2">
        <IconBox class="text-content-muted" />
        {{ truncateFileName(uploadedFiles[0].name) }}
      </div>
      <SimpleIcon v-if="!importing" interactive @click="$emit('remove-file')" :icon="IconTrashX" class="text-content-muted" />
    </div>

    <template v-if="!importing && uploadedFiles.length === 0">
      <FileInput
        :attachedFiles="uploadedFiles"
        :maxFiles="1"
        :maxFileSizeMb="maxFileSizeMb"
        showLabel
        :allowedExtensions="['zip']"
        @error="$emit('file-error', $event)"
        @files-change="$emit('files-change', $event)"
      />
      <ErrorBox :error="uploadedFilesError" />
    </template>

    <div v-if="!importing" class="mt-6 flex flex-row justify-end gap-4">
      <SimpleButton @click="$emit('close')" shape="square">{{ t('cancel') }}</SimpleButton>
      <SimpleButton v-if="uploadedFiles.length > 0" @click="$emit('import')" variant="primary" shape="square">
        {{ t('importAgentButton') }}
      </SimpleButton>
    </div>
  </div>
</template>

<i18n lang="json">
{
  "en": {
    "importAgentTitle": "Import agent",
    "importingTitle": "Importing agent…",
    "idlePlaceholder": "Upload a ZIP file to import an agent configuration.",
    "importWarningText": "Importing a file will replace the agent configuration you're currently editing. {'<'}span class='underline'> This action can't be undone.{'<'}/span>",
    "fileSelectedDescription": "Review the selected file before importing it.",
    "importingDescription": "We're validating the file and applying the configuration.\n\nThis may take a few seconds.",
    "importAgentButton": "Import",
    "cancel": "Cancel"
  },
  "es": {
    "importAgentTitle": "Importar agente",
    "importingTitle": "Importando agente…",
    "idlePlaceholder": "Sube un archivo ZIP para importar la configuración de un agente.",
    "importWarningText": "Importar un archivo reemplazará la configuración del agente que estás editando actualmente. {'<'}span class='underline'> Esta acción no se puede deshacer.{'<'}/span>",
    "fileSelectedDescription": "Revisa el archivo seleccionado antes de importarlo.",
    "importingDescription": "Estamos validando el archivo y aplicando la configuración.\n\nEsto puede tomar unos segundos.",
    "importAgentButton": "Importar",
    "cancel": "Cancelar"
  }
}
</i18n>
