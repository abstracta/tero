<script setup lang="ts">
import { ref, watch } from 'vue'
import { Dialog } from 'primevue'
import { useI18n } from 'vue-i18n'
import { IconUpload, IconX, IconFileText } from '@tabler/icons-vue'
import { UploadedFile } from '../../../../common/src/utils/domain'
import { truncateFileName } from '../../../../common/src/utils/file'
import { ErrorMessage } from '../../../../common/src/components/common/ErrorBox.vue'
import FileInput from '../../../../common/src/components/common/FileInput.vue'


const visible = defineModel<boolean>("visible")
const emit = defineEmits<{
  (e: 'import', file: UploadedFile): void
}>()

const { t } = useI18n()
const uploadedFiles = ref<UploadedFile[]>([])
const uploadedFilesError = ref<ErrorMessage>({
  title: '',
  message: ''
})

const handleFileChange = async (files: UploadedFile[]) => {
  uploadedFiles.value = files
}

const removeFile = () => {
  uploadedFiles.value = []
}

watch(visible, (_) => {
  uploadedFiles.value = []
})
</script>

<template>
  <Dialog v-model:visible="visible" :modal="true" :draggable="false" :resizable="false" :closable="false" class="basic-dialog">
    <FlexCard>
      <template #header>
        <div class="flex flex-row items-center justify-between">
          <div class="flex gap-2 items-center">
            <IconUpload />
            <span> | {{ t('importAgentTitle') }}</span>
          </div>
          <SimpleButton @click="visible = false">
            <IconX />
          </SimpleButton>
        </div>
      </template>
      <div class="mb-4 w-150 min-h-28">
        <div v-if="uploadedFiles.length > 0" class="flex flex-col gap-4">
          <div class="border border-auxiliar-gray rounded-lg flex flex-row justify-between items-center p-2">
            <div class="flex flex-row gap-2 items-center">
              <IconFileText />
              {{ truncateFileName(uploadedFiles[0].name) }}
            </div>
            <InteractiveIcon @click="removeFile()" :icon="IconX"/>
          </div>
          <div class="flex flex-row gap-4 justify-between">
            <div class="flex flex-row gap-2 items-center w-100 text-sm">
              <IconAlertTriangleFilled color="var(--color-error-alt)"/>
              <span v-html="t('importAgentWarning')" class="w-full"></span>
            </div>
            <div class="flex flex-row gap-2 items-center justify-end">
              <SimpleButton @click="visible = false" shape="square">{{ t('cancel') }}</SimpleButton>
              <SimpleButton @click="emit('import', uploadedFiles[0])" variant="primary" shape="square">{{ t('importAgentButton') }}</SimpleButton>
            </div>
          </div>
        </div>
        <div v-else>
          <FileInput :attachedFiles="uploadedFiles" :maxFiles="1" showLabel :allowedExtensions="['zip']" @error="uploadedFilesError = $event" @files-change="handleFileChange"/>
          <ErrorBox :error="uploadedFilesError" />
        </div>
      </div>
    </FlexCard>
  </Dialog>
</template>


<i18n lang="json">
{
  "en": {
    "importAgentTitle": "Import Agent",
    "importAgentWarning": "When importing, you will lose all the configuration of the agent you were editing. {'<'}b>This cannot be undone.{'<'}/b>",
    "importAgentButton": "Import",
    "cancel": "Cancel"
  },
  "es": {
    "importAgentTitle": "Importar agente",
    "importAgentWarning": "Al importar, perderás toda la configuración del agente que estabas editando. {'<'}b>Esto no se puede deshacer.{'<'}/b>",
    "importAgentButton": "Importar",
    "cancel": "Cancelar"
  }
}
</i18n>
