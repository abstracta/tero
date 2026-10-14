<script setup lang="ts">
import { onMounted, ref, watch, computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { useErrorHandler } from '@/composables/useErrorHandler'
import FileInput from '@tero/common/components/common/FileInput.vue'
import { truncateFileName } from '../../../../common/src/utils/file'
import { UploadedFile, FileStatus } from '../../../../common/src/utils/domain'
import { ApiService } from '@/services/api'
import { IconEye, IconRefresh, IconTrashX, IconFile } from '@tabler/icons-vue'

const props = defineProps<{
  agentId: number
  toolId: string
  configuredTool: boolean
  contactEmail?: string
  maxFileSizeMb?: number
  viewMode?: boolean
  allowedExtensions?: string[]
  maxFiles?: number
  onBeforeFileUpload: (filesCount: number) => Promise<void>
  onAfterFileRemove: (filesCount: number) => Promise<void>
}>()

const { t } = useI18n()
const api = new ApiService()
const { handleError } = useErrorHandler()

const defaultAllowedExtensions = ['pdf', 'txt', 'md', 'csv', 'xlsx', 'xls', 'docx']
const allowedExtensions = computed(() => props.allowedExtensions ?? defaultAllowedExtensions)
const attachedFilesError = ref({
  title: '',
  message: ''
})

const toolFiles = ref<UploadedFile[]>([])
const deletingFiles = ref<Set<number>>(new Set())
const fileListRef = ref<HTMLElement | null>(null)
// we use this variable to avoid clearing file list when the first file is uploaded
// since that invokes the onBeforeFileUpload hook which changes the configuredTool flag
// wich triggers a reload of the file list where the file has not been uploaded yet
let uploading = false

onMounted(async () => {
  if (props.configuredTool) {
    await loadFiles()
  }
})

async function loadFiles() {
  const files = await api.findAgentToolFiles(props.agentId, props.toolId)
  toolFiles.value = files.map((f) => new UploadedFile(f.name, f.contentType, f.status, undefined, f.id))
  const pendingFiles = toolFiles.value.filter((f) => f.status === FileStatus.PENDING)
  for (const file of pendingFiles) {
    await awaitFileProcessingCompletes(file)
  }
}

const awaitFileProcessingCompletes = async (file: UploadedFile) => {
  while (file.status === FileStatus.PENDING) {
    await new Promise((resolve) => setTimeout(resolve, 1000))
    const files = await api.findAgentToolFiles(props.agentId, props.toolId)
    const updatedFile = files.find((f) => f.id === file.id)
    file.status = updatedFile!.status
    file.id = updatedFile!.id
    toolFiles.value = toolFiles.value.map((f) => (f.id === file.id ? new UploadedFile(file.name, f.contentType, file.status, f.file, file.id) : f))
  }
}

watch(
  () => props.configuredTool,
  async () => {
    if (props.configuredTool && !uploading) {
      await loadFiles()
    }
  }
)

const scrollToLastUploadedFile = () => {
  fileListRef.value?.scrollTo({
    top: fileListRef.value.scrollHeight,
    behavior: 'smooth'
  });
}

const handleFileChange = async (newFiles: UploadedFile[]) => {
  scrollToLastUploadedFile()
  await onFileUpload(newFiles)
}

const onFileUpload = async (files: UploadedFile[]) => {
  try {
    uploading = true
    await props.onBeforeFileUpload(toolFiles.value.length)
    updateToolFilesWithNewFiles(files)
    for (const file of files) {
      try {
        await uploadFile(file)
      } catch (e) {
        console.error(`Error uploading file ${file.name}: ${e}`)
        file.status = FileStatus.ERROR
        updateToolFilesWithUploadedFiles(file)
      }
    }
  } catch (e) {
    handleError(e)
  } finally {
    uploading = false
  }
}

const updateToolFilesWithNewFiles = (files: UploadedFile[]) => {
  for (const file of files) {
      const existingFile = toolFiles.value.find((f) => f.name === file.name)
      if (!existingFile) {
        toolFiles.value.push(file)
      } else {
        file.id = existingFile.id
        toolFiles.value = toolFiles.value.map((f) => (f.id === file.id ? file! : f))
      }
  }
}

const updateToolFilesWithUploadedFiles = (file: UploadedFile) => {
  toolFiles.value = toolFiles.value.map((f) => (f.id === file.id ? file : f))
}

const uploadFile = async (file: UploadedFile) => {
  try {
    if (file.id) {
      // file.file is only set when a new blob was just picked in this session (e.g. drag & drop replace).
      // Retrying a file loaded from the server (loadFiles()) has no blob, so send an empty file
      // to make the backend reprocess the existing content.
      const content = file.file ?? new File([], file.name, { type: file.contentType })
      await api.updateAgentToolFile(props.agentId, props.toolId, file.id, content)
    } else {
      const configuredFile = await api.uploadAgentToolFile(props.agentId, props.toolId, file.file!)
      file.id = configuredFile.id
    }
  } catch (e) {
    throw new Error(`Error configuring tool file ${file.name}: ${e}`)
  } finally {
    updateToolFilesWithUploadedFiles(file)
    await awaitFileProcessingCompletes(file)
  }
}

const tryAgainFileUpload = async (file: UploadedFile) => {
  file.status = FileStatus.PENDING
  await uploadFile(file)
}

const removeFile = async (file: UploadedFile) => {
  try {
    deletingFiles.value.add(file.id!)
    await api.deleteAgentToolFile(props.agentId, props.toolId, file.id!)
    toolFiles.value = toolFiles.value.filter((f) => f.name !== file.name)
    await props.onAfterFileRemove(toolFiles.value.length)
  } catch (e) {
    handleError(e)
  } finally {
    deletingFiles.value.delete(file.id!)
  }
}

const isDeletingFile = (file: UploadedFile) => {
  return deletingFiles.value.has(file.id!)
}

const viewFile = (file: UploadedFile) => {
  window.open(`/agents/${props.agentId}/tools/${props.toolId}/files/${file.id}${props.viewMode ? '?readOnly=true' : ''}`, '_blank')
}

const filteredFiles = computed(() => props.viewMode ? toolFiles.value.filter(f => f.status === FileStatus.PROCESSED || f.status === FileStatus.QUOTA_EXCEEDED || f.status === FileStatus.RATE_LIMITED) : toolFiles.value)

</script>

<template>
  <div class="flex flex-col gap-4 my-1">
    <div ref="fileListRef" class="flex flex-col gap-2 max-h-55 overflow-y-auto">
      <div class="flex flex-col gap-2 text-sm text-content-muted" v-if="!filteredFiles.length">{{ t('noUploadedFiles') }}</div>
      <div class="flex flex-col gap-2 text-sm" v-else>
        <div v-for="(f, index) in filteredFiles" :key="index" class="border rounded-lg flex flex-row justify-between items-center p-2" :class="{ 'border-error-alt': f.status == FileStatus.ERROR }">
          <div class="flex flex-row gap-2 items-center">
            <IconFile class="text-content-muted"/>
            {{ truncateFileName(f.name) }}
          </div>
          <div class="flex flex-row items-center gap-2">
            <div v-if="(f.status == FileStatus.PENDING || isDeletingFile(f))"><IconLoader class="animate-spin"/></div>
            <div v-if="f.status == FileStatus.ERROR && !isDeletingFile(f)" class="flex flex-row items-center gap-2 cursor-default" v-tooltip.bottom="t('errorUploadingFileTooltip')"><IconAlertTriangle class="text-error-alt" /> {{ t('errorUploadingFile') }}</div>
            <div v-if="f.status == FileStatus.QUOTA_EXCEEDED && !isDeletingFile(f)" class="flex flex-row items-center gap-2 cursor-default" v-tooltip.bottom="t('quotaExceededTooltip', { support: contactEmail })"><IconAlertTriangle class="text-error-alt" /> {{ t('quotaExceeded') }}</div>
            <div v-if="f.status == FileStatus.RATE_LIMITED && !isDeletingFile(f)" class="flex flex-row items-center gap-2 cursor-default" v-tooltip.bottom="t('rateLimitedTooltip')"><IconAlertTriangle class="text-error-alt" /> {{ t('rateLimited') }}</div>
            <div v-if="!isDeletingFile(f)" class="flex flex-row gap-2 border-l pl-2">
              <SimpleIcon interactive v-if="f.status == FileStatus.PROCESSED || f.status == FileStatus.QUOTA_EXCEEDED || f.status == FileStatus.RATE_LIMITED" @click="viewFile(f)" v-tooltip.bottom="t('viewFileTooltip')" :icon="IconEye"/>
              <SimpleIcon interactive v-if="(f.status == FileStatus.ERROR || f.status == FileStatus.QUOTA_EXCEEDED || f.status == FileStatus.RATE_LIMITED)" @click="tryAgainFileUpload(f)" v-tooltip.bottom="t('tryAgainTooltip')" :icon="IconRefresh"/>
              <SimpleIcon interactive v-if="f.status != FileStatus.PENDING && !viewMode" @click="removeFile(f)" v-tooltip.bottom="t('removeFileTooltip')" :icon="IconTrashX"/>
            </div>
          </div>
        </div>
      </div>
      <GradientBottom padding="small" v-if="filteredFiles.length >= 5" />
    </div>
    <FileInput v-if="!viewMode" showLabel :attachedFiles="toolFiles" :allowedExtensions="allowedExtensions" :maxFiles="maxFiles" :maxFileSizeMb="maxFileSizeMb" @error="attachedFilesError = $event" @files-change="handleFileChange" />
    <ErrorBox :error="attachedFilesError" />
  </div>
</template>

<i18n>
  {
    "en": {
      "noUploadedFiles": "No files uploaded",
      "viewFileTooltip": "View file and processing result",
      "tryAgainTooltip": "Try again",
      "removeFileTooltip": "Remove file",
      "errorUploadingFile": "Error uploading file",
      "errorUploadingFileTooltip": "Something went wrong while uploading this file. Please try again.",
      "quotaExceededTooltip": "You have reached the monthly usage quota. Contact {support} to increase your monthly quota or wait for the next month.",
      "quotaExceeded": "Quota limit exceeded",
      "rateLimitedTooltip": "The AI model is currently experiencing high demand. Try again in a few moments.",
      "rateLimited": "Rate limit exceeded"
    },
    "es": {
      "noUploadedFiles": "No hay archivos subidos",
      "viewFileTooltip": "Ver archivo y resultado de procesamiento",
      "tryAgainTooltip": "Intentar de nuevo",
      "removeFileTooltip": "Eliminar archivo",
      "errorUploadingFile": "Error al subir archivo",
      "errorUploadingFileTooltip": "Algo salió mal al subir este archivo. Por favor, inténtalo de nuevo.",
      "quotaExceededTooltip": "Ha alcanzado la cuota de uso mensual. Contacte a {support} para aumentar su cuota mensual o espere al próximo mes.",
      "quotaExceeded": "Límite de cuota excedido",
      "rateLimitedTooltip": "El modelo de IA está experimentando alta demanda. Intenta de nuevo en unos instantes.",
      "rateLimited": "Modelo saturado"
    }
  }
</i18n>
