<script setup lang="ts">
import { ref, onMounted, computed } from 'vue'
import { useRoute } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { useErrorHandler } from '@/composables/useErrorHandler'
import { initializeCodeActionHandler, renderMarkDown } from '../../../common/src/utils/formatter'
import { FileStatus } from '../../../common/src/utils/domain'
import { AgentToolConfig, ApiService, DocToolFile, FileProcessor, findManifest } from '@/services/api'


const route = useRoute()
const api = new ApiService()
const { t } = useI18n()
const { handleError } = useErrorHandler()

const currentBrowser = navigator.userAgent
const loading = ref(false)
const reprocessing = ref(false)
const processedContent = ref<string>('')
const lastUsedProcessor = ref<FileProcessor | null>(null)
const quotaExceeded = ref(false)
const quotaExceededMessage = ref<string>('')
const originalFile = ref<File>()
const originalFileURL = ref<string>('')
const fileName = ref<string>('')

const agentId = route.params.agentId as string || undefined
const toolId = route.params.toolId as string || undefined
const threadId = route.params.threadId as string || undefined
const fileId = parseInt(route.params.fileId as string)
const canReprocess = ref<boolean>(false)
const fileIsNotPreviewable = ref(false)

const isChat = computed(() => threadId !== undefined)
const parsedAgentId = computed(() => parseInt(agentId as string))
const parsedThreadId = computed(() => parseInt(threadId as string))

const processingMode = computed(() => {
  return lastUsedProcessor.value == FileProcessor.ENHANCED ? t('enhancedResult') : t('basicResult')
})

const processingModeButton = computed(() => {
  if (reprocessing.value) return t('reprocessing')
  return lastUsedProcessor.value == FileProcessor.ENHANCED ? t('reprocessWithBasic') : t('reprocessWithAI')
})

const browserIsChrome = computed(() => {
  // based on https://github.com/ICJIA/vue-browser-detect-plugin/blob/master/src/main.js
  return currentBrowser.includes('Chrome') && !currentBrowser.includes('Edge')
})

const browserIsFirefox = computed(() => {
  return currentBrowser.includes('Firefox')
})

const fileIsPdf = computed(() => {
  return fileName.value.toLowerCase().endsWith('.pdf')
})

const backgroundClasses = computed(() => {
  if (!fileIsPdf.value) {
    return 'border-l-[.2px]'
  }
  if (browserIsChrome.value) {
    return 'border-l-10'
  }
  return undefined
})

const headerClasses = computed(() => {
  if (!fileIsPdf.value) return
  if (browserIsChrome.value) {
    return 'bg-surface-muted px-4 py-4 border-l-10 border-l-auxiliar-gray text-content'
  } else if (browserIsFirefox.value) {
    return 'bg-surface-muted py-1 border-b-1 border-b-auxiliar-gray'
  } else{
    return undefined
  }
})

const fileIsImage = computed(() => {
  if (!originalFile.value) return false
  return originalFile.value.type.startsWith('image/')
})

onMounted(async () => {
  initializeCodeActionHandler(t)
  await loadFilePreview()
  await checkUserCanReprocess()
})

const loadFilePreview = async () => {
  loading.value = true
  fileIsNotPreviewable.value = false
  try {
    const [file, processed] = await findFile()
    if (!isChat.value) {
      await checkQuotaExceeded(processed)
    }
    originalFile.value = file
    originalFileURL.value = URL.createObjectURL(file)
    fileName.value = file.name
    processedContent.value = processed.processedContent ?? ''
    lastUsedProcessor.value = processed.fileProcessor
  } catch (e) {
    handleError(e)
  } finally {
    loading.value = false
  }
}

class ProcessedContent {
  status?: FileStatus
  processedContent?: string
  fileProcessor: FileProcessor

  constructor(status: FileStatus, fileProcessor: FileProcessor, processedContent?: string) {
    this.status = status
    this.fileProcessor = fileProcessor
    this.processedContent = processedContent
  }
}

const findFile = async () : Promise<[File, ProcessedContent]> => {
  if (isChat.value) {
    return await Promise.all([api.downloadThreadMessageFile(parsedThreadId.value, fileId), (async () => {
      const messageFile = await api.findThreadMessageFile(parsedThreadId.value, fileId)
      return new ProcessedContent(messageFile.status, FileProcessor.ENHANCED, messageFile.processedContent)
    })()])
  } else {
    return await Promise.all([api.downloadAgentToolFile(parsedAgentId.value, toolId!, fileId), (async () => {
      const docFile = await api.findAgentToolFile(parsedAgentId.value, toolId!, fileId)
      return new ProcessedContent(docFile.status, docFile.fileProcessor, docFile.processedContent)
    })()])
  }
}

const checkQuotaExceeded = async (toolFile: ProcessedContent) => {
  if (toolFile && toolFile.status === FileStatus.QUOTA_EXCEEDED) {
    quotaExceeded.value = true
    const contactEmail = (await findManifest()).contactEmail
    quotaExceededMessage.value = t('quotaExceeded', { contactEmail })
    return true
  }
  return false
}

const checkUserCanReprocess = async () => {
  if (isChat.value) {
    canReprocess.value = false
    return
  }
  try {
    const agent = await api.findAgentById(parsedAgentId.value)
    const tools = await api.findAgentTools()
    const docsTool = tools.find(tool => tool.id === toolId)
    const advancedProcessingAvailable = docsTool?.configSchema?.properties?.advancedFileProcessing !== undefined
    canReprocess.value = (agent.canEdit ?? false) && route.query.readOnly !== 'true' && advancedProcessingAvailable
  }
  catch (e) {
    handleError(e)
  }
}

const close = () => {
  window.close()
}

const downloadFile = () => {
  if (originalFile.value) {
    window.open(originalFileURL.value, '_blank')
  }
}

const reprocess = async () => {
  reprocessing.value = true
  try {
    const newProcessor = lastUsedProcessor.value === FileProcessor.ENHANCED ? FileProcessor.BASIC : FileProcessor.ENHANCED
    const agentId = parsedAgentId.value
    await api.configureAgentTool(agentId, new AgentToolConfig(toolId!, {advancedFileProcessing: newProcessor === FileProcessor.ENHANCED}))
    await api.updateAgentToolFile(agentId, toolId!, fileId, new File([], originalFile.value!.name, { type: originalFile.value!.type }))
    let toolFile = await api.findAgentToolFile(agentId, toolId!, fileId)
    toolFile = await awaitFileProcessingCompletes(toolFile)
    processedContent.value = toolFile.processedContent
    const quotaExceeded = await checkQuotaExceeded(toolFile)
    if (quotaExceeded) return
    lastUsedProcessor.value = newProcessor
  } catch (e) {
    handleError(e)
  } finally {
    reprocessing.value = false
  }
}

const awaitFileProcessingCompletes = async (toolFile: DocToolFile) => {
  while (toolFile.status === FileStatus.PENDING) {
    toolFile = await api.findAgentToolFile(parsedAgentId.value, toolId!, fileId)
    if (toolFile.status === FileStatus.PENDING) {
      await new Promise((resolve) => setTimeout(resolve, 1000))
    }
  }
  return toolFile;
}
</script>

<template>
  <div class="h-screen flex flex-col overflow-hidden">
    <div v-if="loading" class="flex items-center justify-center w-full h-full">
      <div class="flex items-center gap-2">
        <IconLoader class="animate-spin" />
        <span>{{ t('loadingFilePreview') }}</span>
      </div>
    </div>
    <template v-else>
      <div class="bg-surface border-b-[.2px] flex items-center justify-between px-4 py-3 flex-shrink-0">
        <p>{{ fileName }}</p>
        <div class="flex items-center gap-2">
          <SimpleButton v-if="fileIsPdf && canReprocess" @click="reprocess" :disabled="reprocessing || quotaExceeded" v-tooltip.bottom="!reprocessing && (quotaExceeded ? quotaExceededMessage : lastUsedProcessor == FileProcessor.BASIC ? t('reprocessWithAITooltip') : t('reprocessWithBasicTooltip'))">
            <IconLoader v-if="reprocessing" class="animate-spin" />
            <IconRefresh v-else />
            <span>{{ processingModeButton }}</span>
          </SimpleButton>
          <SimpleButton @click="close" :disabled="reprocessing" v-tooltip.bottom="t('closeFilePreviewTooltip')">
            <IconX />
          </SimpleButton>
        </div>
      </div>
      <div class="flex-1 flex min-h-0">
        <div class="w-1/2 flex flex-col">
          <div class="flex-1 overflow-hidden">
            <div v-if="originalFile" class="h-full">
              <div v-if="fileIsNotPreviewable" class="flex flex-col items-center justify-center text-center translate-y-5/6">
                <IconFileText class="w-16 h-16 mb-4" />
                <h3 class="text-lg font-medium mb-2">{{ t('filePreviewNotAvailable') }}</h3>
                <p class="text-content-muted mb-4">{{ t('filePreviewDescription') }}</p>
                <SimpleButton @click="downloadFile" shape="square">
                  {{ t('downloadFile') }}
                </SimpleButton>
              </div>
              <div v-else-if="fileIsImage" class="w-full h-full flex items-start justify-center bg-auxiliar-gray overflow-auto">
                <img
                  :src="originalFileURL"
                  :alt="fileName"
                  @error="fileIsNotPreviewable = true"
                />
              </div>
              <object v-else :data="originalFileURL" :type="originalFile.type" class="w-full h-full" @error="fileIsNotPreviewable = true" />
            </div>
          </div>
        </div>
        <div class="w-1/2 flex flex-col min-h-0">
          <div v-if="headerClasses" :class="headerClasses" class="flex items-center flex-shrink-0">
            <span class="mx-2" v-tooltip.bottom="lastUsedProcessor == FileProcessor.ENHANCED ? t('processingModeEnhancedTooltip') : t('processingModeBasicTooltip')">{{ processingMode }}</span>
          </div>
          <div class="flex-1 overflow-auto p-6" :class="backgroundClasses">
            <div v-if="processedContent" v-html="renderMarkDown(processedContent, true, t)" class="formatted-text" />
            <div v-else class="h-full flex items-center justify-center">
              {{ quotaExceeded ? t('noProcessedContentQuotaExceeded') : t('noProcessedContent') }}
            </div>
          </div>
        </div>
      </div>
    </template>
  </div>
</template>

<i18n>
{
  "en": {
    "loadingFilePreview": "Loading file preview...",
    "noProcessedContent": "No content extracted from the file",
    "noProcessedContentQuotaExceeded": "No content extracted from the file. You have reached the monthly usage quota.",
    "reprocessing": "Reprocessing",
    "basicResult": "Basic processing",
    "enhancedResult": "AI enhanced processing",
    "reprocessWithBasic": "Reprocess (Basic mode)",
    "reprocessWithAI": "Reprocess (AI mode)",
    "reprocessWithBasicTooltip": "You will lose the results generated with AI",
    "reprocessWithAITooltip": "Consumes more budget",
    "quotaExceeded": "You have reached the monthly usage quota. Contact {support} to increase your monthly quota or wait for the next month.",
    "filePreviewNotAvailable": "File preview not available",
    "filePreviewDescription": "You can download the file to view it in a compatible application.",
    "downloadFile": "Download",
    "copyCodeButton": "Copy",
    "copiedMessage": "Copied!",
    "downloadCodeButton": "Download",
    "processingModeBasicTooltip": "Basic processing uses a simple algorithm to extract the content of the file. In general it is less accurate but it is faster and consumes less budget.",
    "processingModeEnhancedTooltip": "Enhanced processing uses AI to extract the content of the file. In general it is more accurate but it consumes more budget and it may take longer to process.",
    "closeFilePreviewTooltip": "Close file preview"
  },
  "es": {
    "loadingFilePreview": "Cargando vista previa de archivo...",
    "noProcessedContent": "No se extrajo contenido del archivo",
    "noProcessedContentQuotaExceeded": "No se extrajo contenido del archivo. Has alcanzado la cuota de uso mensual.",
    "reprocessing": "Reprocesando",
    "basicResult": "Resultado de procesamiento básico",
    "enhancedResult": "Resultado de procesamiento mejorado por IA",
    "reprocessWithBasic": "Reprocesar (Modo básico)",
    "reprocessWithAI": "Reprocesar (Modo IA)",
    "reprocessWithBasicTooltip": "Al reprocesar en modo básico perderás los resultados generados con IA",
    "reprocessWithAITooltip": "Consume más presupuesto",
    "quotaExceeded": "Ha alcanzado la cuota de uso mensual. Contacte a {support} para aumentar su cuota mensual o espere al próximo mes.",
    "filePreviewNotAvailable": "Vista previa de archivo no disponible",
    "filePreviewDescription": "Puedes descargar el archivo para verlo en una aplicación compatible.",
    "downloadFile": "Descargar",
    "copyCodeButton": "Copiar",
    "copiedMessage": "Copiado!",
    "downloadCodeButton": "Descargar",
    "processingModeBasicTooltip": "El procesamiento básico utiliza un algoritmo simple para extraer el contenido del archivo. En general es menos preciso pero es más rápido y consume menos presupuesto.",
    "processingModeEnhancedTooltip": "El procesamiento mejorado utiliza IA para extraer el contenido del archivo. En general es más preciso pero consume más presupuesto y puede tardar más en procesarse.",
    "closeFilePreviewTooltip": "Cerrar vista previa de archivo"
  }
}
</i18n>
