<script setup lang="ts">
import { ref, computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { UploadedFile, FileStatus } from '../../utils/domain'
import { ErrorMessage } from './ErrorBox.vue'

const { t } = useI18n()
const SMOOTH_DRAG_DELAY_MS = 120
const props = withDefaults(defineProps<{
  allowedExtensions?: string[]
  variant?: 'zone' | 'input'
  maxFiles?: number
  maxFileSizeMb?: number
  attachedFiles: UploadedFile[]
  showLabel?: boolean
  disabled?: boolean
}>(), {
  variant: 'input',
  maxFiles: -1,
  maxFileSizeMb: 50,
  attachedFiles: () => [],
  allowedExtensions: () => [],
  showLabel: false,
  disabled: false
})

const maxFileSizeBytes = computed(() => props.maxFileSizeMb * 1024 * 1024)

const inputRef = ref<HTMLInputElement | null>(null)
const emit = defineEmits<{
  (e: 'error', error: ErrorMessage): void
  (e: 'filesChange', files: UploadedFile[]): void
}>()
const isDragging = ref(false)

const formattedFileNames = (fileNames: string[]) => {
  return fileNames.join(', ')
}

const formattedFileAccept = computed(() => {
  return props.allowedExtensions.map((ext) => `.${ext}`).join(',')
})

const isMaxFilesReached = computed(() => {
  return props.maxFiles !== -1 && props.attachedFiles.length >= props.maxFiles
})

const isFileExtensionSupported = (filename: string): boolean => {
  const extension = filename.split('.').pop()?.toLowerCase() || ''
  return props.allowedExtensions.includes(extension)
}

const buildErrorMessage = (files: string[], errorType: string) => {
  if (files.length === 0) return null

  const errorMessages = {
    filesTooLarge: {
      title: t('filesTooLargeTitle'),
      message: t('filesTooLarge', { files: formattedFileNames(files), max: props.maxFileSizeMb })
    },
    unsupportedFiles: {
      title: t('unsupportedFilesTitle'),
      message: t('unsupportedFiles', { files: formattedFileNames(files), extensions: formattedFileNames(props.allowedExtensions) })
    },
    discardedFiles: {
      title: t('tooManyFiles', { max: props.maxFiles }),
      message: t('discardedFiles', { files: formattedFileNames(files), max: props.maxFiles })
    }
  }

  return errorMessages[errorType as keyof typeof errorMessages]
}

const validateFiles = (files: File[] | FileList) => {
  const newFiles = Array.from(files || [])
  if (!newFiles.length) return

  const fileStatus = {
    oversized: [] as string[],
    unsupported: [] as string[],
    discarded: [] as string[],
    valid: [] as File[]
  }

  newFiles.forEach((file) => {
    const isOverMaxSize = file.size > maxFileSizeBytes.value
    const isUnsupportedType = !isFileExtensionSupported(file.name)
    const isDuplicate = props.attachedFiles.some((attached) => attached.name === file.name && attached.file?.size === file.size)

    if (isOverMaxSize) fileStatus.oversized.push(file.name)
    if (isUnsupportedType) fileStatus.unsupported.push(file.name)

    if (!isOverMaxSize && !isUnsupportedType && !isDuplicate) {
      fileStatus.valid.push(file)
    }
  })

  const availableSlots = props.maxFiles === -1 ? Infinity : props.maxFiles - props.attachedFiles.length
  if (fileStatus.valid.length > availableSlots) {
    const excessFiles = fileStatus.valid.slice(availableSlots)
    fileStatus.discarded.push(...excessFiles.map(f => f.name))
    fileStatus.valid = fileStatus.valid.slice(0, availableSlots)
  }

  const errorParts = [
    buildErrorMessage(fileStatus.oversized, 'filesTooLarge'),
    buildErrorMessage(fileStatus.unsupported, 'unsupportedFiles'),
    buildErrorMessage(fileStatus.discarded, 'discardedFiles')
  ].filter(Boolean) as Array<{
    title: string
    message: string
  }>

  if (errorParts.length > 0) {
    const finalTitle = errorParts.length > 1 ? t('fileUploadIssues') : errorParts[0].title
    emit('error', new ErrorMessage(finalTitle, errorParts.map((part) => part.message).join('\n\n')))
  }

  if (fileStatus.valid.length > 0) {
    emit('filesChange', fileStatus.valid.map((file) => new UploadedFile(file.name, file.type, FileStatus.PENDING, file)))
  }
}

const handleDragEvent = (event: DragEvent, value: boolean) => {
  if (props.disabled) return
  event.preventDefault()
  setTimeout(() => {
    isDragging.value = value
  }, SMOOTH_DRAG_DELAY_MS)
}

const handleDrop = (event: DragEvent) => {
  if (props.disabled) return
  handleDragEvent(event, false)
  const files = event.dataTransfer?.files
  if (!files?.length) return
  validateFiles(files)
}

const handleFileChange = (event: Event) => {
  const target = event.target as HTMLInputElement
  if (target.files) {
    validateFiles(target.files)
  }

  // Reset the file input to allow selecting the same files again
  if (inputRef.value) {
    inputRef.value.value = ''
  }
}

const triggerFileInput = () => {
  if (props.disabled) return
  inputRef.value?.click()
}

const addFiles = (files: File[]) => {
  if (props.disabled || !files.length) return
  validateFiles(files)
}

defineExpose({
  triggerFileInput,
  addFiles
})
</script>

<template>
  <div>
    <div @dragover="handleDragEvent($event, true)" @dragleave="handleDragEvent($event, false)" @drop="handleDrop">
      <input type="file" multiple ref="inputRef" class="hidden" :accept="formattedFileAccept" @change="handleFileChange" />
      <div v-if="variant == 'input'" :class="[isDragging ? '!border-abstracta animation-pulse' : '', disabled ? 'cursor-not-allowed opacity-50' : 'cursor-pointer hover:bg-auxiliar-gray/15']" class="flex flex-col items-center border-2 border-dashed rounded-lg p-6 pb-8" @click="triggerFileInput">
        <div class="w-full flex flex-col items-center gap-2">
          <span class="bold-span" v-html="t('dragDropBrowse', maxFiles === 1 ? maxFiles : 2)"></span>
          <span v-if="showLabel" class="text-sm text-content-muted">
            {{ t('supportedFileTypes') }}: <span class="font-semibold">{{ formattedFileNames(allowedExtensions) }}</span> | {{ t('maxFileSize') }}: <span class="font-semibold">{{ maxFileSizeMb }} MB</span>
          </span>
        </div>
      </div>
      <div v-else class="w-full flex flex-col">
        <div v-if="isDragging && !isMaxFilesReached" class="absolute inset-0 bg-surface-muted/90 rounded-xl flex items-center justify-center pointer-events-none">
          <div class="font-medium">{{ t('dropFilesHere', { extensions: formattedFileNames(allowedExtensions) }) }}</div>
        </div>
        <slot />
      </div>
    </div>
  </div>
</template>

<i18n lang="json">
  {
    "en":{
      "tooManyFiles": "Too many files attached",
      "filesTooLargeTitle": "Size limit exceeded",
      "filesTooLarge": "The file size limit is {max} MB. Please reduce the size of the files: {files} and try again.",
      "dropFilesHere": "Drag and drop your {extensions} files here",
      "dragDropBrowse": "Drag and drop @:files here or {'<'}span class='underline'>browse{'<'}/span>",
      "files": "your file | your files",
      "discardedFiles": "The following files were discarded: {files} (max {max} files allowed)",
      "unsupportedFilesTitle": "Unsupported file type",
      "unsupportedFiles": "The following files are not supported: {files}. \n Supported extensions: {extensions}",
      "fileUploadIssues": "File upload issues",
      "supportedFileTypes": "Supported file types",
      "maxFileSize": "Maximum file size"
    },
    "es":{
      "tooManyFiles": "Demasiados archivos adjuntos",
      "filesTooLargeTitle": "Límite de tamaño excedido",
      "filesTooLarge": "El límite de tamaño por archivo es {max} MB. Por favor, reduzca el tamaño de los archivos: {files} y vuelva a intentarlo.",
      "dropFilesHere": "Arrastra y suelta @:files {extensions} aquí",
      "dragDropBrowse": "Arrastra y suelta @:files aquí o {'<'}span class='underline'>selecciona un archivo{'<'}/span>",
      "files": "tu archivo | tus archivos",
      "discardedFiles": "Los siguientes archivos fueron descartados: {files} (máximo {max} archivos permitidos)",
      "unsupportedFilesTitle": "Tipo de archivo no soportado",
      "unsupportedFiles": "Los siguientes archivos no son soportados: {files}. Extensiones soportadas: {extensions}",
      "fileUploadIssues": "Problemas al subir archivos",
      "supportedFileTypes": "Tipos de archivos admitidos",
      "maxFileSize": "Tamaño máximo de archivo"
    }
  }
</i18n>
