<script lang="ts">
import { reactive } from 'vue'
import { type Icon } from '@tabler/icons-vue'
import type { JSONSchema7, JSONSchema7Definition } from 'json-schema'
import { useErrorHandler } from '@/composables/useErrorHandler'
import Ajv, { type ErrorObject } from 'ajv'
import addFormats from 'ajv-formats'
import { handleToolAuthRequestsIn } from '@/services/toolAuth';
import { AuthenticationError } from '@tero/common/utils/toolAuth.js';
import { AgentToolConfig, AgentTool } from '@/services/api'

export class EditingToolConfig {
  agentId: number
  tool: AgentTool
  name: string
  icon: Icon
  toolId: string
  config?: Record<string, unknown>

  constructor(agentId: number, tool: AgentTool, name: string, icon: Icon, toolConfig?: AgentToolConfig) {
    this.agentId = agentId
    this.tool = tool
    this.toolId = toolConfig?.toolId ?? tool.id
    this.config = toolConfig?.config
    this.name = name
    this.icon = icon
  }
}
</script>

<script setup lang="ts">
import { ref, computed, onMounted, watchEffect } from 'vue'
import { useI18n } from 'vue-i18n'
import { ApiService, findManifest, HttpError } from '@/services/api'
import CopyButton from '@tero/common/components/common/CopyButton.vue'
import { buildToolConfigName, toolTranslationKey } from '@tero/common/utils/toolConfig.js'

const props = defineProps<{
  toolConfig: EditingToolConfig
  viewMode?: boolean
}>()

const emit = defineEmits<{
  (e: 'close'): void
  (e: 'update', toolConfig: AgentToolConfig): void
}>()

const { t } = useI18n()
const api = new ApiService()
const { handleError } = useErrorHandler()
const validationErrors = ref<string | null>(null)
const saving = ref(false)
const contactEmail = ref<string>('')
const savedConfig = ref<Record<string, unknown> | undefined>()
const allFilesRemoved = ref(false)
const lastAutoSavedAt = ref<Date | null>(null)
const autosaveStatus = ref<'idle' | 'saving' | 'saved' | 'error'>('idle')
const baseMutableConfig: Record<string, unknown> = { ...(props.toolConfig.config || {}) }
if (Array.isArray(props.toolConfig.config?.customHeaders)) {
  baseMutableConfig.customHeaders = (props.toolConfig.config.customHeaders as Array<Record<string, string>>)
    .map(header => ({ ...header }))
}
const mutableConfig = reactive(baseMutableConfig)
let partialSave = false

onMounted(async () => {
  try {
    const manifest = await findManifest()
    contactEmail.value = manifest.contactEmail
    savedConfig.value = props.toolConfig.config
    initializeToolConfig()
  } catch (error) {
    handleError(error)
  }
})

const initializeToolConfig = () => {
  Object.entries(toolProperties.value).forEach(([propName, propSchema]) => {
    if (isBooleanProperty(propSchema) && mutableConfig[propName] === undefined) {
      mutableConfig[propName] = false
    }
    if (isStringEnumProperty(propSchema) && mutableConfig[propName] === undefined) {
      const defaultEnumValue = js7(propSchema)!.default
      if (defaultEnumValue) {
        mutableConfig[propName] = defaultEnumValue
      }
    }
    if (isCustomHeadersProperty(propName, propSchema) && mutableConfig[propName] === undefined) {
      mutableConfig[propName] = []
    }
  })
}

const toolProperties = computed(() => {
  return props.toolConfig.tool.configSchema.properties!
})

const isBooleanProperty = (toolProp: JSONSchema7Definition) : boolean => {
  const toolPropSchema = js7(toolProp)!
  return toolPropSchema.type === 'boolean'
}

function js7(val: unknown): JSONSchema7 | undefined {
  return val as JSONSchema7
}

const areDependenciesMet = (toolProp: JSONSchema7Definition) : boolean => {
  const toolPropSchema = js7(toolProp)!
  if (toolPropSchema.dependencies === undefined) {
    return true
  }
  return Object.entries(toolPropSchema.dependencies).every(([dependencyKey, dependencyValue]) => {
    const dependencyPropValue = mutableConfig[dependencyKey]
    if (dependencyPropValue === undefined) {
      return false
    }
    const dependencySchema = js7(dependencyValue)
    if (dependencySchema?.const !== undefined) {
      return dependencyPropValue === dependencySchema.const
    }
    return true
  })
}

watchEffect(() => {
  Object.entries(toolProperties.value).forEach(([propName, propSchema]) => {
    if (!areDependenciesMet(propSchema) && mutableConfig[propName] !== undefined) {
      delete (mutableConfig as Record<string, unknown>)[propName]
    }
  })
})

const translateToolPropertyName = (toolId: string, propName: string): string => {
  if (propName === 'customHeaders') {
    const genericTranslation = t(propName)
    if (genericTranslation !== propName) {
      return genericTranslation
    }
  }
  const toolSpecificKey = buildToolPropertyTranslationKey(toolId, propName)
  const toolSpecificTranslation = t(toolSpecificKey)
  return toolSpecificTranslation
}

const solveToolPropertyHelperText = (toolId: string, propName: string): string | null => {
  const helperTextKey = buildToolPropertyTranslationKey(toolId, propName) + 'HelperText'
  const ret = t(helperTextKey)
  return ret != helperTextKey ? ret : null
}

const buildToolPropertyTranslationKey = (toolId: string, propName: string) : string => {
  return toolTranslationKey(toolId) + propName.charAt(0).toUpperCase() + propName.slice(1)
}

const solveToolPropertyTooltip = (toolId: string, propName: string, value: unknown) : string | null => {
  let tooltipKey = buildToolPropertyTranslationKey(toolId, propName) + 'Tooltip'
  let ret = t(tooltipKey)
  if (ret != tooltipKey) {
    return ret
  }
  if (value === undefined || value === null) {
    return null
  }
  const valStr = value.toString()
  tooltipKey = tooltipKey + valStr.charAt(0).toUpperCase() + valStr.slice(1)
  ret = t(tooltipKey)
  return ret != tooltipKey ? ret : null
}

const toolMessage = computed(() => {
  const toolMessageKey = buildToolPropertyTranslationKey(props.toolConfig.tool.id, 'toolMessage')
  const ret = t(toolMessageKey, { frontendUrl: window.location.origin })
  return ret != toolMessageKey ? ret : null
})

const isBrowserTool = computed(() => props.toolConfig.tool.id.startsWith('browser'))
const isWebTool = computed(() => props.toolConfig.tool.id.startsWith('web'))
const isEnableOnlyTool = computed(() => isBrowserTool.value || isWebTool.value)
const showsOAuthRedirect = computed(() =>
  (props.toolConfig.tool.configSchema as Record<string, unknown>)['x-oauth-redirect'] === true
)
const oauthCallbackUrl = computed(() => `${window.location.origin}/tools/${props.toolConfig.tool.id}/oauth-callback`)

const toolAlertMessage = computed(() => {
  const toolAlertMessageKey = buildToolPropertyTranslationKey(props.toolConfig.tool.id, 'toolAlertMessage')
  const ret = t(toolAlertMessageKey, { frontendUrl: window.location.origin })
  return ret != toolAlertMessageKey ? ret : null
})

const toolAlertGuideMessage = computed(() => {
  const toolAlertGuideKey = buildToolPropertyTranslationKey(props.toolConfig.tool.id, 'toolAlertGuideMessage')
  const ret = t(toolAlertGuideKey)
  return ret != toolAlertGuideKey ? ret : null
})

const hasFileProperties = computed(() =>
  Object.values(toolProperties.value).some(prop => isFileProperty(prop) || isFileArrayProperty(prop))
)

const saveButtonText = computed(() => {
  if (isEnableOnlyTool.value) {
    return 'add'
  }
  return hasFileProperties.value ? 'done' : 'save'
})

const isDoneDisabled = computed(() => {
  return saving.value || (hasFileProperties.value && !savedConfig.value && !allFilesRemoved.value)
})

const autosaveMessage = computed(() => {
  switch (autosaveStatus.value) {
    case 'saving':
      return t('savingChanges')
    case 'error':
      return t('errorSavingChanges')
    default:
      return t('changesSavedAutomatically')
  }
})

const autosaveSavedAtMessage = computed(() => {
  if (!lastAutoSavedAt.value || autosaveStatus.value === 'saving' || autosaveStatus.value === 'error') {
    return null
  }
  const formattedTime = lastAutoSavedAt.value.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })
  return t('savedAt', { time: formattedTime })
})

const isFileArrayProperty = (toolProp: JSONSchema7Definition) : boolean => {
  const toolPropSchema = js7(toolProp)
  return toolPropSchema?.type === 'array' && (js7(toolPropSchema?.items)?.$ref?.endsWith('/File') ?? false)
}

const isFileProperty = (toolProp: JSONSchema7Definition) : boolean => {
  return js7(toolProp)?.$ref?.endsWith('/File') ?? false
}

const isArrayEnumProperty = (toolProp: JSONSchema7Definition) : boolean => {
  const toolPropSchema = js7(toolProp)!
  return toolPropSchema.type === 'array' && js7(toolPropSchema.items)?.enum !== undefined
}

const isObjectArrayProperty = (toolProp: JSONSchema7Definition) : boolean => {
  const toolPropSchema = js7(toolProp)!
  return toolPropSchema.type === 'array'
    && js7(toolPropSchema.items)?.type === 'object'
    && js7(toolPropSchema.items)?.properties !== undefined
}

const isCustomHeadersProperty = (propName: string, toolProp: JSONSchema7Definition) : boolean => {
  return propName === 'customHeaders' && isObjectArrayProperty(toolProp)
}

const isStringEnumProperty = (toolProp: JSONSchema7Definition) : boolean => {
  const toolPropSchema = js7(toolProp)!
  return toolPropSchema.type === 'string' && toolPropSchema.enum !== undefined
}

const isSecretStringProperty = (toolProp: JSONSchema7Definition) : boolean => {
  const toolPropSchema = js7(toolProp)!
  return toolPropSchema.type === 'string' && toolPropSchema.writeOnly === true
}

const translateOptionLabel = (propName: string, option: string) : string => {
  return t(buildToolPropertyTranslationKey(props.toolConfig.tool.id, propName) + option.charAt(0).toUpperCase() + option.slice(1))
}

const onBeforeFileUpload = async (_?: unknown) => {
  if (!hasFileProperties.value || savedConfig.value == mutableConfig) {
    return
  }
  saving.value = true
  autosaveStatus.value = 'saving'
  try {
    await api.configureAgentTool(props.toolConfig.agentId, new AgentToolConfig(props.toolConfig.toolId, mutableConfig))
    savedConfig.value = { ...mutableConfig }
    allFilesRemoved.value = false
    lastAutoSavedAt.value = new Date()
    autosaveStatus.value = 'saved'
    partialSave = true
  } catch (error) {
    autosaveStatus.value = 'error'
    await handleError(error)
  } finally {
    saving.value = false
  }
}

const onAfterFileRemove = async (filesCount: number) => {
  try {
    if (filesCount === 0) {
      await api.removeAgentToolConfig(props.toolConfig.agentId, props.toolConfig.toolId)
      savedConfig.value = undefined
      allFilesRemoved.value = true
      lastAutoSavedAt.value = new Date()
      autosaveStatus.value = 'saved'
      partialSave = true
    }
  } catch (error) {
    autosaveStatus.value = 'error'
    await handleError(error)
  }
}

const onClose = () => {
  // when tool was configured (because files were uploaded), or the tool config was removed (because files were removed)
  // then we emit the update event so tools listing includes the new configured tool o removes existing tool
  // partialSave is a temporary solution, until we implement proper tool config drafts,
  // to properly refresh interface (removing the toolConfig) when existing tool config is changed
  // and saved in backend as a draft, but the configuration does not complete (eg: oauth cancellation)
  if (!props.toolConfig.config && savedConfig.value || props.toolConfig.config && !savedConfig.value || partialSave) {
    emit('update', new AgentToolConfig(props.toolConfig.toolId, mutableConfig))
  } else {
    emit('close')
  }
}

const cleanupCustomHeaders = () => {
  Object.entries(toolProperties.value).forEach(([propName, propSchema]) => {
    if (isCustomHeadersProperty(propName, propSchema) && Array.isArray(mutableConfig[propName])) {
      mutableConfig[propName] = (mutableConfig[propName] as Array<Record<string, string>>)
        .filter(entry => entry.name?.trim() && entry.value?.trim())
    }
  })
}

const saveToolConfig = async () => {
  saving.value = true
  clearValidationErrors()
  try {
    cleanupCustomHeaders()
    validateToolConfig()
    let ret = new AgentToolConfig(props.toolConfig.toolId, mutableConfig)
    // avoid saving tool when requires files and none have been uploaded
    if (savedConfig.value !== mutableConfig && !(Object.values(toolProperties.value).some(prop => isFileProperty(prop) || isFileArrayProperty(prop)) && !savedConfig.value)) {
      partialSave = true
      ret = await handleToolAuthRequestsIn(async () => {
        // resetting this flag since we want to allow retry saving tool config to re open oauth popup, or allow cancelling oauth
        // when oauth popup is closed or just ignored
        saving.value = true
        try {
          return await api.configureAgentTool(props.toolConfig.agentId, ret)
        } finally {
          // here is where we re enable save and cancel buttons in case oauth popup is showed or auth completed
          saving.value = false
        }
      }, api, { skipTokenAuth: true })
      partialSave = false
    }
    emit('update', ret)
  } catch (error) {
    console.error("Error saving tool config", error)
    if (error instanceof AuthenticationError) {
      validationErrors.value = t(error.errorCode)
    } else if (error instanceof ValidationErrors) {
      validationErrors.value = error.message
    } else if (error instanceof HttpError && (error.status === 400 || error.status === 401)) {
      validationErrors.value = t('invalidToolConfiguration', { contactEmail: contactEmail.value })
    } else {
      await handleError(error)
    }
  } finally {
    saving.value = false
  }
}

const validateToolConfig = () => {
  const ajv = new Ajv()
  ajv.addVocabulary(['x-oauth-redirect'])
  addFormats(ajv)
  const toolConfig = props.toolConfig
  const schema = ajv.compile(removeFileProperties(toolConfig.tool.configSchema))
  const valid = schema(mutableConfig)
  if (!valid) {
    const errors = schema.errors?.map((error) => findValidationErrorMessage(error, toolConfig.tool.id))
    throw new ValidationErrors(errors?.join('\n') ?? '')
  }
}

const removeFileProperties = (schema: JSONSchema7) : JSONSchema7 => {
  const fileProperties = Object.entries(schema.properties ?? {}).filter(([_, value]) => isFileProperty(value) || isFileArrayProperty(value))
  return {
    ...schema,
    properties: Object.fromEntries(Object.entries(schema.properties ?? {}).filter(([_, value]) => !isFileProperty(value) && !isFileArrayProperty(value))),
    required: schema.required?.filter((property) => !fileProperties.some(([key, _]) => key === property))
  }
}

const findValidationErrorMessage = (error: ErrorObject, toolId: string) : string => {
  console.warn("Validation error", error)
  const property = error.instancePath.split('/').pop() ?? ''
  switch (error.keyword) {
    case 'required':
      return t('missingProperty', { property: translateToolPropertyName(toolId, error.params.missingProperty) })
    case 'format':
      return t('invalidPropertyFormat', { property: translateToolPropertyName(toolId, property), format: t(error.params.format + 'Format') })
    case 'minLength':
      return t('invalidPropertyMinLength', { property: translateToolPropertyName(toolId, property), minLength: error.params.limit }, error.params.limit)
    default:
      throw new Error(`UnknownValidationError: ${error.keyword}`)
  }
}

const clearValidationErrors = () => {
  validationErrors.value = null
}

class ValidationErrors extends Error {
  constructor(message: string) {
    super(message)
  }
}
</script>

<template>
  <FlexCard header-class="flex items-center justify-between px-4 -mx-6 " class="border-0! dark:border! px-6!" :class="{ 'pb-6!': viewMode }">
    <template #header>
      <div class="flex w-full items-center justify-between">
        <div class="flex gap-2 items-center">
          <component :is="toolConfig.icon" />
          {{ buildToolConfigName(props.toolConfig.toolId) }}
        </div>
        <SimpleButton @click="onClose" :disabled="saving">
          <IconX />
        </SimpleButton>
      </div>
    </template>
    <div class="flex flex-col gap-6">
      <div class="flex flex-col gap-3">
        <div v-if="toolMessage" class="tool-message" v-html="toolMessage"></div>
        <div v-if="toolAlertMessage" class="flex w-full flex-row gap-2 rounded-lg bg-surface-muted p-2 text-sm ">
          <IconInfoCircleFilled size="24" class="text-content" />
          <div class="flex flex-col">
            <span v-html="toolAlertMessage" class="inline pt-0.5"></span>
            <div v-if="showsOAuthRedirect" class="mt-1 flex items-center gap-1">
              {{ oauthCallbackUrl }}
              <CopyButton :text="oauthCallbackUrl" @error="handleError" />
            </div>
            <span
              v-if="showsOAuthRedirect && toolAlertGuideMessage"
              v-html="toolAlertGuideMessage"
              class="inline">
            </span>
          </div>
        </div>
      </div>
      <template v-for="propName in Object.keys(toolProperties)" :key="propName">
        <div v-if="isFileArrayProperty(toolProperties[propName]) || isFileProperty(toolProperties[propName])">
          <label :for="propName" class="text-sm! font-semibold!">{{ translateToolPropertyName(toolConfig.tool.id, propName) }}</label>
          <AgentToolFilesEditor
              :id="propName"
              :agent-id="toolConfig.agentId"
              :tool-id="toolConfig.tool.id"
              :configured-tool="savedConfig != undefined"
              :contact-email="contactEmail"
              :view-mode="viewMode"
              :allowed-extensions="(toolProperties[propName] as any)?.['x-allowed-extensions']"
              :max-files="isFileProperty(toolProperties[propName]) ? 1 : undefined"
              :on-before-file-upload="onBeforeFileUpload"
              :on-after-file-remove="onAfterFileRemove"/>
        </div>
        <div v-else-if="isBooleanProperty(toolProperties[propName])" class="flex gap-2 border-t p-6 -mx-6 flex-col">
          <span class="font-semibold text-sm">{{ t('advancedOptions') }}</span>
          <div class="flex w-full flex-row gap-4 items-center">
            <ToggleSwitch
                v-model="mutableConfig[propName] as boolean"
                v-tooltip.bottom="solveToolPropertyTooltip(toolConfig.tool.id, propName, mutableConfig[propName])"
                @update:model-value="onBeforeFileUpload"
                :disabled="viewMode" />
            <span v-html="translateToolPropertyName(toolConfig.tool.id, propName)"></span>
         </div>
        </div>
        <div v-else-if="isArrayEnumProperty(toolProperties[propName])">
          <AgentToolConfigEnumPropertyEditor
              v-model="mutableConfig[propName] as string[]"
              :id="propName"
              :label="translateToolPropertyName(toolConfig.tool.id, propName)"
              :option-values="js7(js7(toolProperties[propName])!.items)!.enum as string[]"
              :option-labels="translateOptionLabel"
              :description="solveToolPropertyHelperText(toolConfig.tool.id, propName)"
              :view-mode="viewMode"/>
        </div>
        <div v-else-if="isStringEnumProperty(toolProperties[propName])" class="flex flex-col gap-1">
          <label :for="propName" class="text-sm font-semibold">{{ translateToolPropertyName(toolConfig.tool.id, propName) }}</label>
          <Select
              v-model="mutableConfig[propName]"
              :id="propName"
              :options="(js7(toolProperties[propName])!.enum as string[]).map(val => ({ label: translateOptionLabel(propName, val), value: val }))"
              optionLabel="label"
              optionValue="value"
              class="w-1/2 my-1"
              :disabled="viewMode"
              @change="clearValidationErrors"/>
          <p
            v-if="solveToolPropertyHelperText(toolConfig.tool.id, propName)"
            class="text-sm text-content-muted">
            {{ solveToolPropertyHelperText(toolConfig.tool.id, propName) }}
          </p>
        </div>
        <AgentToolConfigEditorAdvancedSettings
          v-else-if="isCustomHeadersProperty(propName, toolProperties[propName])"
          v-model="mutableConfig[propName] as Array<Record<string, string>>"
          :field-id="propName"
          :label="translateToolPropertyName(toolConfig.tool.id, propName)"
          :view-mode="viewMode" />
        <div v-else-if="!isBooleanProperty(toolProperties[propName]) && areDependenciesMet(toolProperties[propName])" class="flex flex-col gap-1">
          <label :for="propName" class="text-sm font-semibold">{{ translateToolPropertyName(toolConfig.tool.id, propName) }}</label>
          <InteractiveInput
              v-model="mutableConfig[propName] as string"
              :id="propName"
              :type="isSecretStringProperty(toolProperties[propName]) ? 'password' : 'text'"
              :placeholder="t(buildToolPropertyTranslationKey(toolConfig.tool.id, propName) + 'Placeholder')"
              :disabled="viewMode"/>
          <p
            v-if="solveToolPropertyHelperText(toolConfig.tool.id, propName)"
            class="text-sm text-content-muted" v-html="solveToolPropertyHelperText(toolConfig.tool.id, propName)">
          </p>
        </div>
      </template>
      <div v-if="validationErrors" class="text-error-alt validation-errors" v-html="validationErrors"></div>
    </div>
    <div v-if="!viewMode && hasFileProperties" class="flex gap-2 bg-content dark:bg-surface-muted px-4 py-3 text-sm -mx-6 items-center justify-center text-surface dark:text-content">
      <IconDeviceFloppy class="text-surface dark:text-content" />
      <span class="font-semibold">{{ autosaveMessage }}</span>
      <span v-if="autosaveSavedAtMessage" class="font-normal">| {{ autosaveSavedAtMessage }}</span>
    </div>
    <div class="flex justify-end gap-3 py-4" v-if="!viewMode && !hasFileProperties">
      <SimpleButton @click="onClose" shape="square" :disabled="saving">
        {{ t('cancel') }}
      </SimpleButton>
      <SimpleButton @click="saveToolConfig" variant="primary" shape="square" :disabled="isDoneDisabled">
        {{ t(saveButtonText) }}
      </SimpleButton>
    </div>

  </FlexCard>
</template>

<style>
@import '@/assets/styles.css';

.validation-errors a {
  @apply text-abstracta inline;
}

.tool-message a {
  @apply inline;
}
</style>

<i18n lang="json">
  {
    "en": {
      "save": "Save",
      "add": "Add",
      "done": "Done",
      "cancel": "Cancel",
      "close": "Close",
      "authenticationWindowBlocked": "The authentication popup could not be opened. Please check that popups are allowed for this site and try again.",
      "authenticationCancelled": "The authentication was cancelled. Please complete the authentication process to configure this tool.",
      "authenticationAccessDenied": "The authentication was denied by the server. Please verify that you actually have the permissions necessary to use it.",
      "missingProperty": "A value is required for '{property}'. Please provide one.",
      "invalidPropertyFormat": "Provided '{property}' is not a valid {format}. Please, review the value and try again.",
      "invalidPropertyMinLength": "Provided '{property}' is shorter than {minLength} @:{'character'}. Please, review the value and try again.",
      "character": "character | characters",
      "uriFormat": "URL",
      "invalidToolConfiguration": "Tool configuration failed. Please review the configuration and try again. If the problem persists, contact {'<'}a href='mailto:{contactEmail}?subject=Tero%20Error'>{contactEmail}{'<'}/a>.",
      "docsToolMessage": "Use information from uploaded files in agent responses.",
      "docsToolAlertMessage": "{'<'}span class='font-semibold'>Only upload files that are relevant to this agent.{'<'}/span>{'<'}br/>This helps improve response quality and avoid unnecessary processing.",
      "docsFiles": "Files",
      "docsAdvancedFileProcessing": "{'<'}span class='font-semibold'>Process new PDFs with advanced AI{'<'}/span>{'<'}br/> {'<'}span class='text-content-muted'>Improves understanding of complex PDFs, with higher budget usage.{'<'}/span>",
      "docsAdvancedFileProcessingTooltipFalse": "Basic processing uses a simple algorithm to extract the content of the file. In general it is less accurate but it is faster and consumes less budget. \n\nNote: This option will only apply to new uploaded files.",
      "docsAdvancedFileProcessingTooltipTrue": "Advanced processing uses AI to extract the content of the file. In general it is more accurate but it consumes more budget and it may take longer to process. \n\nNote: This option will only apply to new uploaded files.",
      "mcpToolMessage": "Use tools from any MCP server.",
      "mcpToolAlertMessage": "{'<'}span class='font-semibold'>Only connect servers you trust.{'<'}/span>",
      "mcpServerUrl": "Server URL",
      "mcpServerUrlPlaceholder": "https://example.com/mcp",
      "mcpServerUrlHelperText": "Use the full URL for the MCP server endpoint.",
      "mcpAuthType": "Authentication method",
      "mcpAuthTypeHelperText": "Select the authentication method your MCP server supports, as described in its documentation.",
      "mcpAuthTypeOauth": "OAuth",
      "mcpAuthTypeBearerToken": "Bearer Token",
      "mcpBearerToken": "Bearer Token",
      "mcpBearerTokenPlaceholder": "Paste your Bearer Token",
      "mcpBearerTokenHelperText": "Use a valid token with the permissions required by this server.",
      "customHeaders": "Headers",
      "jiraToolMessage": "Manage issues and track project activity.",
      "jiraToolAlertMessage": "To use this tool, configure a Jira OAuth app using the redirect URL below:",
      "jiraToolAlertGuideMessage": "{'<'}br/>Read the {'<'}a class='inline! font-normal! underline!' href='https://developer.atlassian.com/cloud/confluence/oauth-2-3lo-apps/#enabling-oauth-2-0--3lo-' target='_blank'>Jira setup guide{'<'}/a>.",
      "redmineToolMessage": "Manage issues, projects and log time.",
      "redmineToolAlertMessage": "Check the {'<'}a class='inline! font-normal! underline!' href='https://www.redmine.org/projects/redmine/wiki/rest_api' target='_blank'>API documentation{'<'}/a>",
      "githubToolMessage": "Search code, manage repos, review PRs and track issues.",
      "githubToolAlertMessage": "Check the {'<'}a class='inline! font-normal! underline!' href='https://github.com/github/github-mcp-server' target='_blank'>API documentation{'<'}/a>",
      "githubToken": "Personal access token",
      "githubTokenPlaceholder": "Paste your GitHub personal access token",
      "githubTokenHelperText": "Create one in GitHub account Settings. {'<'}a class='inline! font-normal! underline!' href='https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens' target='_blank'>See guide{'<'}/a>.",
      "youtrackToolMessage": "Report issues, manage agile boards and log work time.",
      "youtrackToolAlertMessage": "Check the {'<'}a class='inline! font-normal! underline!' href='https://www.jetbrains.com/help/youtrack/server/model-context-protocol-server.html' target='_blank'>API documentation{'<'}/a>",
      "youtrackServerUrl": "Server URL",
      "youtrackServerUrlPlaceholder": "https://your-instance.youtrack.cloud/mcp",
      "youtrackToken": "Permanent token",
      "youtrackTokenPlaceholder": "Paste your YouTrack permanent token",
      "youtrackTokenHelperText": "Create one in YouTrack account settings. {'<'}a class='inline! font-normal! underline!' href='https://www.jetbrains.com/help/youtrack/cloud/manage-permanent-token.html' target='_blank'>See guide{'<'}/a>.",
      "zephyrToolMessage": "Manage test cases, plans, cycles, and executions.",
      "zephyrToolAlertMessage": "Attachments are not supported due to Zephyr API limitations.",
      "zephyrProduct": "Product",
      "zephyrProductHelperText": "Choose the Zephyr product that matches your Jira instance. The access token must be generated for that product.",
      "zephyrProductCloud": "Zephyr Cloud",
      "zephyrProductEssential": "Zephyr Essential Cloud",
      "zephyrAccessToken": "Access token",
      "zephyrAccessTokenPlaceholder": "Paste your Zephyr API access token",
      "zephyrAccessTokenHelperText": "Create one in your Jira/Zephyr account settings. {'<'}a class='inline! font-normal! underline!' href='https://support.smartbear.com/zephyr/docs/en/rest-api/api-access-tokens-management.html' target='_blank'>See guide{'<'}/a> (Zephyr Cloud), {'<'}a class='inline! font-normal! underline!' href='https://smartbear.portal.swaggerhub.com/zephyr-squad/default/authentication#generate-a-key' target='_blank'>see guide{'<'}/a> (Essential Cloud).",
      "practitestToolMessage": "Manage tests, test sets and requirements.",
      "practitestToolAlertMessage": "To use this tool, provide your PractiTest MCP server URL.{'<'}br/>{'<'}br/>{'<'}span class='font-semibold'>US:{'<'}/span> https://api.practitest.com/mcp/v1/server{'<'}br/>{'<'}span class='font-semibold'>EU:{'<'}/span> https://eu1-prod-api.practitest.app/mcp/v1/server{'<'}br/>{'<'}br/>Check the {'<'}a class='inline! font-normal! underline!' href='https://www.practitest.com/help/integrations/mcp/' target='_blank'>API documentation{'<'}/a>.",
      "practitestServerUrl": "Server URL",
      "practitestServerUrlPlaceholder": "https://api.practitest.com/mcp/v1/server",
      "practitestToken": "Personal API Token",
      "practitestTokenPlaceholder": "Paste your PractiTest personal API token",
      "practitestTokenHelperText": "Create one in PractiTest account Settings. {'<'}a class='inline! font-normal! underline!' href='https://www.practitest.com/help/account/account-api-tokens/' target='_blank'>See guide{'<'}/a>.",
      "jiraClientId": "Client ID",
      "jiraClientIdPlaceholder": "Paste your OAuth client ID",
      "jiraClientSecret": "Client secret",
      "jiraClientSecretPlaceholder": "Paste your OAuth client secret",
      "jiraScope": "Scopes",
      "jiraScopeHelperText": "Learn more about {'<'}a class='inline! font-normal! underline!' href='https://developer.atlassian.com/cloud/jira/platform/scopes-for-oauth-2-3LO-and-forge-apps/' target='_blank'>Jira scopes{'<'}/a>.",
      "jiraScopeRead:jira-work": "Read",
      "jiraScopeWrite:jira-work": "Write",
      "jiraScopeRead:jira-user": "User Info",
      "azureDevopsToolMessage": "Read user stories, manage test plan cases, and link tests to stories.",
      "azureDevopsToolAlertMessage": "Register a Microsoft Entra app for Azure DevOps and use the redirect URL below:",
      "azureDevopsToolAlertGuideMessage": "{'<'}br/>Read the {'<'}a class='inline! font-normal! underline!' href='https://learn.microsoft.com/en-us/azure/devops/integrate/get-started/authentication/entra-oauth?view=azure-devops' target='_blank'>Entra OAuth setup guide{'<'}/a>.",
      "azureDevopsTenantId": "Tenant ID",
      "azureDevopsTenantIdPlaceholder": "Tenant GUID",
      "azureDevopsClientId": "Client ID",
      "azureDevopsClientIdPlaceholder": "Paste your Entra application client ID",
      "azureDevopsClientSecret": "Client secret",
      "azureDevopsClientSecretPlaceholder": "Paste your Entra application client secret",
      "azureDevopsScope": "Scopes",
      "azureDevopsScopeHelperText": "Grant the same delegated Azure DevOps permissions in the Entra app registration.",
      "azureDevopsScopeProfile": "Profile (read)",
      "azureDevopsScopeProject": "Projects and teams (read)",
      "azureDevopsScopeWork": "Work items (read)",
      "azureDevopsScopeWork_write": "Work items (write)",
      "azureDevopsScopeTest": "Test plans (read)",
      "azureDevopsScopeTest_write": "Test plans (write)",
      "redmineUrl": "Redmine URL",
      "redmineUrlPlaceholder": "https://redmine.example.com",
      "redmineApiKey": "API key",
      "redmineApiKeyPlaceholder": "Paste your Redmine API key",
      "redmineApiKeyHelperText": "Found in Redmine account Settings. {'<'}a class='inline! font-normal! underline!' href='https://www.redmine.org/projects/redmine/wiki/rest_api#Authentication' target='_blank'>See guide{'<'}/a>.",
      "webToolMessage": "Search the web and extract content from any URL.",
      "browserToolMessage": "Interact with web sites and capture screenshots.{'<'}br/>{'<'}br/>Check the {'<'}a class='inline! font-normal! underline!' href='https://hub.docker.com/r/mcp/playwright#available-tools-21' target='_blank'>documentation{'<'}/a> to see supported actions.",
      "sqlToolMessage": "Get and modify information on a database.",
      "sqlToolAlertMessage": "Since this tool can modify existing data, it is recommended to use a database user with only read permissions.",
      "sqlDbType": "Database type",
      "sqlDbTypeAzure": "Azure SQL Server",
      "sqlServer": "Server",
      "sqlServerPlaceholder": "myserver.database.windows.net",
      "sqlAzureTenantId": "Tenant ID",
      "sqlAzureTenantIdPlaceholder": "Paste your Tenant ID",
      "sqlAzureTenantIdHelperText": "Found in Azure Portal under Microsoft Entra ID → Overview.",
      "sqlAzureClientId": "Client ID",
      "sqlAzureClientIdPlaceholder": "Paste your Client ID",
      "sqlAzureClientIdHelperText": "Application (client) ID of your Azure AD app registration.",
      "sqlAzureClientSecret": "Client Secret",
      "sqlAzureClientSecretPlaceholder": "Paste your Client Secret",
      "sqlAzureClientSecretHelperText": "Client secret of your Azure AD app registration.",
      "advancedOptions": "Advanced options",
      "savingChanges": "Saving changes...",
      "errorSavingChanges": "Save failed",
      "changesSavedAutomatically": "Changes are saved automatically",
      "savedAt": "Saved at {time}"
    },
    "es": {
      "save": "Guardar",
      "add": "Agregar",
      "done": "Listo",
      "cancel": "Cancelar",
      "close": "Cerrar",
      "authenticationWindowBlocked": "No se pudo abrir la ventana de autenticación. Por favor, verifica que las ventanas emergentes estén permitidas para este sitio y vuelve a intentarlo.",
      "authenticationCancelled": "La autenticación fue cancelada. Por favor, complete el proceso de autenticación para configurar esta herramienta.",
      "authenticationAccessDenied": "La autenticación fue denegada por el servidor. Por favor, verifica que tengas los permisos necesarios para usarlo.",
      "missingProperty": "Se requiere un valor para '{property}'. Por favor proporciona uno.",
      "invalidPropertyFormat": "El valor proporcionado para '{property}' no es válido. Por favor, revise el valor y vuelve a intentarlo.",
      "invalidPropertyMinLength": "El valor proporcionado para '{property}' es más corto que {minLength} @:{'character'}. Por favor, revise el valor y vuelve a intentarlo.",
      "character": "caracter | caracteres",
      "uriFormat": "URL",
      "invalidToolConfiguration": "La configuración del tool falló. Por favor revise la configuración y vuelve a intentarlo. Si el problema persiste, contacta a {'<'}a href='mailto:{contactEmail}?subject=Tero%20Error'>{contactEmail}{'<'}/a>.",
      "docsToolMessage": "Usa la información de los archivos subidos en las respuestas del agente.",
      "docsToolAlertMessage": "{'<'}span class='font-semibold'>Solo sube archivos que sean relevantes para este agente.{'<'}/span>{'<'}br/>Esto ayuda a mejorar la calidad de las respuestas y evitar procesamiento innecesario.",
      "docsFiles": "Archivos",
      "docsAdvancedFileProcessing": "{'<'}span class='font-semibold'>Procesar nuevos PDF con IA avanzada{'<'}/span>",
      "docsAdvancedFileProcessingTooltipFalse": "El procesamiento básico utiliza un algoritmo simple para extraer el contenido del archivo. En general es menos preciso pero es más rápido y consume menos presupuesto. \n\nNota: Esta opción se aplicará unicamente a los nuevos archivos subidos.",
      "docsAdvancedFileProcessingTooltipTrue": "El procesamiento avanzado utiliza IA para extraer el contenido del archivo. En general es más preciso pero consume más presupuesto y puede tardar más en procesarse.\n\nNota: Esta opción se aplicará unicamente a los nuevos archivos subidos.",
      "mcpToolMessage": "Usa herramientas de cualquier servidor MCP.",
      "mcpToolAlertMessage": "{'<'}span class='font-semibold'>Solo conecta servidores que confíes.{'<'}/span>",
      "mcpServerUrl": "URL del servidor",
      "mcpServerUrlPlaceholder": "https://ejemplo.com/mcp",
      "mcpServerUrlHelperText": "Usa la URL completa para el endpoint del servidor MCP.",
      "mcpAuthType": "Método de autenticación",
      "mcpAuthTypeHelperText": "Selecciona el método de autenticación soportado por el MCP según la documentación.",
      "mcpAuthTypeOauth": "OAuth",
      "mcpAuthTypeBearerToken": "Bearer Token",
      "mcpBearerToken": "Bearer Token",
      "mcpBearerTokenPlaceholder": "Pega tu Bearer Token",
      "mcpBearerTokenHelperText": "Usa un token válido con los permisos requeridos por este servidor.",
      "customHeaders": "Headers",
      "jiraToolMessage": "Gestiona issues y sigue la actividad del proyecto.",
      "jiraToolAlertMessage": "Para usar esta herramienta, configura una app OAuth de Jira con la URL de redirección de abajo:",
      "jiraToolAlertGuideMessage": "{'<'}br/>Revisa la {'<'}a class='inline! font-normal! underline!' href='https://developer.atlassian.com/cloud/confluence/oauth-2-3lo-apps/#enabling-oauth-2-0--3lo-' target='_blank'>guía de configuración de Jira{'<'}/a>.",
      "redmineToolMessage": "Gestiona issues, proyectos y registra tiempo.",
      "redmineToolAlertMessage": "Consulta la {'<'}a class='inline! font-normal! underline!' href='https://www.redmine.org/projects/redmine/wiki/rest_api' target='_blank'>documentación de la API{'<'}/a>",
      "githubToolMessage": "Busca código, gestiona repos, revisa PRs y sigue issues.",
      "githubToolAlertMessage": "Consulta la {'<'}a class='inline! font-normal! underline!' href='https://github.com/github/github-mcp-server' target='_blank'>documentación de la API{'<'}/a>",
      "githubToken": "Personal access token",
      "githubTokenPlaceholder": "Pega tu personal access token de GitHub",
      "githubTokenHelperText": "Créalo en la configuración de tu cuenta de GitHub. {'<'}a class='inline! font-normal! underline!' href='https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens' target='_blank'>Ver guía{'<'}/a>.",
      "youtrackToolMessage": "Reporta issues, gestiona tableros ágiles y registra tiempo de trabajo.",
      "youtrackToolAlertMessage": "Consulta la {'<'}a class='inline! font-normal! underline!' href='https://www.jetbrains.com/help/youtrack/server/model-context-protocol-server.html' target='_blank'>documentación de la API{'<'}/a>",
      "youtrackServerUrl": "URL del servidor",
      "youtrackServerUrlPlaceholder": "https://tu-instancia.youtrack.cloud/mcp",
      "youtrackToken": "Token permanente",
      "youtrackTokenPlaceholder": "Pega tu token permanente de YouTrack",
      "youtrackTokenHelperText": "Puedes crearlo en la configuración de tu cuenta de YouTrack. {'<'}a class='inline! font-normal! underline!' href='https://www.jetbrains.com/help/youtrack/cloud/manage-permanent-token.html' target='_blank'>Ver guía{'<'}/a>.",
      "zephyrToolMessage": "Gestiona casos de prueba, planes, ciclos y ejecuciones.",
      "zephyrToolAlertMessage": "Los archivos adjuntos no son soportados debido a limitaciones de la API de Zephyr.",
      "zephyrProduct": "Producto",
      "zephyrProductHelperText": "Elige el producto Zephyr de tu instancia de Jira. El token de acceso debe generarse para ese producto.",
      "zephyrProductCloud": "Zephyr Cloud",
      "zephyrProductEssential": "Zephyr Essential Cloud",
      "zephyrAccessToken": "Token de acceso",
      "zephyrAccessTokenPlaceholder": "Pega tu token de acceso de la API Zephyr",
      "zephyrAccessTokenHelperText": "Créalo en la configuración de tu cuenta de Jira/Zephyr. {'<'}a class='inline! font-normal! underline!' href='https://support.smartbear.com/zephyr/docs/en/rest-api/api-access-tokens-management.html' target='_blank'>Ver guía{'<'}/a> (Zephyr Cloud), {'<'}a class='inline! font-normal! underline!' href='https://smartbear.portal.swaggerhub.com/zephyr-squad/default/authentication#generate-a-key' target='_blank'>ver guía{'<'}/a> (Essential Cloud).",
      "practitestToolMessage": "Gestiona tests, conjuntos de tests y requisitos.",
      "practitestToolAlertMessage": "Para usar esta herramienta, proporciona la URL del servidor MCP de PractiTest.{'<'}br/>{'<'}br/>{'<'}span class='font-semibold'>US:{'<'}/span> https://api.practitest.com/mcp/v1/server{'<'}br/>{'<'}span class='font-semibold'>EU:{'<'}/span> https://eu1-prod-api.practitest.app/mcp/v1/server{'<'}br/>{'<'}br/>Consulta la {'<'}a class='inline! font-normal! underline!' href='https://www.practitest.com/help/integrations/mcp/' target='_blank'>documentación de la API{'<'}/a>.",
      "practitestServerUrl": "URL del servidor",
      "practitestServerUrlPlaceholder": "https://api.practitest.com/mcp/v1/server",
      "practitestToken": "Token personal de API",
      "practitestTokenPlaceholder": "Pega tu token personal de API de PractiTest",
      "practitestTokenHelperText": "Crealo en la configuración de tu cuenta de PractiTest. {'<'}a class='inline! font-normal! underline!' href='https://www.practitest.com/help/account/account-api-tokens/' target='_blank'>Ver guía{'<'}/a>.",
      "jiraClientId": "ID del cliente",
      "jiraClientIdPlaceholder": "Pega tu OAuth client ID",
      "jiraClientSecret": "Secreto del cliente",
      "jiraClientSecretPlaceholder": "Pega tu OAuth client secret",
      "jiraScope": "Scopes",
      "jiraScopeHelperText": "Conoce más sobre los {'<'}a class='inline! font-normal! underline!' href='https://developer.atlassian.com/cloud/jira/platform/scopes-for-oauth-2-3LO-and-forge-apps/' target='_blank'>Jira scopes{'<'}/a>.",
      "jiraScopeRead:jira-work": "Leer",
      "jiraScopeWrite:jira-work": "Escribir",
      "jiraScopeRead:jira-user": "Info. del usuario",
      "azureDevopsToolMessage": "Lee user stories, gestiona casos del test plan y vincula tests con stories.",
      "azureDevopsToolAlertMessage": "Registra una app de Microsoft Entra para Azure DevOps y usa la URL de redirección de abajo:",
      "azureDevopsToolAlertGuideMessage": "{'<'}br/>Revisa la {'<'}a class='inline! font-normal! underline!' href='https://learn.microsoft.com/en-us/azure/devops/integrate/get-started/authentication/entra-oauth?view=azure-devops' target='_blank'>guía de Entra OAuth{'<'}/a>.",
      "azureDevopsTenantId": "Tenant ID",
      "azureDevopsTenantIdPlaceholder": "GUID del tenant",
      "azureDevopsClientId": "Client ID",
      "azureDevopsClientIdPlaceholder": "Pega el client ID de la app Entra",
      "azureDevopsClientSecret": "Secreto del cliente",
      "azureDevopsClientSecretPlaceholder": "Pega el client secret de la app Entra",
      "azureDevopsScope": "Scopes",
      "azureDevopsScopeHelperText": "Otorga los mismos permisos delegados de Azure DevOps en el registro de la app Entra.",
      "azureDevopsScopeProfile": "Perfil (lectura)",
      "azureDevopsScopeProject": "Proyectos y equipos (lectura)",
      "azureDevopsScopeWork": "Work items (lectura)",
      "azureDevopsScopeWork_write": "Work items (escritura)",
      "azureDevopsScopeTest": "Test plans (lectura)",
      "azureDevopsScopeTest_write": "Test plans (escritura)",
      "redmineUrl": "Redmine URL",
      "redmineUrlPlaceholder": "https://redmine.example.com",
      "redmineApiKey": "API key",
      "redmineApiKeyPlaceholder": "Pega tu API key de Redmine",
      "redmineApiKeyHelperText": "La encuentras en Mi cuenta de Redmine. {'<'}a class='inline! font-normal! underline!' href='https://www.redmine.org/projects/redmine/wiki/rest_api#Authentication' target='_blank'>Ver guía{'<'}/a>.",
      "webToolMessage": "Busca en la web y extrae contenido de cualquier URL.",
      "browserToolMessage": "Interactúa con sitios web y toma capturas de pantalla.{'<'}br/>{'<'}br/>Consulta la {'<'}a class='inline! font-normal! underline!' href='https://hub.docker.com/r/mcp/playwright#available-tools-21' target='_blank'>documentación{'<'}/a> para ver las acciones compatibles.",
      "sqlToolMessage": "Obtiene y modifica información en una base de datos.",
      "sqlToolAlertMessage": "Ya que esta herramienta puede modificar datos existentes, se recomienda usar un usuario de base de datos con permisos de solo lectura.",
      "sqlDbType": "Tipo de base de datos",
      "sqlDbTypeAzure": "Azure SQL Server",
      "sqlServer": "Servidor",
      "sqlServerPlaceholder": "myserver.database.windows.net",
      "sqlAzureTenantId": "Tenant ID",
      "sqlAzureTenantIdPlaceholder": "Pega tu Tenant ID",
      "sqlAzureTenantIdHelperText": "Lo encuentras en Azure Portal bajo Microsoft Entra ID → Resumen.",
      "sqlAzureClientId": "Client ID",
      "sqlAzureClientIdPlaceholder": "Pega tu Client ID",
      "sqlAzureClientIdHelperText": "ID de aplicación (cliente) de tu registro de app en Azure AD.",
      "sqlAzureClientSecret": "Client Secret",
      "sqlAzureClientSecretPlaceholder": "Pega tu Client Secret",
      "sqlAzureClientSecretHelperText": "Client secret de tu registro de app en Azure AD.",
      "advancedOptions": "Opciones avanzadas",
      "savingChanges": "Guardando cambios...",
      "errorSavingChanges": "Guardar falló",
      "changesSavedAutomatically": "Los cambios se guardan automáticamente",
      "savedAt": "Guardado a las {time}"
    }
  }
</i18n>
