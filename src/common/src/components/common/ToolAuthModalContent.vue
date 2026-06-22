<script lang="ts" setup>
import { ref, computed } from 'vue'
import { useI18n } from 'vue-i18n'
import SimpleButton from './SimpleButton.vue'
import InteractiveInput from './InteractiveInput.vue'
import { buildToolConfigName, findToolIcon, toolIdKey } from '@tero/common/utils/toolConfig.js'
import { ToolAuthType } from '@tero/common/utils/toolAuth.js'
import { IconLoader2 } from '@tabler/icons-vue'

const props = defineProps<{
  toolId: string
  authType: ToolAuthType
}>()

const emit = defineEmits<{
  (e: 'submit', token: string): void
  (e: 'cancel'): void
}>()

const { t, te } = useI18n()
const token = ref('')

const toolName = computed(() => buildToolConfigName(props.toolId))
const toolIcon = computed(() => findToolIcon(props.toolId))
const tokenLabel = computed(() => {
  const toolPrefix = props.toolId.split('-', 1)[0]
  const key = `tokenLabel${toolPrefix.charAt(0).toUpperCase()}${toolPrefix.slice(1)}`
  return te(key) ? t(key) : t('tokenLabel')
})

const onSubmit = () => {
  if (token.value.trim()) {
    emit('submit', token.value.trim())
    token.value = ''
  }
}

const onCancel = () => {
  token.value = ''
  emit('cancel')
}
</script>

<template>
  <div v-if="authType === ToolAuthType.OAUTH" class="flex flex-col items-center gap-4 py-8">
    <IconLoader2 class="size-12 text-abstracta animate-spin" />
    <p class="text-content text-center m-0">
      {{ t('authenticationContinuesInAnotherWindow') }}
      <span class="inline-block font-semibold gap-1">
        <component :is="toolIcon" class="inline-block size-5" />
        {{ toolName }}
      </span>
    </p>
    <p class="text-content-muted text-sm text-center mb-6">
      {{ t('completeAuthenticationOrCancel') }}
    </p>
    <div class="flex justify-center">
      <SimpleButton @click="onCancel" shape="square">{{ t('cancel') }}</SimpleButton>
    </div>
  </div>

  <form v-else @submit.prevent="onSubmit" class="flex flex-col gap-4">
    <p class="text-content-muted m-0">{{ t('enterToken') }}
      <span class="inline-block text-content font-semibold gap-1">
        <component :is="toolIcon" class="inline-block size-5" />
        {{ toolName }}
      </span>
    </p>
    <div class="flex flex-col gap-1">
      <label for="authToken" class="font-semibold">{{ tokenLabel }}</label>
      <InteractiveInput
        id="authToken"
        v-model="token"
        type="password"
        required
        autofocus
      />
    </div>
    <div class="flex gap-4 justify-end">
      <SimpleButton @click="onCancel" shape="square">{{ t('cancel') }}</SimpleButton>
      <SimpleButton
        type="submit"
        :disabled="!token.trim()"
        variant="primary"
        shape="square"
      >
        {{ t('submit') }}
      </SimpleButton>
    </div>
  </form>
</template>

<i18n lang="json">
{
  "en": {
    "authenticationContinuesInAnotherWindow": "Authentication is required for",
    "completeAuthenticationOrCancel": "Please complete the authentication process in the opened window, or cancel to abort.",
    "enterToken": "Please enter the token required to use",
    "tokenLabel": "Token",
    "tokenLabelGithub": "Personal Access Token",
    "tokenLabelYoutrack": "Permanent token",
    "tokenLabelPractitest": "Personal API Token",
    "cancel": "Cancel",
    "submit": "Submit"
  },
  "es": {
    "authenticationContinuesInAnotherWindow": "Se requiere autenticación para",
    "completeAuthenticationOrCancel": "Por favor complete el proceso de autenticación en la ventana abierta, o cancele para abortar.",
    "enterToken": "Por favor ingrese el token requerido para usar",
    "tokenLabel": "Token",
    "tokenLabelGithub": "Personal Access Token",
    "tokenLabelYoutrack": "Token permanente",
    "tokenLabelPractitest": "Token personal de API",
    "cancel": "Cancelar",
    "submit": "Enviar"
  }
}
</i18n>
