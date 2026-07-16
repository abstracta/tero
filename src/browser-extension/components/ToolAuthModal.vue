<script lang="ts" setup>
import { useI18n } from 'vue-i18n'
import { ToolAuthType } from '@tero/common/utils/toolAuth.js'

const visible = defineModel<boolean>("visible")

const props = defineProps<{
  toolId: string
  authType: ToolAuthType
}>()

const emit = defineEmits<{
  (e: 'submit', token: string): void
  (e: 'cancel'): void
}>()

const { t } = useI18n()

const onSubmit = (token: string) => {
  emit('submit', token)
}

const onCancel = () => {
  visible.value = false
  emit('cancel')
}
</script>

<template>
  <Modal :show="visible ?? false" :title="t('authenticationRequired')" @close="onCancel">
    <ToolAuthModalContent
      :tool-id="toolId"
      :auth-type="authType"
      @submit="onSubmit"
      @cancel="onCancel"
    />
  </Modal>
</template>

<i18n lang="json">
{
  "en": {
    "authenticationRequired": "Authentication required"
  },
  "es": {
    "authenticationRequired": "Autenticación requerida"
  }
}
</i18n>
