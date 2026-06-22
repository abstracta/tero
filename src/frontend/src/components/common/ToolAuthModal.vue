<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { Dialog } from 'primevue'
import SimpleButton from '@tero/common/components/common/SimpleButton.vue'
import ToolAuthModalContent from '@tero/common/components/common/ToolAuthModalContent.vue'
import { ToolAuthType } from '@tero/common/utils/toolAuth.js'
import { IconX } from '@tabler/icons-vue'

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
  <Dialog
    v-model:visible="visible"
    :modal="true"
    :draggable="false"
    :resizable="false"
    :closable="false"
    class="w-180"
    @update:visible="onCancel"
  >
    <template #header>
      <div class="flex justify-between items-center w-full border-b pb-4">
        <h3>{{ t('authenticationRequired') }}</h3>
        <SimpleButton @click="onCancel"><IconX /></SimpleButton>
      </div>
    </template>

    <ToolAuthModalContent
      :tool-id="toolId"
      :auth-type="authType"
      @submit="onSubmit"
      @cancel="onCancel"
    />
  </Dialog>
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
