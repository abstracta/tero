<script setup lang="ts">
import { useI18n } from 'vue-i18n'
import { computed, onMounted, ref } from 'vue';

const { t } = useI18n()

const props = withDefaults(defineProps<{
  onSave: (value: string) => void
  onCancel: () => void
  value: string
  type?: 'text' | 'number'
  placeholder?: string
}>(), {
  type: 'text'
})

const isNumber = computed(() => props.type === 'number')
const inputValue = ref<string>(props.value)
const inputRef = ref<HTMLInputElement | null>(null);

onMounted(()=>{
  inputRef.value?.focus();
  if (isNumber.value) {
    inputRef.value?.select();
  }
})

</script>
<template>
  <div class="bg-surface rounded-xl shadow-md px-2 py-2 my-2 border">
    <div class="flex items-center justify-between gap-2">
      <Form class="flex w-full" @submit="onSave(inputValue)">
        <input
          ref="inputRef"
          :type="type"
          :maxlength="isNumber ? undefined : 80"
          @keydown.esc.prevent="props.onCancel()"
          v-model="inputValue"
          required="true"
          class="w-full px-1 border-none active:border-none focus:border-none focus:outline-none"
          :placeholder="placeholder ?? t('chatNamePlaceholder')"/>
        <div class="flex gap-2">
          <SimpleButton variant="secondary" size="small" @click="onCancel">
            <IconX/>
          </SimpleButton>
          <SimpleButton variant="primary" size="small" type="submit">
            <IconCheck/>
          </SimpleButton>
      </div>
    </Form>
    </div>
  </div>
</template>

<style scoped>
input[type='number']::-webkit-outer-spin-button,
input[type='number']::-webkit-inner-spin-button {
  -webkit-appearance: none;
  margin: 0;
}

input[type='number'] {
  -moz-appearance: textfield;
  appearance: textfield;
}
</style>

<i18n>
  {
    "en": {
      "chatNamePlaceholder": "Enter chat name",
    },
    "es": {
      "chatNamePlaceholder": "Ingrese el nombre del chat",
    }
  }
</i18n>
