<script setup lang="ts">
const props = defineProps<{
  tooltip?: string
  loading?: boolean
}>()
const emit = defineEmits<{
  (e: 'confirm'): void
  (e: 'cancel'): void
}>()

</script>

<template>
  <div class="bg-surface rounded-xl shadow-md px-2 py-2 mx-1.5 my-2 border">
    <div class="flex items-center justify-between">
      <span class="text-md font-medium">{{ props.tooltip }}</span>
      <div class="relative flex gap-2 items-center">
        <div v-if="props.loading" class="absolute inset-0 flex items-center justify-end">
          <IconLoader class="animate-spin" />
        </div>
        <SimpleButton variant="secondary" size="small" :class="{ invisible: props.loading }" @click="emit('cancel')">
          <IconX/>
        </SimpleButton>
        <SimpleButton variant="error" size="small" :class="{ invisible: props.loading }" @click.prevent="emit('confirm')">
          <IconCheck/>
        </SimpleButton>
      </div>
    </div>
  </div>
</template>
