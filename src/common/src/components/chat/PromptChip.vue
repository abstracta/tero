<script setup lang="ts">
import { computed } from 'vue'
import { IconBook2 } from '@tabler/icons-vue'

const NAME_MAX_LENGTH = 25

const props = defineProps<{
  name: string
  selectable?: boolean
}>()

const emit = defineEmits<{
  (e: 'select'): void
}>()

const displayName = computed(() => {
  if (props.name.length <= NAME_MAX_LENGTH) return props.name
  return `${props.name.slice(0, NAME_MAX_LENGTH - 3)}...`
})
</script>

<template>
  <component
    :is="selectable ? 'button' : 'div'"
    :type="selectable ? 'button' : undefined"
    class="inline-flex items-center whitespace-nowrap rounded-xl bg-surface border px-3 py-1.5 mr-2"
    @click="selectable && emit('select')">
    <IconBook2 class="mr-2 shrink-0" />
    <span class="mr-2" v-tooltip.bottom="{ value: name, showDelay: 1000 }">{{ displayName }}</span>
    <div v-if="$slots.actions" class="flex items-center border-l pl-2 gap-2">
      <slot name="actions" />
    </div>
  </component>
</template>
