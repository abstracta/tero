<script setup lang="ts">
import type { ExternalAgent } from '@/services/api'
import { IconCubeSpark } from '@tabler/icons-vue'

const { externalAgent } = defineProps<{
    externalAgent: ExternalAgent
    size?: 'normal' | 'large' | undefined
    bordered?: boolean
  }>()
</script>

<template>
  <Avatar :class="`'bg-surface' ${bordered ? 'border-1' : ''}`" shape="circle" :size="size">
    <template #icon>
      <IconCubeSpark v-if="!externalAgent.icon" />
      <img v-else :src="`data:image/png;base64,${externalAgent.icon}`" />
    </template>
  </Avatar>
</template>
