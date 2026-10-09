<script setup lang="ts">
import { IconBan, IconLock } from '@tabler/icons-vue'
import type { Agent } from '@/services/api'
import { computed } from 'vue'

const { agent, desaturated = false, showSharedStatus = false } = defineProps<{
    agent: Agent
    desaturated?: boolean,
    showSharedStatus?: boolean
    size?: 'normal' | 'large' | undefined
  }>()

const avatarBgColor = computed(() => {
  if (agent.icon) return '!bg-surface-muted'
  if (desaturated) return '!bg-auxiliar-gray dark:!bg-abstracta'
  return '!bg-abstracta-lighter'
})

</script>

<template>
  <div class="relative inline-flex shrink-0">
    <div class="flex rounded-full overflow-hidden" :class="{ 'grayscale': desaturated }">
      <Avatar class="border" :class="avatarBgColor" shape="circle" :size="size">
        <template #icon>
          <IconSparkles :size="size === 'large' ? '32' : '24'" v-if="!agent.icon" fill="currentColor" :color="desaturated ? 'var(--color-white)' : 'var(--color-abstracta)'"/>
          <img v-else :src="`data:image/png;base64,${agent.icon}`"/>
        </template>
      </Avatar>
    </div>
    <IconBan v-if="showSharedStatus && agent.readOnly" class="absolute -bottom-1 -right-1 bg-surface-muted text-content rounded-full p-0.5 border-[.1px] !w-4 !h-4" />
    <IconLock v-else-if="showSharedStatus && !agent.team" class="absolute -bottom-1 -right-1 bg-surface-muted text-content rounded-full p-0.5 border-[.1px] !w-4 !h-4" />
  </div>
</template>
