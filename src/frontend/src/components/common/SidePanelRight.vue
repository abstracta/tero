<script lang="ts" setup>
import { IconArrowLeft, IconLayoutSidebarRightCollapse } from '@tabler/icons-vue'
import { useSidePanel } from '@/composables/useSidePanel'

defineProps<{
  title: string
}>()

const { togglePanel, popPanel, canGoBack } = useSidePanel()
</script>

<template>
  <div class="h-full w-md">
    <FlexCard header-height="auto" class="h-full rounded-r-none">
      <template #header>
        <div class="flex items-center gap-2 w-full min-w-0">
          <SimpleButton class="shrink-0" @click="canGoBack ? popPanel() : togglePanel()">
            <IconArrowLeft v-if="canGoBack" />
            <IconLayoutSidebarRightCollapse v-else />
          </SimpleButton>
          <span class="min-w-0 flex-1 truncate">{{ title }}</span>
        </div>
      </template>
      <div class="flex flex-col flex-1 min-h-0">
        <div class="flex-1 min-h-0 overflow-y-auto overflow-x-hidden">
          <slot name="content" />
        </div>
        <div v-if="$slots.footer" class="shrink-0">
          <slot name="footer" />
        </div>
      </div>
    </FlexCard>
  </div>
</template>
