<script setup lang="ts">
import ToolAuthModal from '@/components/common/ToolAuthModal.vue'
import { useToolAuthModal } from '@tero/common/utils/useToolAuthModal.js'

const { showToolAuthModal, toolAuthType, toolId, submitAuth, closeAuth } = useToolAuthModal()
</script>

<template>
  <PageLayout left-column-class="shrink-0">
    <template v-slot:left>
      <SidebarPanel />
    </template>
    <template v-slot:center>
      <router-view />
    </template>
  </PageLayout>
  <ToolAuthModal
    v-model:visible="showToolAuthModal"
    :tool-id="toolId"
    :auth-type="toolAuthType"
    @submit="submitAuth"
    @cancel="closeAuth"
  />
</template>
