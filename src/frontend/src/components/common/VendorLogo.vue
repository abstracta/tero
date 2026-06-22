<script setup lang="ts">
import { computed } from 'vue';
import { LlmModelVendor } from '@/services/api';
import openAiIcon from '@/assets/images/openai.svg?raw';
import googleIcon from '@/assets/images/gemini.svg?raw';
import anthropicIcon from '@/assets/images/anthropic.svg?raw';
import qwenIcon from '@/assets/images/qwen.svg?raw';
import amazonIcon from '@/assets/images/nova.svg?raw';

const vendorLogos: Record<LlmModelVendor, string> = {
  [LlmModelVendor.OPENAI]: openAiIcon,
  [LlmModelVendor.ANTHROPIC]: anthropicIcon,
  [LlmModelVendor.GOOGLE]: googleIcon,
  [LlmModelVendor.QWEN]: qwenIcon,
  [LlmModelVendor.AMAZON]: amazonIcon
};

const props = defineProps<{
  vendor: LlmModelVendor;
}>();

const icon = computed(() => vendorLogos[props.vendor]);
</script>

<template>
  <span v-html="icon" class="flex items-center" />
</template>
