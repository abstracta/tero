<script lang="ts" setup>
import { computed, ref, watch } from "vue";
import { ChatUiMessage } from "./ChatMessage.vue";
import { useI18n } from "vue-i18n";
import type { Popover } from "primevue";
import type { FormSubmitEvent } from "@primevue/forms";
import { IconThumbUp, IconThumbDown, IconThumbUpFilled, IconThumbDownFilled, IconLoader } from "@tabler/icons-vue";
import { UserFeedback } from "../../utils/domain";

const props = defineProps<{
  message: ChatUiMessage
  actionsEnabled: boolean
  isFeedbackLoading?: boolean
}>()
const emit = defineEmits<{
  (e: 'feedbackChange', feedback?: UserFeedback): void
}>()
const { t } = useI18n();
const minutesSavedError = ref<boolean>(false);
const feedbackTextError = ref<boolean>(false);
const feedbackPopRef = ref<InstanceType<typeof Popover> | null>(null);
const isPositiveFeedback = ref<boolean>(true);

const getParentChainMinutesSaved = (message: ChatUiMessage): number => {
  if (!message.parent) return 0
  return (message.parent.minutesSaved ?? 0) + getParentChainMinutesSaved(message.parent)
}

const parentMinutesSaved = computed(() => getParentChainMinutesSaved(props.message))
const minutesSavedPending = computed(
  () =>
    !props.message.isComplete &&
    props.message.minutesSaved === null &&
    !props.message.stopped,
)
const minutesSaved = computed(() => {
  if (props.message.minutesSaved == null) return null
  return parentMinutesSaved.value + props.message.minutesSaved
})

const formValues: { minutes: number; feedbackText?: string } = {
  minutes: minutesSaved.value ?? 0,
  feedbackText: props.message.feedbackText
}
const feedbackMesssage = ref<string>("");


const handleToggleFeedback = async ($event: MouseEvent, isPositive: boolean) => {
  isPositiveFeedback.value = isPositive;
  minutesSavedError.value = false;
  feedbackTextError.value = false;
  formValues.minutes = minutesSaved.value ?? 0;
  formValues.feedbackText = props.message.feedbackText;
  if (feedbackPopRef.value) {
    feedbackPopRef.value.toggle($event);
  }
}

const handleSubmitFeedback = async (e: FormSubmitEvent) => {
  feedbackTextError.value = false;
  minutesSavedError.value = false;
  if (isPositiveFeedback.value && !formValues.minutes) {
    minutesSavedError.value = true;
    return
  }
  if (!isPositiveFeedback.value && !formValues.feedbackText) {
    feedbackTextError.value = true;
    return
  };
  const minutesSaved = isPositiveFeedback.value ? formValues.minutes - (parentMinutesSaved.value ?? 0): - parentMinutesSaved.value;
  emit("feedbackChange", new UserFeedback(isPositiveFeedback.value, minutesSaved, formValues.feedbackText))
  feedbackPopRef.value?.hide();
  showFeedbackMessage(t('feedbackThanksMessage'));
}

const handleRemoveFeedback = () => {
  emit("feedbackChange")
}

const showFeedbackMessage = (message: string) => {
  feedbackMesssage.value = message;
  setTimeout(() => {
    feedbackMesssage.value = "";
  }, 3000);
}

watch(props.message, (newMessage:ChatUiMessage) => {
  formValues.minutes = minutesSaved.value ?? 0;
  formValues.feedbackText = newMessage.feedbackText;
})

</script>

<template>
  <div :class="`${!actionsEnabled ? 'invisible' : ''}`" class="flex min-h-[30px]">
    <div class="flex w-full justify-end">
      <div v-if="minutesSavedPending" class="flex items-center border-1 rounded-xl px-3 py-2">
        <IconLoader class="!text-content animate-spin" />
      </div>
      <div v-else-if="minutesSaved != null || message.stopped" class="flex items-center border-1 rounded-xl">
        <div v-if="!feedbackMesssage" class="flex items-center gap-3 pl-3 p-2">
          <span class="text-content text-sm"> {{ t('responseSavedMessage') }} <span class="font-medium">{{minutesSaved}} {{ t(minutesSaved != 1 ? 'minutes':'minute')}}</span></span>
          <div v-if="message.hasPositiveFeedback == null" class="flex items-center gap-2">
            <SimpleIcon interactive class="!text-content" :icon="IconThumbUp" v-tooltip.bottom="t('thumbsUpButton')" @click="handleToggleFeedback($event, true)"/>
            <SimpleIcon interactive class="!text-content" :icon="IconThumbDown" v-tooltip.bottom="t('thumbsDownButton')" @click="handleToggleFeedback($event, false)"/>
          </div>
          <div v-else class="flex items-center gap-2">
            <IconLoader v-if="isFeedbackLoading" class="!text-abstracta animate-spin"/>
            <SimpleIcon interactive v-else-if="message.hasPositiveFeedback" class="!text-abstracta"
              :icon="IconThumbUpFilled" v-tooltip.bottom="t('removeFeedbackButton')" @click="handleRemoveFeedback()"/>
            <SimpleIcon interactive v-else class="!text-abstracta"
              :icon="IconThumbDownFilled" v-tooltip.bottom="t('removeFeedbackButton')" @click="handleRemoveFeedback()"/>
          </div>
        </div>
        <div v-else class="flex items-center h-6 px-3 py-5">
          <span class="text-content text-sm"> {{ feedbackMesssage }}</span>
        </div>
        <Popover ref="feedbackPopRef" class="shadow-light" id="feedback-popover">
          <div class="flex flex-col gap-4">
            <Form @submit="handleSubmitFeedback" class="flex" :class="isPositiveFeedback ? ' items-center gap-2' : 'flex-col items-end'">
              <div class="w-full" v-if="!isPositiveFeedback">
                <!-- Using tailwindcss h-[] class is causing scroll to not working properly, using max-height inline style instead solves this issue -->
                <Textarea v-model="formValues.feedbackText" class="borderless-textarea resize-none !w-130"
                  :class="feedbackTextError ? '!outline-1 !outline-error' : ''" autofocus="true" name="feedbackText"
                  :auto-resize="true" :style="{ 'max-height': '150px' }" :placeholder="t('negativeFeedbackPlaceholder')"></Textarea>
              </div>
              <div v-else class="flex items-center gap-2">
                <span class="text-base">{{ t('positiveFeedbackMessage') }}</span>
                <InputNumber v-model="formValues.minutes" name="minutes" class="w-15 rounded-xl !h-9 feedback-minutes-saved-input" :class="minutesSavedError ? 'border-error':''" :min="0"></InputNumber>
              </div>
              <SimpleButton type="submit" variant="primary" shape="square">{{ t('sendFeedbackButton') }}</SimpleButton>
            </Form>
          </div>
        </Popover>
      </div>
    </div>
  </div>
</template>

<i18n lang="json">{
  "en": {
    "responseSavedMessage": "This chat saved you approx.",
    "minute": "minute",
    "minutes": "minutes",
    "sendFeedbackButton": "Send",
    "feedbackThanksMessage": "Thank you for your feedback!",
    "positiveFeedbackMessage": "How many minutes did this chat save you?",
    "negativeFeedbackPlaceholder": "Please let us know how can we improve the answer",
    "thumbsUpButton": "Adjust time",
    "thumbsDownButton": "Not useful",
    "removeFeedbackButton": "Remove feedback"
  },
  "es": {
    "responseSavedMessage": "Este chat te ahorró aprox.",
    "minute": "minuto",
    "minutes": "minutos",
    "sendFeedbackButton": "Enviar",
    "positiveFeedbackMessage": "¿Cuántos minutos te ahorró este chat?",
    "negativeFeedbackPlaceholder": "Por favor dinos como podemos mejorar esta respuesta",
    "feedbackThanksMessage": "¡Muchas gracias por tu feedback!",
    "thumbsUpButton": "Ajustar tiempo",
    "thumbsDownButton": "No me fue util",
    "removeFeedbackButton": "Quitar feedback"
  }
}
</i18n>

<style lang="scss">
:deep(.feedback-minutes-saved-input.p-inputnumber.border-error) .p-inputtext{
  @apply border-inherit
}
</style>

<style lang="scss">
#feedback-popover.p-popover:after, #feedback-popover.p-popover:before {
  left: calc(.6rem + var(--p-popover-arrow-left)) !important
}
</style>
