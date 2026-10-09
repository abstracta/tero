<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { useBudgetStore } from '@/composables/useBudgetStore'
import { useErrorHandler } from '@/composables/useErrorHandler'
import moment from 'moment';

const { updateBudget, budgetStore } = useBudgetStore()
const { handleError } = useErrorHandler()
const { t } = useI18n()

const maxLowBudgetUsage = 0.50
const maxHighBudgetUsage = 0.80
const remainingDays = ref<number>(0);

onMounted(async () => {
  remainingDays.value = getRemainingDaysUntilNextMonth();
  try {
    await updateBudget()
  } catch (error) {
    handleError(error)
  }
})

const getRemainingDaysUntilNextMonth = (): number=> {
  const now = moment();
  const startOfNextMonth = now.clone().add(1, 'month').startOf('month');
  const diffInDays = startOfNextMonth.diff(now, 'days', true);
  return Math.ceil(diffInDays);
}

</script>
<template>
  <div
    class="rounded-full h-[12px] w-full relative"
    :class="{
      'bg-success-lighter': budgetStore.usage < maxLowBudgetUsage,
      'bg-warn-lighter': budgetStore.usage >= maxLowBudgetUsage && budgetStore.usage < maxHighBudgetUsage,
      'bg-error-lighter': budgetStore.usage >= maxHighBudgetUsage
    }">
    <div
      v-if="budgetStore.usage >= 0.01"
      class="rounded-full absolute top-0 left-0 h-full"
      :class="{
        'bg-success': budgetStore.usage < maxLowBudgetUsage,
        'bg-warn': budgetStore.usage >= maxLowBudgetUsage && budgetStore.usage < maxHighBudgetUsage,
        'bg-error-alt': budgetStore.usage >= maxHighBudgetUsage
      }"
      :style="{ width: (budgetStore.usage <= 1 ? budgetStore.usage * 100 : 100) + '%' }"
    ></div>
  </div>
  <div class="text-content-muted text-sm mt-2 mb-4 whitespace-nowrap">
    {{ t('consumedBudget', { budget: (budgetStore.usage <= 1 ? budgetStore.usage * 100 : 100).toFixed(1) }) }}
    <br/>
    {{ t(remainingDays > 1 ? 'renewDate_plural': 'renewDate_one', {days: remainingDays})}}
  </div>
</template>

<i18n lang="json">
  {
    "en": {
        "consumedBudget": "Token usage: {budget}% consumed",
        "renewDate_one": "Renews in {days} day",
        "renewDate_plural": "Renews in {days} days"
    },
    "es": {
        "consumedBudget": "Uso de tokens: {budget}% consumido",
        "renewDate_one": "Se renueva en {days} día",
        "renewDate_plural": "Se renueva en {days} días"
    }
  }
</i18n>