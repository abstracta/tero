<script setup lang="ts">
import { ref, onMounted, watch } from 'vue';
import { useI18n } from 'vue-i18n';
import { ExternalAgent } from '@/services/api';
import { ApiService } from '@/services/api';
import { useErrorHandler } from '@/composables/useErrorHandler'
import { IconCubeSpark } from '@tabler/icons-vue'

const { t } = useI18n();
const { handleError } = useErrorHandler();
const api = new ApiService();

const props = defineProps<{
    showModal: boolean;
}>();

const emit = defineEmits<{
    (e: 'update:showModal', value: boolean): void
    (e: 'newExternalAgentEntry'): void
}>();

const externalAgents = ref<ExternalAgent[]>([]);
const suggestions = ref<string[]>([]);
const selectedExternalAgent = ref<string | null>(null);
const date = ref<Date>(new Date());
const minutesSaved = ref<number>(0);
const showAddExternalAgentModal = ref<boolean>(false);
const formFieldErrors = ref<{
    date?: boolean;
    tool?: boolean;
    minutesSaved?: boolean;
}>({});
const submitting = ref<boolean>(false);
const maxMinutesSaved = 1440;

const hideRegisterExternalAgentModal = () => {
    emit('update:showModal', false);
}

const loadExternalAgents = async () => {
    try{
        externalAgents.value = await api.findExternalAgents();
        suggestions.value = externalAgents.value.map(agent => agent.name);
    } catch (error) {
        handleError(error);
    }
}

const search = (event: any) => {
    suggestions.value = externalAgents.value.filter((agent) => agent.name.toLowerCase().includes(event.query.toLowerCase())).map(agent => agent.name);
}

const handleAddExternalAgent = async () => {
    showAddExternalAgentModal.value = true;
}

const addNewExternalAgent = async () => {
    try{
        const externalAgent = await api.addExternalAgent({name: selectedExternalAgent.value as string});
        showAddExternalAgentModal.value = false;
        externalAgents.value.push(externalAgent);
        selectedExternalAgent.value = externalAgent.name;
        suggestions.value = externalAgents.value.map(agent => agent.name);
    } catch (error) {
        handleError(error);
    }
}

const validateForm = () => {
    let isValid = true;
    if(!selectedExternalAgent.value || !externalAgents.value.find(agent => agent.name === selectedExternalAgent.value)) {
        formFieldErrors.value.tool = true;
        isValid = false;
    }
    if(!date.value) {
        formFieldErrors.value.date = true;
        isValid = false;
    }
    if(!minutesSaved.value || minutesSaved.value < 1 || minutesSaved.value > maxMinutesSaved) {
        formFieldErrors.value.minutesSaved = true;
        isValid = false;
    }
    return isValid;
}

const registerExternalAgent = async () => {
    if(!validateForm()) return;
    submitting.value = true;
    try {
        await api.addExternalAgentTimeSaving(findAgentByName(selectedExternalAgent.value!)!.id, date.value, minutesSaved.value);
        hideRegisterExternalAgentModal();
        emit('newExternalAgentEntry');
    } catch (error) {
        handleError(error);
    } finally {
        submitting.value = false;
    }
}

const hideAddExternalAgentModal = () => {
    showAddExternalAgentModal.value = false;
}

const findAgentByName = (name: string) => {
    return externalAgents.value.find(agent => agent.name === name);
}

onMounted(async () => {
    await loadExternalAgents();
})

watch(selectedExternalAgent, () => {
    formFieldErrors.value.tool = false;
})

watch(date, () => {
    formFieldErrors.value.date = false;
})

watch(minutesSaved, () => {
    formFieldErrors.value.minutesSaved = false;
})

watch(() => props.showModal, () => {
    if(!props.showModal) {
        date.value = new Date();
        minutesSaved.value = 0;
        selectedExternalAgent.value = null;
        formFieldErrors.value = {}
    }
})
</script>

<template>
    <Dialog :visible="showModal" @update:visible="emit('update:showModal', $event)"
        @hide="hideRegisterExternalAgentModal" :modal="true" :closable="false" :draggable="false" :resizable="false"
        :close-on-escape="false" class="w-200">
        <template #header>
            <div class="flex justify-between items-center w-full border-b pb-4">
                <h3>{{ t('registerExternalAgentTitle') }}</h3>
                <div class="flex gap-4">
                    <SimpleButton @click="hideRegisterExternalAgentModal"><IconX /></SimpleButton>
                </div>
            </div>
        </template>
        <Form class="flex flex-col gap-5" @submit="registerExternalAgent">
            <div class="flex flex-col gap-3">
                <div>{{ t('registerExternalAgentDescription') }}</div>
            </div>
            <div class="flex gap-4">
                <div class="flex flex-col gap-2">
                    <label for="date" class="font-semibold">{{ t('date') }}</label>
                    <DatePicker v-model="date" showIcon fluid iconDisplay="input" inputId="date" name="date" :required="true" class="w-45"
                        :class="{'error': formFieldErrors.date}" :maxDate="new Date()" :manualInput="false">
                        <template #inputicon="slotProps">
                            <IconCalendarPlus
                                class="bg-surface-muted p-1 w-8 h-8 translate-y-[-.5rem] translate-x-[.3rem] rounded-lg"
                                @click="slotProps.clickCallback"></IconCalendarPlus>
                        </template>
                    </DatePicker>
                </div>
                <div class="flex flex-grow flex-col gap-2">
                    <label for="externalAgent" class="font-semibold">{{ t('tool') }}</label>
                    <AutoComplete v-model="selectedExternalAgent" name="tool" :required="true" :suggestions="suggestions"
                        @complete="search" dropdown inputId="externalAgent" :class="{'error': formFieldErrors.tool}">
                        <template #option="slotProps">
                            <div class="flex items-center gap-2">
                                <DashboardExternalAgentAvatar :externalAgent="findAgentByName(slotProps.option)!" />
                                <div>{{ slotProps.option }}</div>
                            </div>
                        </template>
                        <template #footer v-if="suggestions.length == 0">
                            <div class="p-1">
                                <div class="flex items-center gap-2 py-2 px-4 hover:bg-surface-muted cursor-pointer"
                                    @click="handleAddExternalAgent">
                                    <IconCubeSpark class="w-6 h-6" />
                                    <span class="">{{ t('addExternalAgentItem', {
                                        toolName: selectedExternalAgent
                                    }) }}</span>
                                </div>
                            </div>
                        </template>
                    </AutoComplete>
                </div>
                <div class="flex flex-col gap-2">
                    <span class="font-semibold">{{ t('minutesSaved') }}</span>
                    <InteractiveInput class="minutes-saved-input w-45" :class="{'error': formFieldErrors.minutesSaved}" v-model="minutesSaved" type="number" :min="0"
                        :max="maxMinutesSaved" name="minutesSaved" :required="true" />
                </div>
            </div>
            <div class="flex flex-row gap-2 justify-end">
                <SimpleButton type="submit" variant="primary" shape="square" :disabled="submitting">{{ t('register') }}
                </SimpleButton>
            </div>
        </Form>
    </Dialog>
    <Dialog v-model:visible="showAddExternalAgentModal"
        :header="t('addExternalAgentTitle', {toolName: selectedExternalAgent})" :modal="true" :draggable="false" :resizable="false" :closable="true"
        class="w-150" @hide="hideAddExternalAgentModal">
            <div class="flex flex-col gap-3">
                <div>{{ t('addExternalAgentDescription') }}</div>
            </div>
            <div class="flex flex-row gap-2 justify-end">
                <SimpleButton type="button" variant="secondary" shape="square" @click="hideAddExternalAgentModal">{{ t('cancel') }}
                </SimpleButton>
                <SimpleButton type="button" variant="primary" shape="square" @click="addNewExternalAgent">{{ t('add') }}
                </SimpleButton>
            </div>
    </Dialog>
</template>

<style scoped lang="scss">
@import '@/assets/styles.css';

:deep(.minutes-saved-input) input {
    @apply my-[.05rem];
}

:deep(.minutes-saved-input.error) input {
    @apply outline-error;
}
</style>

<i18n lang="json">{
    "en": {
        "registerExternalAgentTitle": "Register other agents",
        "registerExternalAgentDescription": "Record the minutes you saved while using other AI tools to have a complete view of your AI time savings.",
        "register": "Register",
        "date": "Date",
        "tool": "Tool",
        "minutesSaved": "Minutes saved",
        "addExternalAgentItem": "Add \"{toolName}\"",
        "addExternalAgentTitle": "Add new tool \"{toolName}\"",
        "addExternalAgentDescription": "When adding a new tool, it will be available to all users and cannot be deleted.",
        "add": "Add",
        "cancel": "Cancel"
    },
    "es": {
        "registerExternalAgentTitle": "Registrar otros agentes",
        "registerExternalAgentDescription": "Registrá los minutos que ahorraste al usar otras herramientas de IA para tener una visión completa de tu tiempo ahorrado con IA.",
        "register": "Registrar",
        "date": "Fecha",
        "tool": "Herramienta",
        "minutesSaved": "Minutos ahorrados",
        "addExternalAgentItem": "Agregar \"{toolName}\"",
        "addExternalAgentTitle": "Agregar nueva herramienta \"{toolName}\"",
        "addExternalAgentDescription": "Al agregar una nueva herramienta, esta estará disponible para todos los usuarios y no se podrá eliminar.",
        "add": "Agregar",
        "cancel": "Cancelar"
    }
}</i18n>
