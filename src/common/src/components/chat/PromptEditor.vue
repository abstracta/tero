<script lang="ts">

</script>

<script lang="ts" setup>
import { useI18n } from 'vue-i18n'
import { reactive, ref, computed } from 'vue';
import { AgentPrompt } from '../../utils/domain';
import PromptContentDialog from './PromptContentDialog.vue';

const props = defineProps<{
    editingPrompt?: AgentPrompt,
    starter?: boolean,
    shareable?: boolean,
    sharedAgent?: boolean,
    newPromptText?: string,
    promptSaver: (prompt: AgentPrompt) => Promise<void>,
    errorHandler: (error: unknown) => void
}>()
const emit = defineEmits<{
    (e: 'close'): void
}>()

const { t } = useI18n();
const nameMaxLength = 50;
const prompt = reactive({
    name: props.editingPrompt?.name || "",
    instructions: props.editingPrompt?.content || props.newPromptText || "",
    visibility: props.editingPrompt?.shared || false,
    collapsed: props.editingPrompt?.collapsed || false
});
const originalPrompt = reactive({ ...prompt });
const showShareConfirmation = ref<boolean>(false)
const visibilitySwitchKey = ref(0)
const showCloseConfirmation = ref<boolean>(false)
const isModified = () => {
    return (
        prompt.name !== originalPrompt.name ||
        prompt.instructions !== originalPrompt.instructions ||
        prompt.visibility !== originalPrompt.visibility ||
        prompt.collapsed !== originalPrompt.collapsed
    );
};
const submitting = ref<boolean>(false)

const onClose = () => {
    if (isModified()) {
        showCloseConfirmation.value = true;
    } else {
        emit("close");
    }
}

const confirmPrompt = async () => {
    submitting.value = true;
    const payload = {
        name: prompt.name,
        content: prompt.instructions,
        shared: props.starter || prompt.visibility,
        starter: props.starter || false,
        collapsed: prompt.collapsed
    }
    try {
        await props.promptSaver(props.editingPrompt? {...props.editingPrompt, ...payload} : payload as AgentPrompt)
        emit("close")
    } catch (e) {
        props.errorHandler(e)
    } finally {
        submitting.value = false;
    }
}

const onConfirmShare = () => {
    prompt.visibility = !prompt.visibility;
    showShareConfirmation.value = false;
}

const title = computed(() => {
    const isEditing = !!props.editingPrompt?.id
    return props.starter ?
        (isEditing ? t('editStarterTitle') : t('createStarterTitle')) :
        (isEditing ? t('editPromptTitle') : t('createPromptTitle'))
})

const onShowShareConfirmation = () => {
    showShareConfirmation.value = true;
    visibilitySwitchKey.value += 1;
}
</script>

<template>
    <PromptContentDialog :visible="true" :title="title" @close="onClose">
        <template #icon>
            <IconPlayerPlay v-if="starter" />
            <IconBook2 v-else />
        </template>
            <div class="form-field flex flex-col gap-2 shrink-0">
                <label for="name" class="text-sm!">{{ t('name') }}</label>
                <InteractiveInput v-model="prompt.name" name="name" type="text" id="name" :placeholder="t('namePlaceholder')" :required="true"
                    :maxlength="nameMaxLength" />
            </div>
            <div class="form-field flex flex-col flex-1 gap-2">
                <label for="instructions" class="text-sm!">{{ t('instructions') }}</label>
                <InteractiveInput class="flex-1" v-model="prompt.instructions" name="instructions" :rows="10" :placeholder="t('instructionsPlaceholder')" id="instructions" :required="true" />
                <span class="text-sm text-content-muted pt-2">{{ t('inputParameterDescription') }}</span>
            </div>
            <div class="flex items-center justify-between gap-4 shrink-0 mt-3">
                <div class="flex items-center gap-6">
                    <div class="flex items-center gap-2" v-tooltip.top="t('collapsedDescription')">
                        <ToggleSwitch v-model="prompt.collapsed" name="collapsed" id="collapsed" />
                        <label for="collapsed">{{ t('collapsed') }}</label>
                    </div>
                    <div v-if="!starter && shareable" class="flex items-center gap-2" v-tooltip.top="t('publicDescription')">
                        <ToggleSwitch :key="visibilitySwitchKey" :model-value="prompt.visibility" name="visibility" @update:model-value="onShowShareConfirmation"/>
                        <label>{{ t('public') }}</label>
                    </div>
                </div>
                <SimpleButton @click="confirmPrompt" :disabled="submitting || !prompt.name || !prompt.instructions || prompt.name.length > nameMaxLength" class="px-6" variant="primary" shape="rounded">
                   {{ t('confirmButton') }}
                </SimpleButton>
            </div>
    </PromptContentDialog>

    <Dialog v-model:visible="showCloseConfirmation"
        :header="t('closeConfirmTitle')" :modal="true"
        :draggable="false" :resizable="false" :closable="false" class="max-w-150">
        <div class="flex flex-col gap-5">
            <div class="flex flex-row gap-2 items-start whitespace-pre-line">
                {{ t('closeConfirmDescription') }}
            </div>
            <div class="flex flex-row gap-2 justify-end">
                <SimpleButton @click="showCloseConfirmation = false" shape="square" variant="secondary">{{ t('continueEditButton') }}</SimpleButton>
                <SimpleButton @click="emit('close')" shape="square" variant="error">{{ t('discardChangesButton') }}</SimpleButton>
            </div>
        </div>
    </Dialog>

    <Dialog v-model:visible="showShareConfirmation"
        :header="t(!prompt.visibility ? 'shareConfirmationTitle' : 'unshareConfirmationTitle')" :modal="true"
        :draggable="false" :resizable="false" :closable="false" class="max-w-150">
        <div class="flex flex-col gap-5">
            <div class="flex flex-row gap-2 items-start whitespace-pre-line">
                {{ t(!prompt.visibility ? sharedAgent ?
                    'shareSharedAgentConfirmationMessage' : 'sharePrivateAgentConfirmationMessage' :
                    'unshareConfirmationMessage')
                }}
            </div>
            <div class="flex flex-row gap-2 justify-end">
                <SimpleButton @click="showShareConfirmation = false" shape="square" variant="secondary">{{ t('cancel') }}</SimpleButton>
                <SimpleButton @click="onConfirmShare" variant="primary" shape="square">{{ t(!prompt.visibility ? 'publish' : 'unpublish') }}</SimpleButton>
            </div>
        </div>
    </Dialog>
</template>

<i18n lang="json">
    {
      "en": {
        "createPromptTitle": "Create prompt",
        "createStarterTitle": "Create starter",
        "editPromptTitle": "Edit prompt",
        "editStarterTitle": "Edit starter",
        "name": "Name",
        "namePlaceholder": "Type the name",
        "instructions": "Instructions",
        "instructionsPlaceholder": "Type the instructions",
        "collapsed": "Collapsed",
        "collapsedDescription": "Show only the prompt name as a pill when this prompt is sent.",
        "public": "Public",
        "publicDescription": "Make this prompt visible to everyone using this agent.",
        "confirmButton": "Confirm",
        "inputParameterDescription": "Use the syntax {'{'}{'{'} Variable name {'}'}{'}'} for a text field, or {'{'}{'{'} Variable name {'|'} options: A, B {'}'}{'}'} for a dropdown.",
        "shareConfirmationTitle": "Do you want to make this prompt public?",
        "unshareConfirmationTitle": "Do you want to make this prompt private?",
        "shareSharedAgentConfirmationMessage": "When you make a prompt public it will be visible to everyone using this agent.\n\nAdditionally, all future modifications to the prompt will be immediately available to the rest of the users.",
        "sharePrivateAgentConfirmationMessage": "When you make a prompt public it will be visible to everyone using this agent, once the agent is made public.\n\nAdditionally, all future modifications to the prompt will be immediately available to the rest of the users.",
        "unshareConfirmationMessage": "When you make a prompt private it will no longer be visible to other users.",
        "unpublish": "Make private",
        "publish": "Make public",
        "cancel": "Cancel",
        "closeConfirmTitle": "Discard changes?",
        "closeConfirmDescription": "You have unsaved edits to this prompt. If you close now, all changes will be lost.",
        "continueEditButton": "Continue editing",
        "discardChangesButton": "Discard changes"
    },
      "es": {
        "createPromptTitle": "Crear prompt",
        "createStarterTitle": "Crear iniciador",
        "editPromptTitle": "Editar prompt",
        "editStarterTitle": "Editar iniciador",
        "name": "Nombre",
        "namePlaceholder": "Ingresa el nombre",
        "instructions": "Instrucciones",
        "instructionsPlaceholder": "Escribe las instrucciones",
        "collapsed": "Colapsado",
        "collapsedDescription": "Mostrar solo el nombre del prompt como una pastilla al enviarlo.",
        "public": "Público",
        "publicDescription": "Hacer este prompt visible para todos los que usen este agente.",
        "confirmButton": "Confirmar",
        "inputParameterDescription": "Usa la sintaxis {'{'}{'{'} Nombre de variable {'}'}{'}'} para un campo de texto, o {'{'}{'{'} Nombre de variable {'|'} options: A, B {'}'}{'}'} para una lista desplegable.",
        "shareConfirmationTitle": "¿Quieres hacer este prompt público?",
        "unshareConfirmationTitle": "¿Quieres hacer este prompt privado?",
        "shareSharedAgentConfirmationMessage": "Cuando haces un prompt público, este será visible para todos los que utilicen el agente.\n\nAdemás, todas las modificaciones futuras al prompt estarán disponibles inmediatamente para el resto de sus usuarios.",
        "sharePrivateAgentConfirmationMessage": "Cuando haces un prompt público, este será visible para todos los que utilicen el agente, una vez que el agente sea público.\n\nAdemás, todas las modificaciones futuras al prompt estarán disponibles inmediatamente para el resto de sus usuarios.",
        "unshareConfirmationMessage": "Cuando haces un prompt privado ya no será visible para el resto de usuarios.",
        "unpublish": "Hacer privado",
        "publish": "Hacer público",
        "cancel": "Cancelar",
        "closeConfirmTitle": "¿Descartar cambios?",
        "closeConfirmDescription": "Tienes cambios sin guardar en este prompt. Si cierras ahora, todos los cambios se perderán.",
        "continueEditButton": "Seguir editando",
        "discardChangesButton": "Descartar"
    }
}
</i18n>

<style scoped>
:deep(#instructions) {
  height: 100%;
}
:deep(#instructions textarea) {
  min-height: 100%;
}
</style>
