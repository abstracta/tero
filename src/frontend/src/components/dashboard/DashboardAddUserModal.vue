
<script setup lang="ts">
import { onBeforeMount, ref, watch, computed } from 'vue';
import { useI18n } from 'vue-i18n';
import { ApiService, User, Role } from '@/services/api';
import { IconX, IconAlertTriangleFilled, IconSquareCheckFilled } from '@tabler/icons-vue';
import { useErrorHandler } from '@/composables/useErrorHandler';
import type UsersForm from '@/components/user/UsersForm.vue';
import { NewUserRow } from '@/components/user/UsersForm.vue';

const api = new ApiService();

const props = defineProps<{
    showModal: boolean;
    teamId: number;
    canAssignBudgetOwner?: boolean;
}>();

const emit = defineEmits(['update:showModal', 'usersAdded']);

const { t } = useI18n();
const { handleError } = useErrorHandler();
const showConfirmClose = ref<boolean>(false);
const loadingUsers = ref<boolean>(true);
const showSuccess = ref<boolean>(false);
const usersFormRef = ref<InstanceType<typeof UsersForm> | null>(null);
const users = ref<User[]>([]);
const teamUsers = ref<User[]>([])
const newUsers = ref<NewUserRow[]>([new NewUserRow('', Role.TEAM_MEMBER)]);
const newUsersNames = ref<string[]>([]);
const externalUsers = ref<string[]>([]);
const showConfirmExternalUsers = ref<boolean>(false);

const hasChanges = computed(() => {
    return newUsers.value.some(user => user.username);
});

const hasNewUsers = computed(() => {
    const usersWithContent = newUsers.value.filter(user => user.username.trim());
    return usersWithContent.length > 0 && usersWithContent.every(user => usersFormRef.value?.isValidEmail(user.username));
});

onBeforeMount(async () => {
    try{
        await loadUsers();
        await loadTeamUsers();
    } catch (error) {
        handleError(error);
    }
});

const loadTeamUsers = async () => {
    try{
        loadingUsers.value = true;
        teamUsers.value = await api.findTeamUsers(props.teamId);
    } catch (error) {
        handleError(error);
    } finally {
        loadingUsers.value = false;
    }
}

const loadUsers = async () => {
    try{
        users.value = await api.findUsers();
    } catch (error) {
        handleError(error);
    }
}

const addUsersToTeam = async () => {
    try {
        const usersToAdd = usersFormRef.value?.filterUsersToAdd()!;
        await api.addUsersToTeam(props.teamId, usersToAdd);
        newUsersNames.value = usersToAdd.map(user => user.username);
        showSuccess.value = true;
        closeAddUsersModal();
    } catch (error) {
        handleError(error);
    }
}

const checkExternalUsers = async () => {
    const existingUserEmails = users.value.map(user => user.username.toLowerCase()); 
    externalUsers.value = newUsers.value.filter(user => user.username && !existingUserEmails.includes(user.username.toLowerCase())).map(user => user.username);
    if(externalUsers.value.length > 0) {
        showConfirmExternalUsers.value = true;
        return;
    } else {
        await addUsersToTeam();
    }
}

const confirmWithExternalUsers = async () => {
    showConfirmExternalUsers.value = false;
    await addUsersToTeam();
    externalUsers.value = [];
}

const handleHideAddUsersModal = () => {
    if(hasChanges.value && !showSuccess.value) {
        showConfirmClose.value = true;
        return;
    }
    closeAddUsersModal();
}

const closeAddUsersModal = () => {
    emit('update:showModal', false);
}

const handleConfirmClose = () => {
    showConfirmClose.value = false;
    closeAddUsersModal();
}

const handleCloseSuccessModal = () => {
    showSuccess.value = false;
    emit('usersAdded');
}

watch(() => props.showModal, async () => {
    if(props.showModal) {
        await loadTeamUsers();
    }
});

</script>

<template>
    <Dialog :visible="showModal" @update:visible="emit('update:showModal', $event)"
        @hide="handleHideAddUsersModal" :modal="true" :closable="false" :draggable="false" :resizable="false"
        :close-on-escape="false" :dismissable-mask="false" class="w-150">
        <template #header>
            <div class="flex justify-between items-center w-full border-b pb-4">
                <h3>{{ t('addUser') }}</h3>
                <div class="flex gap-4">
                    <SimpleButton @click="handleHideAddUsersModal"><IconX /></SimpleButton>
                </div>
            </div>
        </template>
        <UsersForm
          v-model:newUsers="newUsers" 
          :users="users" 
          :teamUsers="teamUsers" 
          :loadingUsers="loadingUsers"
          :can-assign-budget-owner="canAssignBudgetOwner"
          ref="usersFormRef"
        />
        <template #footer>
            <div class="flex gap-4 mt-3">
                <SimpleButton @click="handleHideAddUsersModal" shape="square">
                    {{ t('cancel') }}
                </SimpleButton>
                <SimpleButton @click="checkExternalUsers" shape="square" variant="primary" :disabled="!hasNewUsers">
                    {{ t('confirm') }}
                </SimpleButton>
            </div>
        </template>
    </Dialog>
    <Dialog v-model:visible="showConfirmClose" :header="t('confirmClose')" :modal="true" :closable="false" :draggable="false" :resizable="false"
        :close-on-escape="true" :dismissable-mask="false" class="w-150">
        <p>{{ t('confirmCloseDescription') }}</p>
        <template #footer>
            <SimpleButton @click="showConfirmClose = false" shape="square">{{ t('continueEditing') }}</SimpleButton>
            <SimpleButton @click="handleConfirmClose" shape="square" variant="error">{{ t('discard') }}</SimpleButton>
        </template>
    </Dialog>
    <Dialog v-model:visible="showSuccess" :header="t('successTitle')" :modal="true" :closable="false" :draggable="false" :resizable="false"
        :close-on-escape="true" :dismissable-mask="false" class="w-150">
        <template #header>
            <div class="flex items-center gap-2 w-full border-b pb-4">
                <IconSquareCheckFilled class="text-success"/>
                <h3>{{ t('successTitle') }}</h3>
            </div>
        </template>
        <div class="flex flex-col gap-4">
            <p v-html="t('successDescription', { count: newUsersNames.length, userNames: newUsersNames.join(', ') })"></p>
        </div>
        <template #footer>
            <SimpleButton @click="handleCloseSuccessModal" shape="square" variant="primary">{{ t('successButton') }}</SimpleButton>
        </template>
    </Dialog>
    <Dialog v-model:visible="showConfirmExternalUsers" :header="t('externalUsersTitle')" :modal="true" :closable="false" :draggable="false" :resizable="false"
        :close-on-escape="true" :dismissable-mask="false" class="w-150">
        <template #header>
            <div class="flex items-center gap-2 w-full border-b pb-4">
                <IconAlertTriangleFilled color="var(--color-warn)"/>
                <h3>{{ t('externalUsersTitle') }}</h3>
            </div>
        </template>
        <div class="flex flex-col gap-4">
            <p class="whitespace-pre-line" v-html="t('externalUsersDescription', { emails: externalUsers.join('\n') })"></p>
        </div>
        <template #footer>
            <SimpleButton @click="showConfirmExternalUsers = false" shape="square">{{ t('cancel') }}</SimpleButton>
            <SimpleButton @click="confirmWithExternalUsers" shape="square" variant="primary">{{ t('confirm') }}</SimpleButton>
        </template>
    </Dialog>
</template>
<i18n lang="json">{
    "en": {
        "addUser": "Add users to team",
        "addUserButton": "Add",
        "userName": "User name",
        "role": "Role",
        "teamOwner": "Leader",
        "teamMember": "Member",
        "userNamePlaceholder": "Enter user email",
        "cancel": "Cancel",
        "confirm": "Confirm",
        "noUsersToAdd": "Add at least one user",
        "confirmClose": "Discard changes?",
        "confirmCloseDescription": "The selected users will not be added to the team.",
        "continueEditing": "Continue editing",
        "discard": "Discard",
        "noUsersFound": "No users found",
        "successTitle": "Invitations sent",
        "successDescription": "The invitation to {'<'}b>{userNames}{'<'}/b> has been sent successfully. {'<'}/br>{'<'}/br> The status will remain pending until the person accepts to join the team. | The invitations to {'<'}b>{userNames}{'<'}/b> have been sent successfully.{'<'}/br>{'<'}/br> The status will remain pending until they accept to join the team.",
        "successButton": "OK",
        "externalUsersTitle": "External users",
        "externalUsersDescription": "The following users are external:\n{'<'}b>{emails}{'<'}/b> \n\nDo you still want to send an invitation to these users?"
    },
    "es": {
        "addUser": "Agregar usuarios al equipo",
        "addUserButton": "Agregar",
        "userName": "Nombre de usuario",
        "role": "Rol",
        "teamOwner": "Líder",
        "teamMember": "Miembro",
        "userNamePlaceholder": "Introduzca email de usuario",
        "cancel": "Cancelar",
        "confirm": "Confirmar",
        "noUsersToAdd": "Agregue al menos un usuario",
        "confirmClose": "¿Descartar cambios?",
        "confirmCloseDescription": "Los usuarios seleccionados no serán agregados al equipo.",
        "continueEditing": "Continuar editando",
        "discard": "Descartar",
        "noUsersFound": "No se encontraron usuarios",
        "successTitle": "Invitaciones enviadas",
        "successDescription": "La invitación a {'<'}b>{userNames}{'<'}/b> fue enviada con éxito. {'<'}/br> {'<'}/br>El estado quedará como pendiente hasta que la persona acepte unirse al equipo. | Las invitaciones a {'<'}b>{userNames}{'<'}/b> fueron enviadas con éxito. {'<'}/br> {'<'}/br>El estado quedará como pendiente hasta que las personas acepten unirse al equipo.",
        "successButton": "Entendido",
        "externalUsersTitle": "Usuarios externos",
        "externalUsersDescription": "Los siguientes usuarios son externos:\n{'<'}b>{emails}{'<'}/b> \n\n¿Aún desea enviar una invitación a estos usuarios?"
    }
}</i18n>
