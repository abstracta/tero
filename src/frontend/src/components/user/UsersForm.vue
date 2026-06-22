<script lang="ts">
    export class NewUserRow {
        username: string;
        role: Role;
        hasError?: boolean;
        constructor(username: string, role: Role) {
            this.username = username;
            this.role = role;
        }
        toNewUser(): NewUser {
            return new NewUser(this.username, this.role);
        }
    }
</script>

<script setup lang="ts">
import { Role, User, NewUser } from '@/services/api';
import { IconX } from '@tabler/icons-vue';
import type { AutoCompleteCompleteEvent } from 'primevue';
import { computed, ref, onBeforeUnmount } from 'vue';
import { useI18n } from 'vue-i18n';

const { t } = useI18n();
const props = defineProps<{
    users: User[];
    teamUsers: User[];
    loadingUsers: boolean;
    newUsers: NewUserRow[];
}>()

const emit = defineEmits<{
    (e: 'update:newUsers', value: NewUserRow[]): void
}>()

const usersSuggestions = ref<string[]>([]);

const filteredUsers = computed(() => {
    return props.users.filter(user => !props.teamUsers.some(teamUser => teamUser.id === user.id) && !props.newUsers.some(newUser => newUser.username === user.username));
})

const roleOptions = ref<{ label: string, value: string }[]>([
    { label: t('teamOwner'), value: Role.TEAM_OWNER.toString() },
    { label: t('teamMember'), value: Role.TEAM_MEMBER.toString() },
    { label: t('teamEditor'), value: Role.TEAM_EDITOR.toString() }
])

const handleUserChange = (user: NewUserRow, index: number) => {
    user.hasError = false;
    if (user.username && !isValidEmail(user.username)) {
        user.hasError = true;
        return;
    }
    if (user.username && index === props.newUsers.length - 1) {
        addUser();
    }
    clearEmptyUsers();
}

const addUser = () => {
    const updatedUsers = [...props.newUsers, new NewUserRow('', Role.TEAM_MEMBER)];
    updateNewUsers(updatedUsers);
}

const removeUser = (index: number) => {
    const updatedUsers = [...props.newUsers];
    updatedUsers.splice(index, 1);
    updateNewUsers(updatedUsers);
}

const onUserAutoCompleteSuggestionsRequest = (event: AutoCompleteCompleteEvent) => {
    const query = event.query.toLowerCase();
    usersSuggestions.value = [...filteredUsers.value.filter(user => user.username.toLowerCase().includes(query)).map(user => user.username)];
    if (usersSuggestions.value.length === 0 && isValidEmail(event.query)) {
        usersSuggestions.value = [event.query];
    }
}

const clearEmptyUsers = () => {
    const nonEmptyUsers = props.newUsers.filter((user) => user.username && user.username.length > 0)
    if (nonEmptyUsers.length < props.newUsers.length - 1) {
      const updatedUsers = [...nonEmptyUsers, new NewUserRow('', Role.TEAM_MEMBER)]
      updateNewUsers(updatedUsers);
    }
    return nonEmptyUsers
}

const clearAllUsers = () => {
    updateNewUsers([new NewUserRow('', Role.TEAM_MEMBER)]);
}

const filterUsersToAdd = () => {
    return props.newUsers.filter(user => user.username).map(user => user.toNewUser())
}

const updateNewUsers = (updatedUsers: NewUserRow[]) => {
    emit('update:newUsers', updatedUsers);
}

onBeforeUnmount(() => {
    clearAllUsers();
})

const handleEmptyMessage = (username: string): string => {
  if (!username) return '';
  if (isValidEmail(username)) return username;
  return t('invalidEmail');
}

const isValidEmail = (email: string): boolean => {
  return (/^[^\s@]+@[^\s@]+\.[^\s@]+$/).test(email);
}

defineExpose({
    isValidEmail,
    filterUsersToAdd,
})

</script>
<template>
    <div v-for="(user, index) in props.newUsers" :key="index" class="flex gap-4 mb-4 w-full">
        <div class="flex flex-col gap-2 w-full">
            <span v-if="index === 0">{{ t('userName') }}</span>
            <AutoComplete
                dropdown
                v-model="user.username"
                @change="handleUserChange(user, index)"
                @blur="handleUserChange(user, index)"
                class="w-full"
                :class="{'error': user.hasError}"
                @keydown.enter="handleUserChange(user, index)"
                :placeholder="t('userNamePlaceholder')"
                :suggestions="usersSuggestions"
                @complete="onUserAutoCompleteSuggestionsRequest"
                :force-selection="false"
                :loading="props.loadingUsers"
                :empty-search-message="handleEmptyMessage(user.username)" />
        </div>
        <div class="flex flex-col gap-2 w-100 max-w-[165px]">
            <span v-if="index === 0">{{ t('role') }}</span>
            <div class="flex gap-2 items-center">
                <Select v-model="user.role" :options="roleOptions" optionLabel="label" optionValue="value" class="w-full" />
                <SimpleIcon interactive @click="removeUser(index)" :icon="IconX" :size="24" :class="user.username ? '' : 'invisible'" />
            </div>
        </div>
    </div>
</template>

<i18n lang="json">
{
    "en": {
        "userName": "User name",
        "userNamePlaceholder": "Enter user email",
        "noUsersFound": "No users found",
        "role": "Role",
        "teamOwner": "Leader",
        "teamMember": "Member",
        "teamEditor": "Editor",
        "invalidEmail": "Invalid email"
    },
    "es": {
        "userName": "Nombre de usuario",
        "userNamePlaceholder": "Introduzca email de usuario",
        "noUsersFound": "No se encontraron usuarios",
        "role": "Rol",
        "teamOwner": "Líder",
        "teamMember": "Miembro",
        "teamEditor": "Editor",
        "invalidEmail": "Email inválido"
    }
}
</i18n>
