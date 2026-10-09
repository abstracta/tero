<script setup lang="ts">
import { ref, onMounted, computed } from 'vue';
import { IconDots, IconLogout, IconMessages, IconMoon, IconSun } from '@tabler/icons-vue';
import { useRouter } from 'vue-router';
import { useI18n } from 'vue-i18n';
import { useTheme, UiTheme } from '@/composables/useTheme';
import { useSidebar } from '@/composables/useSidebar';
import auth from '@/services/auth';

const { t } = useI18n();
const router = useRouter();
const { theme, toggleTheme } = useTheme();
const { isGroupedViewEnabled, setGroupedViewEnabled } = useSidebar();

const menu = ref();
const isMenuOpen = ref(false);
const userName = ref('');

onMounted(async () => {
  const user = await auth.getUser();
  userName.value = user?.profile?.name || user?.profile?.email || t('user');
});

const currentThemeLabel = computed(() => theme.value === UiTheme.DARK ? t('dark') : t('light'));
const groupedViewLabel = computed(() => isGroupedViewEnabled.value ? t('grouped') : t('list'));

const handleCloseMenu = () => {
  isMenuOpen.value = false;
};

const handleMenuToggle = (event: Event) => {
  menu.value.toggle(event);
};

const toggleGroupedView = () => {
  setGroupedViewEnabled(!isGroupedViewEnabled.value);
};

const logout = () => {
  router.push('/logout');
};
</script>

<template>
  <div class="px-2 sm:px-3 py-3">
    <div class="flex items-center justify-between">
      <div class="flex items-center gap-2 min-w-0 flex-1">
        <Avatar shape="circle" class="!bg-abstracta shrink-0">
          <template #icon>
            <span class="text-white text-sm font-medium">{{ userName.charAt(0).toUpperCase() }}</span>
          </template>
        </Avatar>
        <span class="text-sm font-medium text-content truncate">{{ userName }}</span>
      </div>
      <SimpleButton 
        class="rounded-lg shrink-0" 
        variant="muted" 
        size="small" 
        @click.prevent="handleMenuToggle($event)"
      >
        <IconDots stroke-width="2" :class="isMenuOpen ? 'text-abstracta' : 'text-content-muted'" />
      </SimpleButton>
    </div>
    
    <Menu 
      ref="menu" 
      :popup="true" 
      @show="isMenuOpen = true" 
      @hide="handleCloseMenu" 
      @keydown.esc="handleCloseMenu"
      class="!min-w-48"
    >
      <template #start>
        <div class="px-2 py-2">
          <button
            @click="toggleTheme"
            class="flex items-center justify-between w-full px-2 py-1.5 cursor-pointer hover:bg-surface-muted rounded-md transition-colors"
          >
            <div class="flex items-center gap-2">
              <component :is="theme === UiTheme.DARK ? IconMoon : IconSun" size="20" />
              <span class="text-sm">{{ t('themeLabel') }}: {{ currentThemeLabel }}</span>
            </div>
          </button>

          <button
            @click="toggleGroupedView"
            class="flex items-center justify-between w-full px-2 py-1.5 cursor-pointer hover:bg-surface-muted rounded-md transition-colors">
            <div class="flex items-center gap-2">
              <IconMessages size="20" />
              <span class="text-sm">{{ t('chats') }}: {{ groupedViewLabel }}</span>
            </div>
          </button>
          
          <button 
            @click="logout"
            class="flex items-center gap-2 w-full px-2 py-1.5 cursor-pointer hover:bg-surface-muted rounded-md transition-colors">
            <IconLogout size="20" />
            <span class="text-sm">{{ t('logout') }}</span>
          </button>
        </div>
      </template>
    </Menu>
  </div>
</template>

<i18n lang="json">
{
  "en": {
    "themeLabel": "Theme",
    "light": "Light",
    "dark": "Dark",
    "chats": "Chats",
    "grouped": "Grouped",
    "list": "List",
    "logout": "Log out",
    "user": "User"
  },
  "es": {
    "themeLabel": "Tema",
    "light": "Claro",
    "dark": "Oscuro",
    "chats": "Chats",
    "grouped": "Agrupados",
    "list": "Lista",
    "logout": "Cerrar sesión",
    "user": "Usuario"
  }
}
</i18n>
