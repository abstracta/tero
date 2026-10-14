<script lang="ts" setup>
import { useI18n } from 'vue-i18n'
import { onMounted, ref, watch, computed } from 'vue'
import { IconSparkles, IconUsers, IconSettings, IconMessages } from '@tabler/icons-vue'
import { Team, MY_TEAM_ID, GLOBAL_TEAM_ID, TeamRoleStatus, UserProfile, OWNER_ROLES } from '@/services/api'
import { loadUserProfile } from '@/composables/useUserProfile'
import { useErrorHandler } from '@/composables/useErrorHandler'
import { useSelectedTeam } from '@/composables/useSelectedTeam'
import DashboardImpact from '@/components/dashboard/DashboardImpact.vue'
import DashboardUsers from '@/components/dashboard/DashboardUsers.vue'
import DashboardTeams from '@/components/dashboard/DashboardTeams.vue'
import DashboardTabsSkeleton from '@/components/dashboard/DashboardTabsSkeleton.vue'
import DashboardUsage from '@/components/dashboard/DashboardUsage.vue'

const IMPACT_TAB = '0'
const USERS_TAB = '1'
const TEAMS_TAB = '2'
const USAGE_TAB = '3'

const { t } = useI18n()
const { handleError } = useErrorHandler()
const { selectedTeam, updateSelectedTeam } = useSelectedTeam()

const teams = ref<Team[]>([])
const globalTeamOwner = ref<boolean | undefined>(undefined)
const userProfile = ref<UserProfile | null>(null)
const activeTab = ref(IMPACT_TAB)
const loading = ref<boolean>(true)

onMounted(async () => {
  await loadTeams()
  updateSelectedTeam(teams.value[0]?.id ?? null)
  loading.value = false
})

watch(
  () => selectedTeam.value,
  async () => {
    await loadTeams()
  }
)

const loadTeams = async () => {
  userProfile.value = await loadUserProfile()
  try {
    const userTeams = (userProfile.value?.teams ?? []).filter((t) => OWNER_ROLES.includes(t.role) && t.status === TeamRoleStatus.ACCEPTED).map((t) => new Team(t.id, t.name))
    const globalTeam = userTeams.find((t) => t.id === GLOBAL_TEAM_ID)
    const otherTeams = userTeams.filter((t) => t.id !== GLOBAL_TEAM_ID).sort((a, b) => a.name.localeCompare(b.name))

    teams.value = [...(activeTab.value === IMPACT_TAB || activeTab.value === USAGE_TAB ? [new Team(MY_TEAM_ID, t('me'))] : []), ...(globalTeam ? [globalTeam] : []), ...otherTeams]
    globalTeamOwner.value = !!globalTeam
  } catch (error) {
    handleError(error)
  }
}

const handleInvitationAccepted = async () => {
  await loadTeams()
}

const tabs = computed(() => {
  const baseTabs = [
    { label: t('impactLabel'), value: IMPACT_TAB, icon: IconSparkles, component: DashboardImpact },
    { label: t('usageLabel'), value: USAGE_TAB, icon: IconMessages, component: DashboardUsage }
  ]
  
  if (userProfile.value?.teams.some((t) => OWNER_ROLES.includes(t.role) && t.status === TeamRoleStatus.ACCEPTED)) {
    baseTabs.push({ label: t('usersLabel'), value: USERS_TAB, icon: IconUsers, component: DashboardUsers })
  }
  
  if (globalTeamOwner.value) {
    baseTabs.push({ label: t('teamsLabel'), value: TEAMS_TAB, icon: IconSettings, component: DashboardTeams })
  }

  return baseTabs
})

const handleTabChange = async (path: string) => {
  activeTab.value = path
  await loadTeams()
  if (selectedTeam.value === MY_TEAM_ID) {
    updateSelectedTeam(teams.value[0]?.id ?? null)
  }
}

</script>

<template>
  <FlexCard>
    <template #header>
      <RightPanelHeader :title="t('dashboardTitle')" icon="IconSparkles" :invitation-accepted="true" @invitation-accepted="handleInvitationAccepted" />
    </template>
    <div class="flex flex-col h-full w-full relative overflow-hidden">
      <Tabs :value="activeTab" :key="activeTab" class="container h-full mx-auto p-4 pt-0 max-w-6xl flex flex-col gap-4">
        <DashboardTabsSkeleton v-if="loading" :tabs-length="4"  />
        <div v-else class="flex items-center w-full border-b-1 pb-2 min-h-[51px]" :class="{ 'justify-end': !globalTeamOwner, 'justify-between': globalTeamOwner }">
          <TabList class="!p-0 !w-full !border-none">
            <Tab v-for="tab in tabs" :key="tab.value" :value="tab.value" @click="handleTabChange(tab.value)" class="!p-0 !border-none">
              <div class="flex items-center gap-2">
                <component :is="tab.icon" size="20"  />
                <span :class="{ 'font-medium': activeTab === tab.value }">{{ tab.label }}</span>
              </div>
            </Tab>
          </TabList>
          <div v-if="activeTab !== TEAMS_TAB" class="flex items-center gap-2">
            <span>{{ t('team') }}:</span>
            <Select :model-value="selectedTeam" @update:model-value="updateSelectedTeam" :options="teams" option-label="name" option-value="id" class="w-40" />
          </div>
        </div>
        <TabPanels class="flex-1 overflow-y-auto !px-0 !pt-0 !pb-1">
          <TabPanel v-for="tab in tabs" :key="tab.value" :value="tab.value" class="h-full">
            <component :is="tab.component" />
          </TabPanel>
        </TabPanels>
      </Tabs>
    </div>
  </FlexCard>
</template>

<i18n lang="json">
{
  "en": {
    "dashboardTitle": "AI Console",
    "team": "Team",
    "me": "Me",
    "manageTeams": "Manage teams",
    "impactLabel": "Impact",
    "usersLabel": "Users",
    "teamsLabel": "Teams",
    "usageLabel": "Usage"
  },
  "es": {
    "dashboardTitle": "Consola IA",
    "team": "Equipo",
    "me": "Yo",
    "manageTeams": "Administrar equipos",
    "impactLabel": "Impacto",
    "usersLabel": "Usuarios",
    "teamsLabel": "Equipos",
    "usageLabel": "Uso"
  }
}
</i18n>

<style scoped>
:deep(.p-tabs),
:deep(.p-tablist),
:deep(.p-tablist-content),
:deep(.p-tablist-viewport),
:deep(.p-tablist-tab-list) {
  overflow: visible !important;
}
:deep(.p-tablist-tab-list) {
  border: none !important;
  gap: 24px !important;
}
:deep(.p-tablist-active-bar) {
  bottom: -19px !important;
  border-radius: 4px !important;
}
</style>
