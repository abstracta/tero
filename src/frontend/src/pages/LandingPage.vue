<script lang="ts" setup>
import { useI18n } from 'vue-i18n'
import auth from '@/services/auth';
import { config } from '@/services/auth';

const { t } = useI18n();
const keycloakLogin = config.auth.url.includes('/keycloak') || config.auth.url.startsWith('http://localhost');

async function login() {
  await auth.login()
}
</script>

<template>
  <div class="relative min-h-screen w-full overflow-hidden">
    <div class="absolute inset-0 z-0">
      <img src="@/assets/images/bg.webp" alt="Background" class="w-full h-full object-cover" />
      <div class="absolute inset-0 bg-opacity-50"></div>
    </div>
    <div class="relative z-10 min-h-screen w-full flex items-center justify-center">
      <div class="w-full max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col lg:flex-row items-center justify-center lg:gap-24 md:gap-12 sm:gap-12">
        <div class="flex-shrink-0">
          <Logo height="120px" class="transform -translate-y-1/5" />
        </div>
        <div class="bg-surface rounded-2xl p-6 sm:p-8 shadow-lg w-full max-w-[450px]">
          <h1 class="mb-4">
            {{ t("welcome") }} <br/> <span class="text-primary">Tero</span>
          </h1>
          <p class="mb-6 sm:mb-8 text-md sm:text-md leading-relaxed">
            {{ t("description") }}
          </p>
          <SimpleButton size="large" shape="square" class="gap-2" @click="login">
            <img src="@/assets/images/google-icon.svg" alt="Google" class="w-4 h-4 sm:w-5 sm:h-5" v-if="!keycloakLogin" />
            <span>{{ t(keycloakLogin ? "signIn" : "googleLogin") }}</span>
          </SimpleButton>
        </div>
      </div>
    </div>
  </div>
</template>

<i18n>
{
  "en": {
    "welcome": "Welcome to",
    "description": "Discover, create and collaborate with artificial intelligence agents designed to enhance your day-to-day. Connect and start harnessing their power.",
    "googleLogin": "Continue with Google",
    "signIn": "Sign in"
  },
  "es": {
    "welcome": "Te damos la bienvenida a",
    "description": "Descubre, crea y colabora con agentes de inteligencia artificial diseñados para potenciar tu día a día. Conéctate y empieza a aprovechar su poder.",
    "googleLogin": "Continuar con Google",
    "signIn": "Iniciar sesión"
  }
}
</i18n>
