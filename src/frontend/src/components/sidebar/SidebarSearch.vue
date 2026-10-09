<script lang="ts" setup>
import { ref, watch, nextTick, onMounted } from 'vue';
import { useI18n } from 'vue-i18n';
import { ApiService, type Thread, type Agent } from '@/services/api';
import { useErrorHandler } from '@/composables/useErrorHandler';
import { useDebounce } from '@/composables/useDebounce';

const { t } = useI18n();
const api = new ApiService();
const { handleError } = useErrorHandler();

const isSearching = ref(false);
const defaultPageLimit = 30;
const searchInput = ref<HTMLInputElement | null>(null);

const emit = defineEmits<{
  (e: 'update:showingSearchInput', value: boolean): void
  (e: 'update:searchQuery', value: string): void
  (e: 'searchResults', results: { agents: Agent[], chats: Thread[] }): void
}>();

const props = defineProps<{
  showingSearchInput: boolean,
  searchQuery: string
}>();

const focusSearch = () => {
  nextTick(() => {
    searchInput.value?.focus();
  });
}

onMounted(() => {
  if (props.showingSearchInput) {
    focusSearch();
  }
});

watch(() => props.showingSearchInput, (showing) => {
  if (showing) {
    focusSearch();
  }
});

const emitUpdateSearchQuery = (value: string | number | undefined) => {
  emit('update:searchQuery', value as string);
};

const emitSearchResults = (results: { agents: Agent[], chats: Thread[] }) => {
  emit('searchResults', results);
};

const performSearch = async (query: string) => {
  if (!query || !isSearching.value) {
    isSearching.value = false;
    return;
  }

  try {
    isSearching.value = true;
    const [agents, chats] = await Promise.all([
      api.findUserAgents(query.trim()),
      api.findChatsByText(query.trim(), defaultPageLimit, true)
    ]);

    emitSearchResults({ agents, chats });

  } catch (error) {
    handleError(error);
    emitSearchResults({ agents: [], chats: [] });
  } finally {
    isSearching.value = false;
  }
};

const debouncedSearch = useDebounce(() => {
  performSearch(props.searchQuery);
});

watch(() => props.searchQuery, () => {
  isSearching.value = true;
  clearSearchResults();
  debouncedSearch();
});

const emitUpdateShowingSearchInput = (value: boolean) => {
  emit('update:showingSearchInput', value);
};

const clearSearchQuery = () => {
  emitUpdateSearchQuery('');
};

const clearSearchResults = () => {
  emitSearchResults({ agents: [], chats: [] });
};

const clearSearch = () => {
  clearSearchQuery();
  clearSearchResults();
  emitUpdateShowingSearchInput(false);
};

defineExpose({
  clearSearch: () => clearSearch(),
  isSearching
});
</script>

<template>
    <div class="w-full px-1 my-2">
      <div class="relative">
        <InteractiveInput
          ref="searchInput"
          :model-value="props.searchQuery"
          @update:model-value="emitUpdateSearchQuery"
          @keydown.esc="emitUpdateShowingSearchInput(false)"
          :placeholder="t('search')"
          :loading="isSearching"
          start-icon="IconSearch"
          :end-icon="'IconX'"
          @end-icon-click="clearSearch"
          autofocus
          fluid />
      </div>
    </div>
</template>

<i18n>
{
  "en": {
    "search": "Type to search..."
  },
  "es": {
    "search": "Escribe para buscar..."
  }
}
</i18n>
