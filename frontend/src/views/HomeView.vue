<template>
  <template v-if="dataStore.hasData">
    <DataPreview />
    <hr class="border-white/10 my-6" />
    <AutoEda />
    <MlSuggestion />
    <hr class="border-white/10 my-6" />
    <ChatView />
  </template>
  <WelcomeMessage v-else />
</template>

<script setup lang="ts">
import { useDataStore } from '@/stores/data';
import DataPreview from '@/components/DataPreview.vue';
import AutoEda from '@/components/AutoEda.vue';
import MlSuggestion from '@/components/MlSuggestion.vue';
import ChatView from '@/components/ChatView.vue';
import WelcomeMessage from '@/components/WelcomeMessage.vue';

const dataStore = useDataStore();
</script>
