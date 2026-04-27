<template>
  <div>
    <div class="flex justify-between items-center mb-4">
      <h3 class="text-lg font-semibold text-white">Analyse exploratoire automatique des donnees</h3>
      <button
        @click="runEda"
        :disabled="chatStore.isLoading"
        class="px-5 py-2.5 rounded-xl text-sm font-semibold text-white transition-colors cursor-pointer disabled:opacity-50"
        style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);"
      >
        Lancer l'EDA
      </button>
    </div>

    <div v-if="chatStore.isLoading && !chatStore.edaResult" class="flex items-center gap-3 py-8">
      <div class="spinner"></div>
      <span class="text-white/60">L'agent analyse vos donnees... Cela peut prendre 30 a 60 secondes.</span>
    </div>

    <div v-if="chatStore.edaResult">
      <hr class="border-white/10 mb-4" />
      <div class="markdown-body mb-4" v-html="renderedContent(chatStore.edaResult.message)"></div>
      <div v-if="chatStore.edaResult.figures?.length" class="grid grid-cols-1 gap-4">
        <PlotlyChart
          v-for="(fig, i) in chatStore.edaResult.figures"
          :key="i"
          :chart-data="fig"
        />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { useChatStore } from '@/stores/chat';
import { useDataStore } from '@/stores/data';
import PlotlyChart from './PlotlyChart.vue';
import MarkdownIt from 'markdown-it';

const chatStore = useChatStore();
const dataStore = useDataStore();
const md = new MarkdownIt({ html: true, linkify: true, breaks: true, typographer: false });

function renderedContent(text: string): string {
  return md.render(text);
}

async function runEda() {
  if (!dataStore.sessionId || chatStore.isLoading) return;
  await chatStore.triggerAutoEda(dataStore.sessionId);
}
</script>
