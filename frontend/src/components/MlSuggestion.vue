<template>
  <div>
    <div class="flex justify-between items-center mb-4">
      <h3 class="text-lg font-semibold text-white">Suggestion de pipeline Machine Learning</h3>
      <button
        @click="runMl"
        :disabled="mlLoading"
        class="px-5 py-2.5 rounded-xl text-sm font-semibold text-white transition-colors cursor-pointer disabled:opacity-50"
        style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);"
      >
        Generer le rapport ML
      </button>
    </div>

    <div v-if="mlLoading" class="flex items-center gap-3 py-8">
      <div class="spinner"></div>
      <span class="text-white/60">Analyse du dataset et generation des recommandations ML...</span>
    </div>

    <div v-if="chatStore.mlResult" class="markdown-body" v-html="renderedContent(chatStore.mlResult)"></div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import { useChatStore } from '@/stores/chat';
import { useDataStore } from '@/stores/data';
import { suggestMl } from '@/api/client';
import { useToast } from '@/composables/useToast';
import MarkdownIt from 'markdown-it';

const chatStore = useChatStore();
const dataStore = useDataStore();
const { error: toastError } = useToast();
const mlLoading = ref(false);
const md = new MarkdownIt({ html: true, linkify: true, breaks: true, typographer: false });

function renderedContent(text: string): string {
  return md.render(text);
}

async function runMl() {
  if (!dataStore.sessionId || mlLoading.value) return;
  mlLoading.value = true;
  try {
    const { data } = await suggestMl(dataStore.sessionId);
    chatStore.mlResult = data.text;
  } catch (e) {
    const msg = e instanceof Error ? e.message : 'inconnue';
    chatStore.mlResult = 'Erreur lors de la suggestion ML.';
    toastError(`Erreur ML: ${msg}`);
  }
  mlLoading.value = false;
}
</script>
