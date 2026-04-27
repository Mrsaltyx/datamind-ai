<template>
  <div>
    <!-- Chat Messages -->
    <div ref="messagesContainer" class="space-y-3 mb-4 max-h-[500px] overflow-y-auto pr-2">
      <div
        v-for="(msg, i) in chatStore.messages"
        :key="i"
      >
        <!-- User message -->
        <div v-if="msg.role === 'user'"
             class="rounded-xl p-4"
             style="background: rgba(102,126,234,0.15); border-left: 4px solid #667eea;">
          <p class="text-sm"><strong class="text-[#667eea]">Vous :</strong> {{ msg.content }}</p>
        </div>
        <!-- Assistant message -->
        <div v-else
             class="rounded-xl p-4"
             style="background: rgba(118,75,162,0.15); border-left: 4px solid #764ba2;">
          <div class="text-sm prose prose-invert">
            <strong class="text-[#764ba2]">DataMind :</strong>
            <div class="markdown-body" v-html="renderedContent(msg.content)"></div>
          </div>
          <div v-if="msg.figures?.length" class="mt-3 space-y-3">
            <PlotlyChart v-for="(fig, j) in msg.figures" :key="j" :chart-data="fig" />
          </div>
        </div>
      </div>
    </div>

    <!-- Loading -->
    <div v-if="chatStore.isLoading" class="flex items-center gap-3 py-4">
      <div class="spinner"></div>
      <span class="text-sm text-white/60">Reflexion...</span>
    </div>

    <!-- Input Form -->
    <div class="mt-4">
      <div class="flex gap-3">
        <input
          v-model="userInput"
          type="text"
          placeholder="Posez une question sur vos donnees..."
          class="flex-1 bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-sm text-white
                 placeholder-white/30 focus:outline-none focus:border-[#667eea] transition-colors"
          @keyup.enter="sendMessage"
          :disabled="chatStore.isLoading"
        />
        <button
          @click="sendMessage"
          :disabled="chatStore.isLoading || !userInput.trim()"
          class="px-6 py-3 rounded-xl text-sm font-semibold text-white transition-colors cursor-pointer disabled:opacity-50"
          style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);"
        >
          Envoyer
        </button>
      </div>

      <!-- Quick Actions -->
      <div class="flex gap-2 mt-3">
        <button
          v-for="(prompt, i) in quickPrompts"
          :key="i"
          @click="handleQuickAction(prompt)"
          :disabled="chatStore.isLoading"
          class="flex-1 px-2 py-2 rounded-lg text-xs font-medium bg-white/5 border border-white/10 text-white/70 hover:bg-white/10 hover:text-white transition-colors cursor-pointer"
        >
          {{ prompt }}
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, nextTick, onMounted } from 'vue';
import { useChatStore } from '@/stores/chat';
import { useDataStore } from '@/stores/data';
import PlotlyChart from './PlotlyChart.vue';
import MarkdownIt from 'markdown-it';

const chatStore = useChatStore();
const dataStore = useDataStore();
const userInput = ref('');
const messagesContainer = ref<HTMLDivElement | null>(null);
const md = new MarkdownIt({ html: true, linkify: true, breaks: true, typographer: false });

const quickPrompts = [
  'Decrire le jeu de donnees',
  'Afficher la carte de correlations',
  'Detecter les valeurs aberrantes',
  'Trouver les 5 motifs les plus interessants',
  'Suggerer une pipeline ML',
];

function renderedContent(text: string): string {
  return md.render(text);
}

async function sendMessage() {
  const text = userInput.value.trim();
  if (!text || !dataStore.sessionId || chatStore.isLoading) return;
  userInput.value = '';
  await chatStore.sendMessage(dataStore.sessionId, text);
  await nextTick();
  scrollToBottom();
}

function handleQuickAction(prompt: string) {
  userInput.value = prompt;
  sendMessage();
}

function scrollToBottom() {
  if (messagesContainer.value) {
    messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight;
  }
}
</script>
