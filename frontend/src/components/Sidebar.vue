<template>
  <aside class="w-80 flex-shrink-0 h-screen overflow-y-auto border-r border-white/10 flex flex-col"
         style="background-color: var(--bg-primary);">
    <!-- AI Config -->
    <div class="p-4">
      <h3 class="text-sm font-semibold text-white/80 uppercase tracking-wider mb-3">Configuration IA</h3>

      <!-- Provider Toggle -->
      <div class="flex rounded-lg overflow-hidden border border-white/10 mb-3">
        <button
          @click="configStore.setProvider('embedded')"
          class="flex-1 py-2 text-[10px] font-semibold transition-colors"
          :class="configStore.provider === 'embedded' ? 'bg-[#667eea] text-white' : 'bg-white/5 text-white/50 hover:text-white/70'"
        >
          Embarque
        </button>
        <button
          @click="configStore.setProvider('ollama')"
          class="flex-1 py-2 text-[10px] font-semibold transition-colors"
          :class="configStore.provider === 'ollama' ? 'bg-[#667eea] text-white' : 'bg-white/5 text-white/50 hover:text-white/70'"
        >
          Ollama
        </button>
        <button
          @click="configStore.setProvider('remote')"
          class="flex-1 py-2 text-[10px] font-semibold transition-colors"
          :class="configStore.provider === 'remote' ? 'bg-[#667eea] text-white' : 'bg-white/5 text-white/50 hover:text-white/70'"
        >
          API
        </button>
      </div>

      <!-- LLM Status -->
      <div v-if="configStore.llmStatus" class="mb-3 rounded-lg p-2 text-xs"
           :class="configStore.llmStatus.available ? 'bg-green-500/10 border border-green-500/20' : 'bg-red-500/10 border border-red-500/20'">
        <div class="flex items-center gap-2">
          <span class="w-2 h-2 rounded-full flex-shrink-0"
                :class="configStore.llmStatus.available ? 'bg-green-400' : 'bg-red-400'"></span>
          <span class="text-white/70">{{ configStore.llmStatus.message }}</span>
        </div>
      </div>

      <!-- Embedded config info -->
      <div v-if="configStore.provider === 'embedded'" class="space-y-2">
        <div class="rounded-lg p-3 bg-white/5 border border-white/10">
          <div class="text-xs text-white/40 mb-1">Modele embarque</div>
          <div class="text-sm text-white/80 font-mono break-all">{{ configStore.model }}</div>
        </div>
        <div class="text-xs text-white/30">
          Modele GGUF embarque dans le process. Aucune connexion internet requise.
        </div>
      </div>

      <!-- Ollama config info -->
      <div v-else-if="configStore.provider === 'ollama'" class="space-y-2">
        <div class="rounded-lg p-3 bg-white/5 border border-white/10">
          <div class="text-xs text-white/40 mb-1">Modele Ollama</div>
          <div class="text-sm text-white/80 font-mono">{{ configStore.model }}</div>
        </div>
        <div class="text-xs text-white/30">
          Modele via Ollama. Assurez-vous que Ollama est lance.
        </div>
      </div>

      <!-- Remote config (only when provider = remote) -->
      <div v-else class="space-y-3">
        <div>
          <label class="block text-xs text-white/50 mb-1">Cle API</label>
          <input v-model="configStore.apiKey" type="password"
                 class="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white
                        focus:outline-none focus:border-[#667eea] transition-colors"
                 placeholder="sk-..." @change="saveConfig" />
        </div>
        <div>
          <label class="block text-xs text-white/50 mb-1">URL de base</label>
          <input v-model="configStore.baseUrl" type="text"
                 class="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white
                        focus:outline-none focus:border-[#667eea] transition-colors"
                 @change="saveConfig" />
        </div>
        <div>
          <label class="block text-xs text-white/50 mb-1">Modele</label>
          <input v-model="configStore.model" type="text"
                 class="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white
                        focus:outline-none focus:border-[#667eea] transition-colors"
                 @change="saveConfig" />
        </div>
      </div>
    </div>

    <hr class="border-white/10 mx-4" />

    <!-- Dataset -->
    <div class="p-4 flex-1">
      <h3 class="text-sm font-semibold text-white/80 uppercase tracking-wider mb-3">Jeu de donnees</h3>
      <div
        class="border-2 border-dashed border-white/20 rounded-xl p-4 text-center cursor-pointer transition-colors hover:border-[#667eea] hover:bg-white/5"
        @click="triggerUpload"
        @dragover.prevent="dragOver = true"
        @dragleave="dragOver = false"
        @drop.prevent="handleDrop"
        :class="{ 'border-[#667eea] bg-white/5': dragOver }"
      >
        <div class="text-2xl mb-1" :class="dataStore.isLoading ? 'opacity-50' : ''">+</div>
        <p class="text-sm text-white/50">
          {{ dataStore.isLoading ? 'Chargement...' : 'Glissez-deposez ou cliquez' }}
        </p>
        <input ref="fileInput" type="file" accept=".csv" class="hidden" @change="onFileSelect" />
      </div>

      <!-- Data Summary -->
      <div v-if="dataStore.summary" class="mt-4">
        <div class="rounded-xl p-4 text-center"
             style="background: var(--bg-card); border: 1px solid var(--border-card);">
          <div class="text-2xl font-bold text-white">
            {{ dataStore.summary.shape[0].toLocaleString() }}
          </div>
          <div class="text-sm text-white/60">lignes</div>
          <div class="text-xl font-bold text-white mt-1">{{ dataStore.summary.shape[1] }}</div>
          <div class="text-sm text-white/60">colonnes</div>
          <div class="text-lg font-bold text-white/80 mt-1">{{ dataStore.summary.memory_mb }} Mo</div>
        </div>

        <!-- Columns -->
        <details class="mt-3">
          <summary class="text-sm text-white/70 cursor-pointer hover:text-white transition-colors">
            Colonnes ({{ dataStore.summary.columns.length }})
          </summary>
          <div class="mt-2 space-y-1">
            <div v-for="col in dataStore.summary.columns" :key="col" class="flex items-center justify-between text-xs py-1">
              <span class="text-white/80 truncate flex-1">{{ col }}</span>
              <span class="font-mono text-white/40">{{ dataStore.summary.dtypes[col] }}</span>
              <span
                class="inline-block px-2 py-0.5 rounded-full text-xs font-semibold"
                :style="{
                  background: badgeColor(dataStore.summary.missing_pct[col] || 0),
                  color: 'white',
                }"
              >
                {{ (dataStore.summary.missing_pct[col] || 0).toFixed(1) }}%
              </span>
            </div>
          </div>
        </details>
      </div>
    </div>

    <!-- Footer -->
    <div class="p-4 text-center" style="opacity: 0.4; font-size: 0.75rem;">
      DataMind AI v2.1<br>IA locale Gemma 4
    </div>
  </aside>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue';
import { useDataStore } from '@/stores/data';
import { useConfigStore } from '@/stores/config';
import { useChatStore } from '@/stores/chat';
import { useToast } from '@/composables/useToast';

const dataStore = useDataStore();
const configStore = useConfigStore();
const chatStore = useChatStore();
const { error: toastError } = useToast();
const fileInput = ref<HTMLInputElement | null>(null);
const dragOver = ref(false);

onMounted(() => {
  configStore.fetchLlmStatus();
});

function saveConfig() {
  configStore.save();
  configStore.pushToBackend().catch(() => { /* best-effort sync */ });
}

function triggerUpload() {
  fileInput.value?.click();
}

function onFileSelect(event: Event) {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  if (file) handleFile(file);
  input.value = '';
}

function handleDrop(event: DragEvent) {
  dragOver.value = false;
  const file = event.dataTransfer?.files?.[0];
  if (file && file.name.endsWith('.csv')) handleFile(file);
}

async function handleFile(file: File) {
  try {
    await dataStore.uploadFile(file);
    chatStore.reset();
  } catch {
    toastError('Erreur lors du telechargement du fichier');
  }
}

function badgeColor(pct: number): string {
  if (pct === 0) return 'rgba(34,197,94,0.2)';
  if (pct < 5) return 'rgba(234,179,8,0.2)';
  if (pct < 20) return 'rgba(249,115,22,0.2)';
  return 'rgba(239,68,68,0.2)';
}
</script>
