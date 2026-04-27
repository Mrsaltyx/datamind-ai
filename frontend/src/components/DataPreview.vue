<template>
  <div>
    <div class="flex gap-2 mb-4">
      <button
        :class="activeTab === 'preview' ? 'bg-white/15 text-white' : 'bg-white/5 text-white/60 hover:text-white'"
        class="px-4 py-2 rounded-lg text-sm font-medium transition-colors cursor-pointer"
        @click="activeTab = 'preview'"
      >
        Apercu des donnees
      </button>
      <button
        :class="activeTab === 'stats' ? 'bg-white/15 text-white' : 'bg-white/5 text-white/60 hover:text-white'"
        class="px-4 py-2 rounded-lg text-sm font-medium transition-colors cursor-pointer"
        @click="loadStats"
      >
        Statistiques
      </button>
    </div>

    <!-- Preview Tab -->
    <div v-if="activeTab === 'preview'">
      <div v-if="previewData" class="overflow-x-auto rounded-xl border border-white/10">
        <table class="w-full text-sm">
          <thead>
            <tr class="border-b border-white/10">
              <th
                v-for="col in previewData.columns"
                :key="col"
                class="px-3 py-2 text-left text-xs font-semibold text-white/60 whitespace-nowrap"
              >
                {{ col }}
              </th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="(row, i) in previewData.rows"
              :key="i"
              class="border-b border-white/5 hover:bg-white/5 transition-colors"
            >
              <td
                v-for="col in previewData.columns"
                :key="col"
                class="px-3 py-1.5 text-xs text-white/70 whitespace-nowrap max-w-[200px] truncate"
              >
                {{ formatCell(row[col]) }}
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      <div v-else-if="loading" class="flex justify-center py-8">
        <div class="spinner"></div>
      </div>
    </div>

    <!-- Stats Tab -->
    <div v-if="activeTab === 'stats'">
      <div v-if="statsData" class="overflow-x-auto rounded-xl border border-white/10">
        <table class="w-full text-sm">
          <thead>
            <tr class="border-b border-white/10">
              <th
                v-for="stat in statsData.columns"
                :key="stat"
                class="px-3 py-2 text-left text-xs font-semibold text-white/60 whitespace-nowrap"
              >
                {{ stat }}
              </th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="(stat, i) in statsData.stats"
              :key="i"
              class="border-b border-white/5"
            >
              <td class="px-3 py-1.5 text-xs font-mono text-white/70">
                {{ stat }}
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      <div v-else-if="loading" class="flex justify-center py-8">
        <div class="spinner"></div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, watch } from 'vue';
import { useDataStore } from '@/stores/data';
import { getPreview, getStatistics } from '@/api/client';
import { useToast } from '@/composables/useToast';

const dataStore = useDataStore();
const activeTab = ref('preview');
const loading = ref(false);
const previewData = ref<{ columns: string[]; rows: Record<string, unknown>[] } | null>(null);
const statsData = ref<{ columns: string[]; stats: Record<string, unknown[]> } | null>(null);

const { error: toastError } = useToast();

async function loadPreview() {
  if (!dataStore.sessionId) return;
  activeTab.value = 'preview';
  loading.value = true;
  try {
    const { data } = await getPreview(dataStore.sessionId, 20);
    previewData.value = data;
  } catch {
    toastError("Erreur lors du chargement de l'apercu");
  }
  loading.value = false;
}

async function loadStats() {
  if (!dataStore.sessionId) return;
  activeTab.value = 'stats';
  loading.value = true;
  try {
    const { data } = await getStatistics(dataStore.sessionId);
    statsData.value = data;
  } catch {
    toastError("Erreur lors du chargement des statistiques");
  }
  loading.value = false;
}

function formatCell(val: unknown): string {
  if (val === null || val === undefined) return '-';
  if (typeof val === 'number') return val.toLocaleString('fr-FR');
  return String(val);
}

onMounted(() => {
  if (dataStore.sessionId) loadPreview();
});

watch(() => dataStore.sessionId, (newId) => {
  if (newId) loadPreview();
});
</script>
