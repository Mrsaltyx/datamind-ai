import { defineStore } from 'pinia';
import { ref, computed } from 'vue';
import type { DataSummary } from '@/types';
import { uploadCsv, getSummary } from '@/api/client';
import { useToast } from '@/composables/useToast';

export const useDataStore = defineStore('data', () => {
  const sessionId = ref<string | null>(null);
  const summary = ref<DataSummary | null>(null);
  const isLoading = ref(false);
  const error = ref<string | null>(null);

  const hasData = computed(() => !!sessionId.value);

  async function uploadFile(file: File) {
    isLoading.value = true;
    error.value = null;
    try {
      const { data } = await uploadCsv(file);
      sessionId.value = data.session_id;
      summary.value = data.summary;
    } catch (e: any) {
      error.value = e.message;
      throw e;
    } finally {
      isLoading.value = false;
    }
  }

  async function refreshSummary() {
    if (!sessionId.value) return;
    const { error: toastError } = useToast();
    try {
      const { data } = await getSummary(sessionId.value);
      summary.value = data;
    } catch {
      toastError("Impossible de rafraichir le resume des donnees");
    }
  }

  function reset() {
    sessionId.value = null;
    summary.value = null;
    error.value = null;
  }

  return { sessionId, summary, isLoading, error, hasData, uploadFile, refreshSummary, reset };
});
