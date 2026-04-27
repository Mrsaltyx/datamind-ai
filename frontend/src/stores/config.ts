import { defineStore } from 'pinia';
import { ref } from 'vue';
import { updateConfig as apiUpdateConfig, getLlmStatus } from '@/api/client';
import type { LlmStatus } from '@/types';

export const useConfigStore = defineStore('config', () => {
  const provider = ref<'embedded' | 'ollama' | 'remote'>(
    (localStorage.getItem('dm_provider') as 'embedded' | 'ollama' | 'remote') || 'ollama'
  );
  const apiKey = ref(localStorage.getItem('dm_api_key') || '');
  const baseUrl = ref(localStorage.getItem('dm_base_url') || 'https://api.z.ai/api/coding/paas/v4/');
  const model = ref(localStorage.getItem('dm_model') || 'gemma4:e4b');
  const llmStatus = ref<LlmStatus | null>(null);
  const statusLoading = ref(false);

  function save() {
    localStorage.setItem('dm_provider', provider.value);
    localStorage.setItem('dm_api_key', apiKey.value);
    localStorage.setItem('dm_base_url', baseUrl.value);
    localStorage.setItem('dm_model', model.value);
  }

  function load() {
    provider.value = (localStorage.getItem('dm_provider') as 'embedded' | 'ollama' | 'remote') || 'ollama';
    apiKey.value = localStorage.getItem('dm_api_key') || '';
    baseUrl.value = localStorage.getItem('dm_base_url') || 'https://api.z.ai/api/coding/paas/v4/';
    model.value = localStorage.getItem('dm_model') || 'gemma4:e4b';
  }

  async function pushToBackend() {
    await apiUpdateConfig(apiKey.value, baseUrl.value, model.value, provider.value);
  }

  async function fetchLlmStatus() {
    statusLoading.value = true;
    try {
      const { data } = await getLlmStatus();
      llmStatus.value = data;
    } catch {
      llmStatus.value = null;
    } finally {
      statusLoading.value = false;
    }
  }

  function setProvider(newProvider: 'embedded' | 'ollama' | 'remote') {
    provider.value = newProvider;
    if (newProvider === 'embedded') {
      model.value = 'gemma-4-4b-it-Q4_K_M.gguf';
    } else if (newProvider === 'ollama') {
      model.value = 'gemma4:e4b';
    } else {
      model.value = localStorage.getItem('dm_model') || 'glm-5.1';
    }
    save();
    pushToBackend().then(() => fetchLlmStatus());
  }

  return {
    provider,
    apiKey,
    baseUrl,
    model,
    llmStatus,
    statusLoading,
    save,
    load,
    pushToBackend,
    fetchLlmStatus,
    setProvider,
  };
});
