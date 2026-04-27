import { defineStore } from 'pinia';
import { ref } from 'vue';
import type { ChatMessage } from '@/types';
import { sendChat, autoEda, clearHistory } from '@/api/client';
import { useToast } from '@/composables/useToast';

export const useChatStore = defineStore('chat', () => {
  const messages = ref<ChatMessage[]>([]);
  const isLoading = ref(false);
  const error = ref<string | null>(null);
  const edaResult = ref<{ message: string; figures: string[] } | null>(null);
  const mlResult = ref<string | null>(null);

  async function sendMessage(sessionId: string, text: string) {
    messages.value.push({ role: 'user', content: text });
    isLoading.value = true;
    error.value = null;
    try {
      const { data } = await sendChat(sessionId, text);
      messages.value.push({
        role: 'assistant',
        content: data.message,
        figures: data.figures || [],
      });
    } catch (e: any) {
      messages.value.push({ role: 'assistant', content: `Erreur : ${e.message}`, figures: [] });
    } finally {
      isLoading.value = false;
    }
  }

  async function triggerAutoEda(sessionId: string) {
    isLoading.value = true;
    error.value = null;
    try {
      const { data } = await autoEda(sessionId);
      edaResult.value = { message: data.message, figures: data.figures || [] };
    } catch (e: any) {
      error.value = e.message;
    } finally {
      isLoading.value = false;
    }
  }

  async function clear(sessionId: string) {
    const { error: toastError } = useToast();
    try {
      await clearHistory(sessionId);
    } catch {
      toastError("Impossible de supprimer l'historique");
    }
    messages.value = [];
    edaResult.value = null;
    mlResult.value = null;
  }

  function reset() {
    messages.value = [];
    edaResult.value = null;
    mlResult.value = null;
    error.value = null;
    isLoading.value = false;
  }

  return { messages, isLoading, error, edaResult, mlResult, sendMessage, triggerAutoEda, clear, reset };
});
