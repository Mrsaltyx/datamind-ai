import axios from 'axios';

const api = axios.create({
  baseURL: '/api',
  timeout: 180000, // 3 min for local model responses
});

api.interceptors.response.use(
  (response) => response,
  (error) => {
    const msg = error.response?.data?.detail || error.message || 'Erreur inconnue';
    return Promise.reject(new Error(msg));
  }
);

export const uploadCsv = (file: File) => {
  const formData = new FormData();
  formData.append('file', file);
  return api.post('/data/upload', formData);
};

export const getSummary = (sessionId: string) =>
  api.get(`/data/${sessionId}/summary`);

export const getPreview = (sessionId: string, rows = 20) =>
  api.get(`/data/${sessionId}/preview`, { params: { rows } });

export const getStatistics = (sessionId: string) =>
  api.get(`/data/${sessionId}/statistics`);

export const executeTool = (sessionId: string, toolName: string, args: Record<string, unknown> = {}) =>
  api.post(`/tools/${sessionId}/execute`, { tool_name: toolName, arguments: args });

export const sendChat = (sessionId: string, message: string) =>
  api.post(`/chat/${sessionId}/send`, { message });

export const autoEda = (sessionId: string) =>
  api.post(`/chat/${sessionId}/auto-eda`);

export const clearHistory = (sessionId: string) =>
  api.delete(`/chat/${sessionId}/history`);

export const suggestMl = (sessionId: string) =>
  api.post(`/ml/${sessionId}/suggest`);

export const detectTarget = (sessionId: string) =>
  api.post(`/ml/${sessionId}/detect-target`);

export const updateConfig = (apiKey: string, baseUrl: string, model: string, llmProvider: string = '') =>
  api.post('/config/update', { api_key: apiKey, base_url: baseUrl, model, llm_provider: llmProvider });

export const getLlmStatus = () =>
  api.get('/config/llm-status');

export default api;
