import { ref } from 'vue';

export interface Toast {
  id: number;
  message: string;
  type: 'error' | 'success' | 'info';
}

const toasts = ref<Toast[]>([]);
let nextId = 0;

export function useToast() {
  function show(message: string, type: Toast['type'] = 'error', duration = 4000) {
    const id = nextId++;
    toasts.value.push({ id, message, type });
    setTimeout(() => {
      toasts.value = toasts.value.filter(t => t.id !== id);
    }, duration);
  }

  function error(message: string) {
    show(message, 'error');
  }

  function success(message: string) {
    show(message, 'success');
  }

  function info(message: string) {
    show(message, 'info');
  }

  function dismiss(id: number) {
    toasts.value = toasts.value.filter(t => t.id !== id);
  }

  return { toasts, show, error, success, info, dismiss };
}
