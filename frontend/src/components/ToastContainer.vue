<template>
  <div class="fixed top-4 right-4 z-50 flex flex-col gap-2 max-w-sm">
    <TransitionGroup name="toast">
      <div
        v-for="toast in toasts"
        :key="toast.id"
        class="flex items-start gap-3 px-4 py-3 rounded-xl shadow-lg border backdrop-blur-sm"
        :class="bgClass(toast.type)"
      >
        <span class="text-sm flex-1" :class="textClass(toast.type)">{{ toast.message }}</span>
        <button
          @click="dismiss(toast.id)"
          class="text-current opacity-50 hover:opacity-100 transition-opacity cursor-pointer text-sm leading-none"
        >
          &times;
        </button>
      </div>
    </TransitionGroup>
  </div>
</template>

<script setup lang="ts">
import { useToast } from '@/composables/useToast';

const { toasts, dismiss } = useToast();

function bgClass(type: string): string {
  switch (type) {
    case 'success': return 'bg-green-900/80 border-green-700/50';
    case 'info': return 'bg-blue-900/80 border-blue-700/50';
    default: return 'bg-red-900/80 border-red-700/50';
  }
}

function textClass(type: string): string {
  switch (type) {
    case 'success': return 'text-green-200';
    case 'info': return 'text-blue-200';
    default: return 'text-red-200';
  }
}
</script>

<style scoped>
.toast-enter-active {
  transition: all 0.3s ease-out;
}
.toast-leave-active {
  transition: all 0.2s ease-in;
}
.toast-enter-from {
  opacity: 0;
  transform: translateX(100%);
}
.toast-leave-to {
  opacity: 0;
  transform: translateX(100%);
}
</style>
