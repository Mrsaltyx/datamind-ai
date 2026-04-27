<template>
  <div class="flex h-screen overflow-hidden">
    <Sidebar />
    <main class="flex-1 overflow-y-auto p-6">
      <AppHeader />
      <router-view />
    </main>
    <ToastContainer />
  </div>
</template>

<script setup lang="ts">
import AppHeader from '@/components/AppHeader.vue';
import Sidebar from '@/components/Sidebar.vue';
import ToastContainer from '@/components/ToastContainer.vue';
</script>
