<template>
  <div class="flex flex-col items-center justify-center" style="padding: 6rem 2rem;">
    <div class="text-8xl font-bold mb-4" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
      404
    </div>
    <h2 class="text-2xl font-semibold text-white mb-3">Page non trouvee</h2>
    <p class="text-white/60 mb-8 text-center max-w-md">
      La page que vous recherchez n'existe pas ou a ete deplacee.
    </p>
    <button
      @click="router.push('/')"
      class="px-6 py-3 rounded-xl text-sm font-semibold text-white transition-all hover:scale-105 cursor-pointer"
      style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);"
    >
      Retour a l'accueil
    </button>
  </div>
</template>

<script setup lang="ts">
import { useRouter } from 'vue-router';

const router = useRouter();
</script>
