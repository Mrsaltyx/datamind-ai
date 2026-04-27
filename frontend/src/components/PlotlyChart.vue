<template>
  <div ref="chartEl" style="width: 100%; min-height: 200px;"></div>
</template>

<script setup lang="ts">
import { ref, onMounted, watch } from 'vue';
import Plotly from 'plotly.js-dist-min';
import { useToast } from '@/composables/useToast';

const props = defineProps<{ chartData: string }>();
const chartEl = ref<HTMLDivElement | null>(null);
const { error: toastError } = useToast();

function render() {
  if (!chartEl.value || !props.chartData) return;
  try {
    const spec = JSON.parse(props.chartData);
    Plotly.newPlot(chartEl.value, spec.data, {
      responsive: true,
      displayModeBar: false,
      displaylogo: false,
    });
  } catch {
    console.error('Failed to render Plotly chart');
    toastError('Erreur lors du rendu du graphique');
  }
}

onMounted(render);
watch(() => props.chartData, render);
</script>
