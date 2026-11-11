<script setup>
/* A plain, non-interactive image grid: rows are patterns, columns are a
 * hand-picked set of configurations (or the fixed target). Every cell is
 * always visible at once — no selector needed to see the comparison.
 */
defineProps({
  rows: { type: Array, required: true },
  columns: { type: Array, required: true },
  images: { type: Object, required: true },
});
</script>

<template>
  <div class="image-grid" :style="{ '--columns': columns.length }">
    <div class="corner" />
    <div v-for="column in columns" :key="column.key" class="column-label">{{ column.label }}</div>
    <template v-for="row in rows" :key="row">
      <div class="row-label">{{ row }}</div>
      <img v-for="column in columns" :key="column.key" :src="images[`${row}-${column.key}`]" :alt="`${row}, ${column.label}`">
    </template>
  </div>
</template>

<style scoped>
.image-grid {
  display: grid;
  grid-template-columns: auto repeat(var(--columns), minmax(90px, 140px));
  gap: 8px;
  align-items: center;
  justify-content: center;
  margin: 14px 0;
}
.corner {}
.column-label, .row-label { font-size: 11px; color: #8798ab; text-align: center; }
.row-label { text-align: right; padding-right: 4px; }
img { width: 100%; aspect-ratio: 1; image-rendering: pixelated; border: 1px solid #263140; border-radius: 4px; }
</style>
