import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { fileURLToPath } from 'node:url'

export default defineConfig({
  plugins: [react()],
  // Formatters can briefly truncate a module while rewriting it on Windows.
  // Wait for stable content before transforming/caching a hot update.
  server: { watch: { awaitWriteFinish: { stabilityThreshold: 200, pollInterval: 20 } } },
  worker: { format: 'es' },
  resolve: {
    alias: [
      {
        find: /^libavoid-wasm(?=\?|$)/,
        replacement: fileURLToPath(
          new URL('./node_modules/libavoid-js/dist/libavoid.wasm', import.meta.url),
        ),
      },
    ],
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes('/katex/')) return 'math'
          if (id.includes('/@xyflow/')) return 'flow'
        },
      },
    },
  },
})
