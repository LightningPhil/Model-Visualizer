import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { fileURLToPath } from 'node:url'

export default defineConfig({
  plugins: [react()],
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
