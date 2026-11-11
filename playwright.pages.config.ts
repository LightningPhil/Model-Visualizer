import { defineConfig } from '@playwright/test'

export default defineConfig({
  testDir: './tests/pages',
  outputDir: './test-results/pages',
  workers: 1,
  timeout: 180000,
  use: {
    baseURL: 'http://127.0.0.1:5201/Model-Visualizer/',
    channel: process.env.CI ? undefined : 'chrome',
    screenshot: 'only-on-failure',
    trace: 'retain-on-failure',
  },
  webServer: {
    command: 'npm run preview -- --base=/Model-Visualizer/ --port 5201 --strictPort',
    url: 'http://127.0.0.1:5201/Model-Visualizer/',
    reuseExistingServer: false,
  },
})
