import { defineConfig } from '@playwright/test'

export default defineConfig({
  testDir: './tests/e2e',
  fullyParallel: false,
  workers: 1,
  timeout: 30000,
  use: {
    baseURL: 'http://127.0.0.1:5200',
    channel: process.env.CI ? undefined : 'chrome',
    viewport: { width: 1440, height: 960 },
    screenshot: 'only-on-failure',
    trace: 'retain-on-failure',
  },
  webServer: {
    command: 'npm run dev -- --port 5200 --strictPort',
    url: 'http://127.0.0.1:5200',
    reuseExistingServer: !process.env.CI,
  },
})
