import { expect, test } from '@playwright/test'

test('project-path deployment loads graph, WASM, fonts, licenses and trained inference', async ({
  page,
  request,
}) => {
  const errors: string[] = []
  const failures: string[] = []
  const assets: string[] = []
  page.on('pageerror', (error) => {
    errors.push(error.message)
    console.error(error.message)
  })
  page.on('response', (response) => {
    if (response.status() >= 400) failures.push(`${response.status()} ${response.url()}`)
    assets.push(response.url())
  })
  await page.goto('./')
  await expect(page.locator('.react-flow__node').first()).toBeVisible({ timeout: 30000 })
  await expect(page.locator('.routed-connection').first()).toBeAttached({ timeout: 30000 })
  await expect(page.locator('.routing-message')).toHaveCount(0)
  await page.getByRole('button', { name: /Local trained model/ }).click()
  await page.getByLabel('Local model output budget').selectOption('1')
  await page.getByRole('button', { name: /Load trained model/ }).click()
  await expect(page.getByRole('button', { name: 'Generate & inspect' })).toBeEnabled({
    timeout: 30000,
  })
  await page.getByRole('button', { name: 'Generate & inspect' }).click()
  const lab = page.getByRole('dialog', { name: 'The mathematics, alive.' })
  await expect(lab.getByRole('button', { name: 'Play', exact: true })).toBeEnabled({
    timeout: 120000,
  })
  await expect(lab.locator('.numerical-badge')).toContainText('LOCAL · TRAINED')
  expect(assets.some((url) => url.endsWith('.wasm'))).toBe(true)
  expect(assets.some((url) => url.includes('.woff'))).toBe(true)
  expect(assets.filter((url) => url.includes('/models/stories260k/'))).toHaveLength(2)
  for (const url of assets.filter((url) => url.startsWith('http:'))) {
    expect(url).toContain('/Model-Visualizer/')
  }
  for (const path of [
    'THIRD_PARTY_NOTICES.txt',
    'licenses/README.txt',
    'licenses/libavoid-LGPL-2.1.txt',
    'licenses/sources/libavoid-js-0.5.0-beta.5.tar.gz',
    'licenses/sources/adaptagrams-v1.0.5.tar.gz',
    'models/stories260k/LICENSE',
  ]) {
    const response = await request.get(path)
    expect(response.status(), path).toBe(200)
    expect(response.headers()['content-type'], path).not.toContain('text/html')
  }
  expect(failures).toEqual([])
  expect(errors).toEqual([])
})
