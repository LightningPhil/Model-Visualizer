import { test, expect } from '@playwright/test'

test('compares distinct attention state representations and exposes their context scaling', async ({
  page,
}) => {
  await page.goto('/')
  await page.getByRole('button', { name: 'Open model wiki', exact: true }).click()
  const wiki = page.getByRole('dialog', { name: 'The model wiki', exact: true })
  await wiki
    .getByRole('navigation', { name: 'Wiki topics' })
    .getByRole('button', { name: 'Compare attention mechanisms and memory', exact: true })
    .click()
  await expect(wiki.getByRole('tabpanel')).toContainText('Several ways', { ignoreCase: true })
  await wiki.getByRole('tab', { name: 'Interactive lab', exact: true }).click()
  const table = wiki.locator('.wiki-memory-table')
  await expect(table.locator('tbody tr')).toHaveCount(8)
  await expect(table.getByRole('row').filter({ hasText: 'Full MHA' })).toContainText('2 GiB')
  await expect(table.getByRole('row').filter({ hasText: 'Full GQA' })).toContainText('512 MiB')
  await wiki.getByLabel('Comparison context positions', { exact: true }).fill('')
  await wiki.getByLabel('Comparison context positions', { exact: true }).pressSequentially('8192')
  await expect(table.getByRole('row').filter({ hasText: 'Full MHA' })).toContainText('4 GiB')
  await wiki.getByLabel('Comparison context positions', { exact: true }).fill('8193')
  await wiki.getByLabel('Comparison context positions', { exact: true }).blur()
  await expect(wiki.getByLabel('Comparison context positions', { exact: true })).toHaveValue('8192')
  await table.getByRole('button', { name: 'Kernel linear attention', exact: true }).click()
  await expect(wiki.locator('.wiki-memory-growth .is-alternative')).toHaveCount(6)
  const fixed = await wiki.locator('.wiki-memory-growth .is-alternative').allTextContents()
  expect(new Set(fixed.map((v) => v.trim())).size).toBe(1)
  await wiki.getByLabel('Comparison execution phase').selectOption('decode')
  await expect(wiki.locator('.wiki-memory-workspace')).toContainText('8,192 entries')
  await wiki.locator('.wiki-memory-table-wrap').scrollIntoViewIfNeeded()
  await page.screenshot({ path: 'docs/images/wiki-memory-comparison.png' })
  await wiki
    .getByRole('region', { name: 'Selected attention mechanism' })
    .getByRole('button', { name: 'Read this mechanism in detail' })
    .click()
  await expect(wiki.locator('.wiki-article h3')).toHaveText('Linear attention and recurrent state')
  await wiki.getByRole('tab', { name: 'Technical detail', exact: true }).click()
  await expect(wiki.locator('.katex').first()).toBeVisible()
  await expect(wiki.locator('.katex-error')).toHaveCount(0)
})

test('the memory comparison fits narrow light layouts and keeps hypothetical settings separate', async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await page.goto('/')
  await page.getByRole('button', { name: 'Switch to light mode' }).click()
  await expect
    .poll(() => page.evaluate(() => localStorage.getItem('model-visualizer.workspace.v1')))
    .not.toBeNull()
  const before = await page.evaluate(() => localStorage.getItem('model-visualizer.workspace.v1'))
  await page.getByRole('button', { name: 'Open model wiki', exact: true }).click()
  const wiki = page.getByRole('dialog', { name: 'The model wiki', exact: true })
  await wiki.getByRole('button', { name: 'Browse wiki topics' }).click()
  await wiki
    .getByRole('navigation', { name: 'Wiki topics' })
    .getByRole('button', { name: 'Compare attention mechanisms and memory', exact: true })
    .click()
  await wiki.getByRole('tab', { name: 'Interactive lab', exact: true }).click()
  await wiki.locator('.wiki-memory-assumptions summary').click()
  await wiki.getByLabel('Comparison hybrid full layers', { exact: true }).fill('0')
  await wiki.getByLabel('Comparison layer count', { exact: true }).fill('1')
  await wiki.getByLabel('Comparison query heads', { exact: true }).selectOption('8')
  await expect(wiki.getByLabel('Comparison KV heads', { exact: true })).toHaveValue('8')
  await wiki.locator('.wiki-memory-assumptions summary').click()
  await wiki.locator('.wiki-memory-table-wrap').scrollIntoViewIfNeeded()
  expect(await wiki.evaluate((e) => e.scrollWidth <= e.clientWidth)).toBe(true)
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(390)
  await page.screenshot({ path: 'docs/images/wiki-memory-mobile.png' })
  await page.keyboard.press('Escape')
  expect(await page.evaluate(() => localStorage.getItem('model-visualizer.workspace.v1'))).toBe(
    before,
  )
})
