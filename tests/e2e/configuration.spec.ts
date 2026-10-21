import { expect, test } from '@playwright/test'
import { parseDocument } from '../../src/model/schema'

test('teaching dimensions repartition together and only offer compatible head sharing', async ({
  page,
}) => {
  await page.goto('/')
  await page.getByRole('button', { name: 'Edit model configuration' }).click()
  const dialog = page.getByRole('dialog', { name: 'Shape the teaching model.' })
  await dialog.getByLabel('Model width D', { exact: true }).selectOption('768')
  await expect(dialog.getByLabel('Query heads', { exact: true })).toHaveValue('8')
  await expect(dialog.getByLabel('Features per head', { exact: true })).toHaveValue('96')
  await dialog.getByLabel('Query heads', { exact: true }).selectOption('6')
  await expect(dialog.getByLabel('Features per head', { exact: true })).toHaveValue('128')
  await expect(dialog.getByLabel('KV heads', { exact: true }).locator('option')).toHaveText([
    '2 KV heads · 3 queries each',
    '3 KV heads · 2 queries each',
  ])
  await dialog.getByLabel('KV heads', { exact: true }).selectOption('3')
  await dialog.getByLabel('Features per head', { exact: true }).selectOption('64')
  await expect(dialog.getByLabel('Query heads', { exact: true })).toHaveValue('12')
  await expect(dialog.locator('.head-partition')).toContainText('768 = 12 heads × 64 features')
  await dialog.getByLabel('Attention architecture').selectOption('MHA')
  await expect(dialog.getByLabel('KV heads', { exact: true })).toHaveValue('12')
  await dialog.getByLabel('Query heads', { exact: true }).selectOption('3')
  await expect(dialog.getByLabel('Features per head', { exact: true })).toHaveValue('256')
  await dialog.getByLabel('Attention architecture').selectOption('GQA')
  await expect(dialog.getByLabel('Query heads', { exact: true })).toHaveValue('4')
  await expect(dialog.getByLabel('Features per head', { exact: true })).toHaveValue('192')
  await expect(dialog.getByLabel('KV heads', { exact: true })).toHaveValue('2')
  // Odd RoPE widths and fractional partitions never appear as choices.
  const widths = await dialog
    .getByLabel('Features per head', { exact: true })
    .locator('option')
    .evaluateAll((options) => options.map((o) => Number((o as HTMLOptionElement).value)))
  expect(widths.every((d) => d % 2 === 0 && 768 % d === 0)).toBe(true)
  await page.screenshot({ path: 'docs/images/teaching-model-controls.png' })
  await dialog.getByRole('button', { name: 'Back to the model' }).click()
  await expect
    .poll(() =>
      page.evaluate(
        () => JSON.parse(localStorage.getItem('model-visualizer.workspace.v1') ?? 'null')?.config,
      ),
    )
    .toMatchObject({
      dModel: 768,
      nQueryHeads: 4,
      headDim: 192,
      nKvHeads: 2,
      attention: 'GQA',
    })
  const saved = await page.evaluate(() =>
    JSON.parse(localStorage.getItem('model-visualizer.workspace.v1')!),
  )
  expect(parseDocument(saved).config).toEqual(saved.config)
  await page.reload()
  await page.getByRole('button', { name: 'Edit model configuration' }).click()
  await expect(dialog.getByLabel('Model width D', { exact: true })).toHaveValue('768')
  await page.getByRole('button', { name: 'Close dialog', exact: true }).click()
  await page.getByRole('button', { name: 'Switch to light mode' }).click()
  await page.setViewportSize({ width: 390, height: 844 })
  await page.getByRole('button', { name: 'Open navigation', exact: true }).click()
  await page.getByRole('button', { name: 'Edit model configuration' }).click()
  await expect(dialog.locator('.head-partition')).toBeVisible()
  await dialog.getByLabel('Attention architecture').selectOption('MQA')
  await expect(dialog.getByLabel('KV heads', { exact: true })).toHaveValue('1')
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(390)
  await page.screenshot({ path: 'docs/images/teaching-model-mobile.png' })
})
