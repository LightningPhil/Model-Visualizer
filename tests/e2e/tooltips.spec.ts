import { expect, test, type Page } from '@playwright/test'

const KEY = 'model-visualizer.help-preferences.v1'
async function builder(page: Page) {
  await page.goto('/')
  await page.getByRole('button', { name: 'Edit model configuration' }).click()
  return page.getByRole('dialog', { name: 'Shape the teaching model.' })
}

test('700 ms belongs to one continuous hover, resets between controls and cancels on exit', async ({
  page,
}) => {
  await page.clock.install({ time: new Date('2026-10-21T12:00:00Z') })
  const dialog = await builder(page)
  await page.clock.pauseAt('2026-10-21T13:00:00Z')
  const width = dialog.getByLabel('Model width D', { exact: true })
  const heads = dialog.getByLabel('Query heads', { exact: true })
  const tip = page.getByRole('tooltip')
  await width.hover()
  await page.clock.runFor(650)
  await expect(tip).toHaveCount(0)
  await heads.hover()
  await page.clock.runFor(100)
  await expect(tip).toHaveCount(0)
  await page.mouse.move(5, 5)
  await page.clock.runFor(1000)
  await expect(tip).toHaveCount(0)
  await width.hover()
  await page.clock.runFor(400)
  const box = (await width.boundingBox())!
  await page.mouse.move(box.x + 20, box.y + 12)
  await page.clock.runFor(299)
  await expect(tip).toHaveCount(0)
  await page.clock.runFor(1)
  await expect(tip).toContainText('Model width · D')
  await expect(tip.locator('button,a,input,select,textarea,[tabindex]')).toHaveCount(0)
  await expect(width).toHaveAttribute('aria-describedby', (await tip.getAttribute('id')) as string)
  await page.keyboard.press('Escape')
  await expect(tip).toHaveCount(0)
  await expect(dialog).toBeVisible()
  await expect(width).not.toHaveAttribute('aria-describedby')
  await page.clock.runFor(2000)
  await expect(tip).toHaveCount(0)
})

test('timing is configurable inside a dialog, persists independently, and can be disabled', async ({
  page,
}) => {
  const dialog = await builder(page)
  await dialog.getByRole('button', { name: 'Tooltip settings' }).click()
  const settings = page.getByRole('dialog', { name: 'Tooltip timing & help.' })
  await expect(settings.getByLabel('Tooltip delay', { exact: true })).toHaveValue('0.7')
  await settings.getByLabel('Tooltip delay', { exact: true }).fill('1.2')
  await expect
    .poll(() => page.evaluate((key) => JSON.parse(localStorage.getItem(key)!).delayMs, KEY))
    .toBe(1200)
  await page.keyboard.press('Escape')
  await expect(settings).toHaveCount(0)
  await expect(dialog).toBeVisible()
  await dialog.getByRole('button', { name: 'Tooltip settings' }).click()
  await settings.getByLabel('Tooltip delay', { exact: true }).fill('0')
  await settings.getByRole('button', { name: 'Done', exact: true }).click()
  const control = dialog.getByLabel('Attention architecture', { exact: true })
  await control.hover()
  const tip = page.getByRole('tooltip')
  await expect(tip).toContainText('Equal-sized query groups')
  await expect(tip.locator('dt')).toHaveText(['MHA', 'GQA', 'MQA'])
  await page.screenshot({ path: 'docs/images/detailed-tooltips.png' })
  await page.keyboard.press('Escape')
  await dialog.getByRole('button', { name: 'Tooltip settings' }).click()
  await settings.getByLabel('Show explanatory tooltips').uncheck()
  await settings.getByRole('button', { name: 'Done', exact: true }).click()
  await control.hover()
  await expect(tip).toHaveCount(0)
  await dialog.getByRole('button', { name: 'Back to the model' }).click()
  await page.reload()
  await page.getByRole('button', { name: 'Tooltip settings' }).click()
  await expect(settings.getByLabel('Tooltip delay', { exact: true })).toHaveValue('0')
  await expect(settings.getByLabel('Show explanatory tooltips')).not.toBeChecked()
  await settings.getByRole('button', { name: 'Restore tooltip defaults' }).click()
  await expect(settings.getByLabel('Tooltip delay', { exact: true })).toHaveValue('0.7')
  expect(
    await page.evaluate(
      (key) => JSON.parse(localStorage.getItem('model-visualizer.workspace.v1')!).extensions?.[key],
      KEY,
    ),
  ).toBeUndefined()
})

test('all builder choices have detailed help and hover cards remain readable in light and narrow views', async ({
  page,
}) => {
  await page.addInitScript(
    ({ key }) =>
      localStorage.setItem(key, JSON.stringify({ version: 1, delayMs: 0, enabled: true })),
    { key: KEY },
  )
  const dialog = await builder(page)
  for (const name of [
    'Model width D',
    'Decoder layers',
    'Query heads',
    'Features per head',
    'Attention architecture',
    'KV heads',
    'Feed-forward width',
    'Context capacity',
    'Cache scalar storage',
    'KV cache retention',
  ]) {
    await dialog.getByLabel(name, { exact: true }).hover()
    await expect(page.getByRole('tooltip')).toBeVisible()
    expect((await page.getByRole('tooltip').innerText()).length).toBeGreaterThan(180)
    await page.keyboard.press('Escape')
  }
  await dialog.getByLabel('KV cache retention').selectOption('sparse')
  for (const name of ['KV window size', 'Sparse anchor stride']) {
    await dialog.getByLabel(name, { exact: true }).hover()
    await expect(page.getByRole('tooltip')).toBeVisible()
    await page.keyboard.press('Escape')
  }
  await dialog.getByRole('button', { name: 'Back to the model' }).click()
  await page.getByRole('button', { name: 'Switch to light mode' }).click()
  await page.setViewportSize({ width: 390, height: 844 })
  await page.getByRole('button', { name: 'Open navigation' }).click()
  await page.getByRole('button', { name: 'Edit model configuration' }).click()
  await dialog.getByLabel('Model width D', { exact: true }).hover()
  const tip = page.getByRole('tooltip')
  await expect(tip).toBeVisible()
  const bounds = (await tip.boundingBox())!
  expect(bounds.x).toBeGreaterThanOrEqual(12)
  expect(bounds.x + bounds.width).toBeLessThanOrEqual(378)
  expect(bounds.y).toBeGreaterThanOrEqual(12)
  expect(bounds.y + bounds.height).toBeLessThanOrEqual(832)
  await tip.hover()
  await expect(tip).toBeVisible()
  await page.screenshot({ path: 'docs/images/detailed-tooltips-mobile.png' })
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(390)
})

test('keyboard focus retains focus and Escape dismisses help before the laboratory dialog', async ({
  page,
}) => {
  await page.addInitScript(
    ({ key }) =>
      localStorage.setItem(key, JSON.stringify({ version: 1, delayMs: 0, enabled: true })),
    { key: KEY },
  )
  await page.goto('/')
  await page.getByRole('button', { name: /Tensor laboratory Run/ }).click()
  const lab = page.getByRole('dialog', { name: 'The mathematics, alive.' })
  await expect(lab.getByRole('button', { name: 'Play', exact: true })).toBeEnabled({
    timeout: 60000,
  })
  await lab.getByRole('button', { name: 'Open numerical run setup' }).click()
  await page.keyboard.press('Tab')
  const control = lab.getByLabel('Numerical head sharing')
  await control.focus()
  await expect(page.getByRole('tooltip')).toContainText('Numerical head sharing')
  await expect(control).toBeFocused()
  await page.keyboard.press('Escape')
  await expect(page.getByRole('tooltip')).toHaveCount(0)
  await expect(lab).toBeVisible()
  await lab.getByRole('button', { name: 'Tooltip settings' }).click()
  await expect(page.getByRole('dialog', { name: 'Tooltip timing & help.' })).toBeVisible()
  await page.keyboard.press('Escape')
  await expect(lab).toBeVisible()
  await expect(page.getByRole('dialog', { name: 'Tooltip timing & help.' })).toHaveCount(0)
  await expect(lab.locator('[title]')).toHaveCount(0)
})
