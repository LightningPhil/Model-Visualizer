import { expect, test } from '@playwright/test'

test('Adam and AdamW expose moment state, history and a distinct zero-gradient decay rule', async ({
  page,
}) => {
  await page.goto('/')
  await page.getByRole('button', { name: 'Open model wiki', exact: true }).click()
  const wiki = page.getByRole('dialog', { name: 'The model wiki', exact: true })
  await wiki
    .getByRole('navigation', { name: 'Wiki topics' })
    .getByRole('button', { name: 'AdamW: decoupled weight decay', exact: true })
    .click()
  await expect(wiki.getByRole('tabpanel')).toContainText('separate shrinking operation')
  await wiki.getByRole('tab', { name: 'Interactive lab', exact: true }).click()
  await wiki.getByRole('button', { name: 'Zero loss gradients', exact: true }).click()
  await wiki.getByRole('button', { name: 'Advance optimizer step', exact: true }).click()
  const plain = wiki.getByRole('region', { name: 'Plain Adam scalar state' }),
    coupled = wiki.getByRole('region', { name: 'Adam + coupled L2 scalar state' }),
    decoupled = wiki.getByRole('region', { name: 'AdamW scalar state' })
  await expect(
    plain.locator('.wiki-calculation').filter({ hasText: 'Parameter after' }),
  ).toContainText('2')
  await expect(
    coupled.locator('.wiki-calculation').filter({ hasText: 'Parameter after' }),
  ).toContainText('1.9')
  await expect(
    decoupled.locator('.wiki-calculation').filter({ hasText: 'Parameter after' }),
  ).toContainText('1.98')
  await expect(
    decoupled.locator('.wiki-calculation').filter({ hasText: 'Separate decay' }),
  ).toContainText('0.02')
  await wiki.locator('.wiki-optimizer-cards').scrollIntoViewIfNeeded()
  await page.screenshot({ path: 'docs/images/wiki-adamw-comparison.png' })
  await wiki.getByRole('button', { name: 'Sign change then zero', exact: true }).click()
  for (let i = 0; i < 4; i++)
    await wiki.getByRole('button', { name: 'Advance optimizer step', exact: true }).click()
  await expect(
    wiki.getByRole('button', { name: 'Advance optimizer step', exact: true }),
  ).toBeDisabled()
  await expect(
    plain.locator('.wiki-calculation').filter({ hasText: 'Adaptive subtraction' }),
  ).not.toHaveText('Adaptive subtraction0')
  await wiki.getByRole('tab', { name: 'Technical detail', exact: true }).click()
  await expect(wiki.locator('.katex').first()).toBeVisible()
  await wiki.getByRole('tab', { name: 'Interactive lab', exact: true }).click()
  await expect(wiki.locator('.wiki-scratch-insight')).toContainText('step 4 of 4')
  await wiki.getByRole('button', { name: 'Reset worked example', exact: true }).click()
  await expect(wiki.locator('.wiki-scratch-insight')).toContainText('step 0 of 4')
})

test('Muon shows a finite matrix transform separately from its ideal target and fits narrow light layouts', async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await page.goto('/')
  await page.getByRole('button', { name: 'Switch to light mode' }).click()
  await expect
    .poll(() =>
      page.evaluate(
        () => JSON.parse(localStorage.getItem('model-visualizer.workspace.v1') ?? '{}').ui?.theme,
      ),
    )
    .toBe('light')
  const before = await page.evaluate(() => localStorage.getItem('model-visualizer.workspace.v1'))
  await page.getByRole('button', { name: 'Open model wiki', exact: true }).click()
  const wiki = page.getByRole('dialog', { name: 'The model wiki', exact: true })
  await wiki.getByRole('button', { name: 'Browse wiki topics' }).click()
  await wiki
    .getByRole('navigation', { name: 'Wiki topics' })
    .getByRole('button', { name: 'Muon: matrix momentum and Newton–Schulz updates', exact: true })
    .click()
  await wiki.getByRole('tab', { name: 'Interactive lab', exact: true }).click()
  await expect(
    wiki.getByRole('group', { name: 'Finite polynomial output', exact: true }),
  ).toBeVisible()
  await expect(
    wiki.getByRole('group', {
      name: 'Ideal partial-polar target · separate reference',
      exact: true,
    }),
  ).toBeVisible()
  await wiki.getByLabel('Set Muon polynomial iterations', { exact: true }).fill('1')
  await expect(
    wiki
      .getByRole('list', { name: 'Muon singular-factor iteration history' })
      .getByRole('listitem'),
  ).toHaveCount(2)
  await wiki.getByLabel('Set Muon gradient factor 0', { exact: true }).fill('0')
  await wiki.getByLabel('Set Muon gradient factor 1', { exact: true }).fill('0')
  await expect(wiki.locator('.wiki-scratch-insight')).toContainText('[0, 0]')
  expect(await wiki.evaluate((e) => e.scrollWidth <= e.clientWidth)).toBe(true)
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(390)
  await wiki.getByRole('button', { name: 'Reset worked example', exact: true }).click()
  await wiki.locator('.wiki-optimizer-matrices').scrollIntoViewIfNeeded()
  await page.screenshot({ path: 'docs/images/wiki-muon-mobile.png' })
  await page.keyboard.press('Escape')
  expect(await page.evaluate(() => localStorage.getItem('model-visualizer.workspace.v1'))).toBe(
    before,
  )
})
