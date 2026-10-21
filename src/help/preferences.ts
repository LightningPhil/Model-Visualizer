export const HELP_PREFERENCES_KEY = 'model-visualizer.help-preferences.v1'
export const DEFAULT_TOOLTIP_DELAY = 700
export type HelpPreferences = { delayMs: number; enabled: boolean }
export const defaultHelpPreferences: HelpPreferences = {
  delayMs: DEFAULT_TOOLTIP_DELAY,
  enabled: true,
}

export function readHelpPreferences(raw: string | null): HelpPreferences {
  try {
    const value = JSON.parse(raw ?? 'null')
    if (value?.version !== 1) return defaultHelpPreferences
    return {
      delayMs:
        typeof value.delayMs === 'number' &&
        Number.isInteger(value.delayMs) &&
        value.delayMs >= 0 &&
        value.delayMs <= 5000
          ? value.delayMs
          : DEFAULT_TOOLTIP_DELAY,
      enabled: typeof value.enabled === 'boolean' ? value.enabled : true,
    }
  } catch {
    return defaultHelpPreferences
  }
}

/** Preserve additional preference metadata; never overwrite a future version. */
export function serializeHelpPreferences(
  raw: string | null,
  preferences: HelpPreferences,
): string | undefined {
  let existing: Record<string, unknown> = {}
  try {
    const parsed: unknown = JSON.parse(raw ?? 'null')
    if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
      existing = parsed as Record<string, unknown>
      if (existing.version !== undefined && existing.version !== 1) return undefined
    }
  } catch {
    /* An invalid preference file has no recoverable settings. */
  }
  return JSON.stringify({ ...existing, version: 1, ...preferences })
}

export function positionHelp(
  anchor: { x: number; y: number; width: number; height: number },
  size: { width: number; height: number },
  viewport: { width: number; height: number },
) {
  const gap = 10,
    pad = 12
  const candidates = [
    { x: anchor.x, y: anchor.y + anchor.height + gap },
    { x: anchor.x, y: anchor.y - size.height - gap },
    { x: anchor.x + anchor.width + gap, y: anchor.y },
    { x: anchor.x - size.width - gap, y: anchor.y },
  ]
  const overflow = (p: { x: number; y: number }) =>
    Math.max(0, pad - p.x) +
    Math.max(0, p.x + size.width + pad - viewport.width) +
    Math.max(0, pad - p.y) +
    Math.max(0, p.y + size.height + pad - viewport.height)
  const best = candidates
    .map((p, i) => ({ ...p, score: overflow(p) * 100 + i }))
    .sort((a, b) => a.score - b.score)[0]
  return {
    x: Math.max(pad, Math.min(best.x, viewport.width - size.width - pad)),
    y: Math.max(pad, Math.min(best.y, viewport.height - size.height - pad)),
  }
}
