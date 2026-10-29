import type { WikiSection } from './types'

export type TopicPerspectives = { introduction: WikiSection[]; technical: WikiSection[] }
export type ReadingMode = 'introduction' | 'technical' | 'explore'

/** Authored alternative explanations, not an automatic rewrite of a model or
 * an assumption that an unfamiliar imported component uses these semantics. */
export function introduction(
  purpose: string,
  picture: string,
  steps: string[],
  boundary: string,
): WikiSection[] {
  return [
    { id: 'intro-purpose', title: 'What it is and why it exists', paragraphs: [purpose] },
    { id: 'intro-picture', title: 'Another way to picture it', paragraphs: [picture] },
    { id: 'intro-sequence', title: 'Follow what happens', paragraphs: [], bullets: steps },
    {
      id: 'intro-boundary',
      title: 'What this picture does—and does not—mean',
      paragraphs: [boundary],
    },
  ]
}
export const detail = (
  id: string,
  title: string,
  paragraphs: string[],
  equation?: string,
): WikiSection => ({ id, title, paragraphs, equation })
