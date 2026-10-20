import { writeFileSync } from 'node:fs'
import { z } from 'zod'
import { documentSchema } from '../src/model/schema.ts'

writeFileSync(
  'examples/model.schema.json',
  `${JSON.stringify(z.toJSONSchema(documentSchema), null, 2)}\n`,
)
console.log(
  'Exported examples/model.schema.json. Semantic cross-references are checked by parseDocument().',
)
