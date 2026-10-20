import { readFileSync, statSync } from 'node:fs'
import katex from 'katex'
import { parseDocument } from '../src/model/schema.ts'
import { expandWholeModel } from '../src/model/whole-model.ts'

const paths = process.argv.slice(2)
for (const path of paths.length ? paths : ['examples/transformer-foundations.model.json']) {
  if (statSync(path).size > 3 * 1024 * 1024)
    throw new Error(`${path}: exceeds the 3 MiB import limit`)
  const doc = parseDocument(JSON.parse(readFileSync(path, 'utf8')))
  for (const authored of doc.graphs) {
    const graph = expandWholeModel(doc, authored)
    for (const item of [...graph.nodes, ...graph.edges]) {
      if (item.equation)
        katex.renderToString(item.equation, {
          throwOnError: true,
          trust: false,
          maxExpand: 200,
          maxSize: 20,
        })
    }
  }
  console.log(
    `${path}: valid v${doc.schemaVersion}, ${doc.graphs.length} graphs, all equations rendered`,
  )
}
