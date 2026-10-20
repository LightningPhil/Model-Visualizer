# Repository instructions

- Work directly on `main`. Do not create branches, worktrees or pull requests unless the user changes this instruction.
- Use TypeScript. Keep model content independent from the React graph renderer.
- Read `docs/architecture.md` and `docs/model-format.md` before extending graph semantics.
- The authored example JSON is the source of truth. Do not introduce a second generated copy of it.
- Preserve IDs, existing user layouts, notebook content and supported extension metadata. Version and migrate breaking document changes.
- Every mathematical claim needs an accurate explanation and appropriate primary-source attribution. Distinguish architecture, execution kernel and physical storage layout.
- Clearly label synthetic examples. Never present generated numbers as trained-model activations or the teaching tokenizer as a production tokenizer.
- A KV cache is K/V state indexed by layer, batch, KV head, sequence and feature, not a nested-table metaphor. Different attention architectures can require different state representations.
- Maintain dark/light themes and red–green colourblind accessibility. Use text, shapes or patterns alongside colour.
- Verify `npm run build` and `npm test`. Run `npm run test:e2e` for graph interaction, import/export or responsive-layout changes. Browser tests use an isolated workspace.
- Use `npm run format` after edits. Regenerate `examples/model.schema.json` with `npm run schema` when the contract changes.
- Do not add network inference or send prompts to a service without a user-requested design and explicit visible controls.
