import { latestBinding, type NumericalTrace } from './engine'

/** Prefer a tensor actually shared by the two recorded operations. Multi-output
 * events can bind different tensors to each component (e.g. probabilities vs ID).
 * Destination operands are included only for the same decoder call. */
export function inspectConnection(
  trace: NumericalTrace,
  edge: { source: string; target: string },
  cursor: number,
) {
  const source = latestBinding(trace, edge.source, cursor),
    target = latestBinding(trace, edge.target, cursor),
    record = source ?? target,
    outputs = source?.bindings[edge.source] ?? source?.outputs ?? [],
    consumer = target?.call === source?.call ? target : undefined,
    used = new Set([...(consumer?.inputs ?? []), ...(consumer?.bindings[edge.target] ?? [])]),
    preferred =
      outputs.find((id) => used.has(id)) ?? outputs[0] ?? target?.bindings[edge.target]?.[0],
    available = [
      ...new Set([
        ...outputs,
        ...(consumer?.inputs ?? []),
        ...(source ? [] : (target?.outputs ?? [])),
      ]),
    ].filter((id) => trace.tensors[id].createdAt <= cursor)
  return { record, preferred, available }
}
