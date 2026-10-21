import type { VisualNode } from './GraphCanvas'
import type { RoutingResult } from '../model/routing'

const emptyPorts: VisualNode['data']['ports'] = []

/** Keep unchanged nodes/data referentially stable. A drag updates one node,
 * rather than rendering every component and handle in the model. */
export function reconcileNodes(
  authored: VisualNode[],
  state: VisualNode[],
  previous: VisualNode[],
  routes: RoutingResult | null,
): VisualNode[] {
  const stateById = new Map(state.map((node) => [node.id, node]))
  const previousById = new Map(previous.map((node) => [node.id, node]))
  return authored.map((node) => {
    const current = stateById.get(node.id)
    const old = previousById.get(node.id)
    // Keep handles mounted while a new route is computed. Stale paths remain
    // withheld, but removing/recreating every handle during a drag is wasteful.
    let ports = routes?.ports[node.id] ?? old?.data.ports ?? emptyPorts
    if (
      old &&
      ports.length === old.data.ports.length &&
      ports.every((port, i) => {
        const before = old.data.ports[i]
        return (
          port.id === before.id &&
          port.side === before.side &&
          port.offset === before.offset &&
          port.type === before.type &&
          port.point.x === before.point.x &&
          port.point.y === before.point.y &&
          port.nodeId === before.nodeId &&
          port.edgeId === before.edgeId &&
          port.classId === before.classId
        )
      })
    )
      ports = old.data.ports
    let data = { ...node.data, ports }
    if (
      old &&
      (Object.keys(data) as (keyof typeof data)[]).every((key) => data[key] === old.data[key])
    )
      data = old.data
    const next = { ...current, ...node, measured: current?.measured ?? node.measured, data }
    return old && (Object.keys(next) as (keyof VisualNode)[]).every((key) => next[key] === old[key])
      ? old
      : next
  })
}
