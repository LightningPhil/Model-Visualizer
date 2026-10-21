import type { VisualEdge, VisualNode } from '../GraphCanvas'
import { instanceOf } from '../../model/whole-model'
import { NODE_HEIGHT, NODE_WIDTH, roundedPath, type Rect } from '../../model/geometry'

export const OVERVIEW_ZOOM = 0.18
export const OVERVIEW_MIN_NODES = 80
export type OverviewScene = ReturnType<typeof overviewScene>
export type OverviewView = {
  x: number
  y: number
  zoom: number
  width: number
  height: number
  dpr: number
}
export type OverviewPalette = Record<
  | 'panel'
  | 'border'
  | 'text'
  | 'muted'
  | 'canvas'
  | 'cyan'
  | 'compute'
  | 'violet'
  | 'amber'
  | 'edge',
  string
>

/** Drawing-only projection of the same validated routes and component boxes.
 * No model content, hit-test geometry or persisted layout is replaced. */
export function overviewScene(nodes: VisualNode[], edges: VisualEdge[], stages: [string, Rect][]) {
  return {
    nodes: nodes.map((node) => ({
      id: node.id,
      ...node.position,
      width: NODE_WIDTH,
      height: NODE_HEIGHT,
      category: node.data.component.category,
      title: node.data.component.title,
      selected: node.selected,
      execution: node.data.execution,
      optional: instanceOf(node.data.component)?.optional,
    })),
    edges: edges.map((edge) => ({
      id: edge.id,
      kind: edge.data!.connection.kind,
      path: roundedPath(edge.data!.route.points, 5),
      leader: edge.data!.route.labelLeader
        ? roundedPath(edge.data!.route.labelLeader, 0)
        : undefined,
      points: edge.data!.route.points,
      label: edge.data!.route.label,
      selected: edge.selected,
      active: edge.data!.active,
      playing: edge.data!.playing,
    })),
    stages,
  }
}

/** Hold an image only when its existing paths could obstruct changed geometry.
 * Selection, playback styling and successive route-free drag frames can update
 * in place without exposing the full SVG on every pointer movement. */
export function staleOverviewRoutes(before: OverviewScene | undefined, next: OverviewScene) {
  if (!before?.edges.length) return false
  if (before.nodes.length !== next.nodes.length || before.edges.length !== next.edges.length)
    return true
  return (
    before.nodes.some((node, i) => {
      const other = next.nodes[i]
      return node.id !== other.id || node.x !== other.x || node.y !== other.y
    }) ||
    before.edges.some((edge, i) => {
      const other = next.edges[i]
      return (
        edge.id !== other.id ||
        edge.path !== other.path ||
        edge.leader !== other.leader ||
        edge.label.x !== other.label.x ||
        edge.label.y !== other.label.y ||
        edge.label.width !== other.label.width ||
        edge.label.height !== other.label.height
      )
    })
  )
}

export type OverviewRequest =
  | { type: 'init'; canvas: OffscreenCanvas }
  | { type: 'scene'; scene: OverviewScene; palette: OverviewPalette; revision: number }
  | { type: 'view'; view: OverviewView }
