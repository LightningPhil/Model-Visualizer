// The upstream package's declaration export points at a missing file and its
// legacy declarations describe the previous WebIDL API. Keep a small, typed
// boundary around the actual Embind API rather than leaking `any` into the app.
export type NativeEnum = { readonly value: number }
export interface Disposable {
  delete(): void
}
export interface NativePoint extends Disposable {
  x: number
  y: number
}
export interface NativeRoute extends Disposable {
  size(): number
  at(index: number): NativePoint
}
export interface NativeShape extends Disposable {
  readonly __shape?: true
}
export interface NativePin extends Disposable {
  setExclusive(value: boolean): void
}
export interface NativeConnector extends Disposable {
  displayRoute(): NativeRoute
  setRoutingType(type: NativeEnum): void
  hasValidRoute(): boolean
  hasCrossingObstacles(): boolean
}
export interface NativeRouter extends Disposable {
  processTransaction(): void
  deleteConnector(connector: NativeConnector): void
  setRoutingParameter(parameter: NativeEnum, value: number): void
  setRoutingOption(option: NativeEnum, value: boolean): void
}
export interface AvoidEngine {
  Router: new (flags: number) => NativeRouter
  RouterFlag: { OrthogonalRouting: NativeEnum }
  ConnType: { ConnType_Orthogonal: NativeEnum }
  RoutingParameter: Record<
    | 'segmentPenalty'
    | 'crossingPenalty'
    | 'fixedSharedPathPenalty'
    | 'portDirectionPenalty'
    | 'shapeBufferDistance'
    | 'idealNudgingDistance'
    | 'reverseDirectionPenalty',
    NativeEnum
  >
  RoutingOption: Record<
    | 'nudgeOrthogonalSegmentsConnectedToShapes'
    | 'penaliseOrthogonalSharedPathsAtConnEnds'
    | 'nudgeOrthogonalTouchingColinearSegments'
    | 'nudgeSharedPathsWithCommonEndPoint',
    NativeEnum
  >
  Point: new (x: number, y: number) => NativePoint
  Rectangle: new (a: NativePoint, b: NativePoint) => Disposable
  ShapeRef: new (router: NativeRouter, rectangle: Disposable) => NativeShape
  ShapeConnectionPin: new (
    shape: NativeShape,
    classId: number,
    x: number,
    y: number,
    proportional: boolean,
    insideOffset: number,
    directions: number,
  ) => NativePin
  ConnEnd: new (shape: NativeShape, classId: number) => Disposable
  ConnRef: new (router: NativeRouter, a: Disposable, b: Disposable) => NativeConnector
}

export interface AvoidLibrary {
  load(path?: string): Promise<void>
  getInstance(): AvoidEngine
}
