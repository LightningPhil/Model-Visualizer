import {
  Check,
  ChevronLeft,
  ChevronRight,
  FastForward,
  Pause,
  Play,
  RotateCcw,
  Route,
  X,
} from 'lucide-react'
import type { ModelConfig } from '../model/schema'
import type { Packet } from '../model/simulation'
import type { SimulationController } from './useSimulation'
import { formatBytes } from '../model/schema'
import { cachePolicyLabel, type CachePolicy } from '../model/attention'
import { CachePolicyControls } from './CachePolicyControls'

export function PacketGlyph({ packet }: { packet: Packet }) {
  return (
    <span
      className={`packet-glyph packet-${packet.status}`}
      title={`${packet.id} · position ${packet.position} · ${packet.status}`}
      data-packet-status={packet.status}
    >
      {packet.status === 'accepted' ? (
        <Check size={11} />
      ) : packet.status === 'rejected' || packet.status === 'discarded' ? (
        <X size={10} />
      ) : (
        <i />
      )}
      <b>{packet.id}</b>
      <small>
        {packet.status === 'prompt' ? 'in' : packet.status === 'confirmed' ? 'out' : packet.status}
      </small>
    </span>
  )
}

export function SimulationStrip({ sim }: { sim: SimulationController }) {
  const frame = sim.frame
  return (
    <section className="simulation-strip" aria-label="Graphical token stream">
      <div className="stream-label">
        <span className="eyebrow">TOKEN STREAM</span>
        <span>Abstract packets · graphical simulation</span>
      </div>
      <div className="packet-stream">
        <div className="packet-stream-group">
          <small>INPUT</small>
          {frame?.confirmed
            .filter((p) => p.status === 'prompt')
            .map((p) => (
              <PacketGlyph key={p.position} packet={p} />
            ))}
        </div>
        <span className="stream-divider">→</span>
        <div className="packet-stream-group">
          <small>CONFIRMED OUTPUT</small>
          {frame?.confirmed
            .filter((p) => p.status !== 'prompt')
            .map((p) => (
              <PacketGlyph key={p.position} packet={p} />
            ))}
          {!frame?.generated && <span className="packet-placeholder">Waiting for the decoder</span>}
        </div>
      </div>
    </section>
  )
}

export function SimulationPanel({
  sim,
  config,
  ready,
  onClose,
  onCachePolicy,
}: {
  sim: SimulationController
  config: ModelConfig
  ready: boolean
  onClose: () => void
  onCachePolicy: (policy: CachePolicy) => void
}) {
  const { frame, options } = sim
  const change = (changes: Partial<typeof options>) => sim.setOptions({ ...options, ...changes })
  return (
    <aside className="simulation-panel" aria-label="Execution simulator">
      <header>
        <span>
          <Route size={16} /> EXECUTION SIMULATOR
        </span>
        <button className="icon-button" aria-label="Close simulator" onClick={onClose}>
          <X size={16} />
        </button>
      </header>
      <div className="simulation-scroll">
        <span className="simulation-provenance">ILLUSTRATIVE · NO MODEL WEIGHTS</span>
        <h2>Watch the model flow.</h2>
        <p className="simulation-intro">
          Follow packets, heads and state. Proposal decisions are scripted.
        </p>
        <details className="simulation-setup">
          <summary>
            Run settings
            <span>
              {options.promptTokens} in · {options.newTokens} out · {options.mtp ? 'MTP' : 'AR'}
            </span>
          </summary>
          <CachePolicyControls config={config} onChange={onCachePolicy} />
          <div className="simulation-mode segmented">
            <button aria-pressed={!options.mtp} onClick={() => change({ mtp: false })}>
              Autoregressive
            </button>
            <button aria-pressed={options.mtp} onClick={() => change({ mtp: true })}>
              MTP + verification
            </button>
          </div>
          <div className="simulation-settings">
            <label>
              Input packets
              <input
                aria-label="Simulation input packets"
                type="number"
                min="1"
                max="32"
                value={options.promptTokens}
                onChange={(e) => change({ promptTokens: Number(e.target.value) })}
              />
            </label>
            <label>
              Output budget
              <input
                aria-label="Simulation output budget"
                type="number"
                min="1"
                max="16"
                value={options.newTokens}
                onChange={(e) => change({ newTokens: Number(e.target.value) })}
              />
            </label>
          </div>
          {options.mtp && (
            <label className="simulation-select">
              Verification scenario
              <select
                aria-label="Verification scenario"
                value={options.scenario}
                onChange={(e) => change({ scenario: e.target.value as typeof options.scenario })}
              >
                <option value="mixed">Mixed · accept and reject</option>
                <option value="accept">All drafts match</option>
                <option value="reject">First MTP draft mismatches</option>
              </select>
            </label>
          )}
        </details>
        {sim.error ? (
          <p role="alert">{sim.error}</p>
        ) : (
          <>
            <div className="playback-controls">
              <button className="icon-button" aria-label="Reset simulation" onClick={sim.reset}>
                <RotateCcw size={15} />
              </button>
              <button
                className="icon-button"
                aria-label="Previous simulation step"
                onClick={() => sim.step(-1)}
                disabled={sim.cursor === 0}
              >
                <ChevronLeft size={19} />
              </button>
              <button
                className="button simulation-play"
                disabled={!ready || frame?.phase === 'done'}
                onClick={() => sim.setPlaying(!sim.playing)}
              >
                {sim.playing ? <Pause size={15} /> : <Play size={15} />}
                {sim.playing ? 'Pause' : ready ? 'Play' : 'Preparing map…'}
              </button>
              <button
                className="icon-button"
                aria-label="Next simulation step"
                disabled={!ready || frame?.phase === 'done'}
                onClick={() => sim.step()}
              >
                <ChevronRight size={19} />
              </button>
              <button
                className="icon-button"
                aria-label="Run to completion"
                disabled={!ready || frame?.phase === 'done'}
                onClick={sim.finish}
              >
                <FastForward size={17} />
              </button>
            </div>
            <input
              className="simulation-timeline"
              aria-label="Simulation timeline"
              type="range"
              min="0"
              max={(sim.trace?.frames.length ?? 1) - 1}
              value={sim.cursor}
              onChange={(e) => sim.seek(Number(e.target.value))}
            />
            <div className="playback-options">
              <label>
                Detail
                <select
                  aria-label="Simulation step detail"
                  value={sim.detail}
                  onChange={(e) => sim.setDetail(e.target.value as typeof sim.detail)}
                >
                  <option value="stage">Per stage</option>
                  <option value="operation">Every operation</option>
                </select>
              </label>
              <label>
                Speed
                <select
                  aria-label="Playback speed"
                  value={sim.delay}
                  onChange={(e) => sim.setDelay(Number(e.target.value))}
                >
                  <option value="1100">Slow</option>
                  <option value="650">Normal</option>
                  <option value="250">Fast</option>
                </select>
              </label>
            </div>
            <label className="simulation-follow">
              <input
                type="checkbox"
                checked={sim.follow}
                onChange={(e) => sim.setFollow(e.target.checked)}
              />
              Follow the active operation
            </label>
            <div className="execution-card" aria-live="polite">
              <div>
                <span className="execution-phase">{frame?.phase}</span>
                <small>
                  {sim.cursor + 1} / {sim.trace?.frames.length}
                </small>
              </div>
              <h3>{frame?.title}</h3>
              <p>{frame?.description}</p>
              <code>
                B = 1 · T = {frame?.queryLength} · S ={' '}
                {Math.max(frame?.contextLength ?? 0, ...(frame?.cacheLengths ?? []))}
              </code>
            </div>
            <div className="simulation-metrics">
              <span>
                Confirmed
                <strong data-testid="generated-count">
                  {frame?.generated} / {options.newTokens}
                </strong>
              </span>
              <span>
                Round<strong>{frame?.round ?? 0}</strong>
              </span>
              <span>
                Drafts accepted<strong>{frame?.accepted ?? 0}</strong>
              </span>
              <span>
                Drafts discarded<strong>{frame?.rejected ?? 0}</strong>
              </span>
            </div>
            {options.mtp && (
              <section className="proposal-tray">
                <h3>
                  SPECULATIVE PACKETS <span>{frame?.tentative ? 'TENTATIVE' : 'PROPOSALS'}</span>
                </h3>
                <div>
                  {frame?.proposals.map((p) => (
                    <PacketGlyph key={`${frame.round}-${p.position}`} packet={p} />
                  ))}
                  {!frame?.proposals.length && (
                    <span className="packet-placeholder">
                      Proposals appear before verification.
                    </span>
                  )}
                </div>
                <p>
                  ✓ accepted · × rejected or discarded. The base next token starts each proposal
                  chain.
                </p>
              </section>
            )}
            <section className="simulation-cache">
              <h3>
                KV STATE <span>{frame?.tentative ? 'TENTATIVE WRITES' : 'REUSABLE PREFIX'}</span>
              </h3>
              <p>
                {cachePolicyLabel(config)} · two tensors per layer · [B, Hkv, R, Dh]
                <br />
                Cells identify retained absolute positions. K/V vectors are distinct for each KV
                head. Trial positions remain pinned until verification commits.
              </p>
              <div className="cache-layer-list">
                {frame?.cachePositions.map((positions, layer) => (
                  <div key={layer} className={frame.layer === layer ? 'active-cache-layer' : ''}>
                    <code>L{layer}</code>
                    <span className="cache-position-strip">
                      {positions.slice(0, 48).map((s) => (
                        <i
                          key={s}
                          data-sequence-position={s}
                          title={`Absolute position ${s}`}
                          className={s >= frame.contextLength ? 'tentative-position' : ''}
                        >
                          {s}
                        </i>
                      ))}
                    </span>
                    <b>{positions.length}</b>
                  </div>
                ))}
              </div>
              <code>
                {formatBytes(
                  2 *
                    (frame?.cachePositions.reduce((a, b) => a + b.length, 0) ?? 0) *
                    config.nKvHeads *
                    config.headDim *
                    config.dtypeBytes,
                )}{' '}
                logical K + V
              </code>
              {frame?.evictedPositions.some((p) => p.length > 0) && (
                <p className="cache-evictions" data-testid="cache-evictions">
                  Evicted absolute positions:{' '}
                  {[...new Set(frame.evictedPositions.flat())].join(', ')}. Their slots can be
                  reused.
                </p>
              )}
            </section>
          </>
        )}
        <p className="simulation-footnote">
          This run uses illustrative packets and decisions. Numerical or real-model execution can be
          added as a separate mode. No inference service is connected.
        </p>
      </div>
    </aside>
  )
}
