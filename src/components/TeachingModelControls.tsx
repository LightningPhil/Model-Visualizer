import { useMemo } from 'react'
import { headPartitions, kvHeadChoices, updateTeachingConfig } from '../model/teaching-config'
import { expandWholeModel } from '../model/whole-model'
import type { ModelConfig, ModelDocument } from '../model/schema'

export function TeachingModelControls({
  document,
  onChange,
}: {
  document: ModelDocument
  onChange: (change: Partial<ModelConfig>) => void
}) {
  const config = document.config
  const partitions = headPartitions(config.dModel, config.attention)
  const choices = useMemo(() => {
    const available = (change: Partial<ModelConfig>) => {
      try {
        const candidate = { ...document, config: updateTeachingConfig(config, change) }
        for (const graph of document.graphs) expandWholeModel(candidate, graph)
        return true
      } catch {
        return false
      }
    }
    const unique = (values: number[]) => [...new Set(values)].sort((a, b) => a - b)
    return {
      widths: unique([
        64,
        128,
        256,
        384,
        512,
        768,
        1024,
        1536,
        2048,
        3072,
        4096,
        8192,
        config.dModel,
      ]).filter((dModel) => available({ dModel })),
      layers: unique([1, 2, 4, 6, 8, 12, 24, config.nLayers]).filter((nLayers) =>
        available({ nLayers }),
      ),
      heads: headPartitions(config.dModel, config.attention).filter((p) => available(p)),
      attention: (['MHA', 'GQA', 'MQA'] as const).map((attention) => ({
        attention,
        available: available({ attention }),
      })),
      kvHeads: kvHeadChoices(config.nQueryHeads, config.attention).filter((nKvHeads) =>
        available({ nKvHeads }),
      ),
    }
  }, [document, config])
  return (
    <>
      <div className="head-partition" aria-live="polite">
        <span className="eyebrow">ONE RESIDUAL WIDTH · WHOLE HEADS</span>
        <strong>
          {config.dModel.toLocaleString()} = {config.nQueryHeads} heads × {config.headDim} features
        </strong>
        <div className="head-partition-strip" aria-hidden="true">
          {Array.from({ length: config.nQueryHeads }, (_, h) => (
            <span key={h} data-group={Math.floor(h / (config.nQueryHeads / config.nKvHeads)) % 2}>
              {h < 16 ? `Q${h}` : ''}
            </span>
          ))}
        </div>
        <p>
          Head count and head width divide D together. Changing either keeps the model width fixed.
          RoPE uses even feature pairs.
        </p>
      </div>
      <div className="configuration-form">
        <label>
          Model width D
          <select
            aria-label="Model width D"
            value={config.dModel}
            onChange={(e) => onChange({ dModel: Number(e.target.value) })}
          >
            {choices.widths.map((v) => (
              <option key={v} value={v}>
                {v.toLocaleString()}
              </option>
            ))}
          </select>
        </label>
        <label>
          Decoder layers
          <select
            aria-label="Decoder layers"
            value={config.nLayers}
            onChange={(e) => onChange({ nLayers: Number(e.target.value) })}
          >
            {choices.layers.map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
          </select>
        </label>
        <label>
          Query heads
          <select
            aria-label="Query heads"
            value={config.nQueryHeads}
            onChange={(e) => onChange({ nQueryHeads: Number(e.target.value) })}
          >
            {choices.heads.map((p) => (
              <option key={p.nQueryHeads} value={p.nQueryHeads}>
                {p.nQueryHeads} heads · {p.headDim} features each
              </option>
            ))}
          </select>
        </label>
        <label>
          Features per head
          <select
            aria-label="Features per head"
            value={config.headDim}
            onChange={(e) => onChange({ headDim: Number(e.target.value) })}
          >
            {choices.heads.toReversed().map((p) => (
              <option key={p.headDim} value={p.headDim}>
                {p.headDim} features · {p.nQueryHeads} heads
              </option>
            ))}
          </select>
        </label>
        <label>
          Attention architecture
          <select
            aria-label="Attention architecture"
            value={config.attention}
            onChange={(e) => onChange({ attention: e.target.value as ModelConfig['attention'] })}
          >
            {choices.attention.map(({ attention, available }) => (
              <option key={attention} value={attention} disabled={!available}>
                {attention}
                {!available ? ' · unavailable at this width / view size' : ''}
              </option>
            ))}
          </select>
        </label>
        <label>
          KV heads
          <select
            aria-label="KV heads"
            value={config.nKvHeads}
            onChange={(e) => onChange({ nKvHeads: Number(e.target.value) })}
          >
            {choices.kvHeads.map((v) => (
              <option key={v} value={v}>
                {v} KV {v === 1 ? 'head' : 'heads'} · {config.nQueryHeads / v}{' '}
                {config.nQueryHeads / v === 1 ? 'query' : 'queries'} each
              </option>
            ))}
          </select>
        </label>
        <p className="configuration-hint">
          {config.attention === 'MHA'
            ? 'MHA gives every query head its own K/V pair.'
            : config.attention === 'MQA'
              ? 'MQA shares one K/V pair across all query heads.'
              : 'GQA shares each K/V pair across an equal group of query heads.'}{' '}
          Only valid partitions that fit the expanded view are offered.
        </p>
        <label>
          Feed-forward width
          <select
            aria-label="Feed-forward width"
            value={config.ffnDim}
            onChange={(e) => onChange({ ffnDim: Number(e.target.value) })}
          >
            {[...new Set([768, 1536, 3072, config.ffnDim])]
              .sort((a, b) => a - b)
              .map((v) => (
                <option key={v} value={v}>
                  {v}
                </option>
              ))}
          </select>
        </label>
        <label>
          Context capacity
          <select
            aria-label="Context capacity"
            value={config.contextLength}
            onChange={(e) => onChange({ contextLength: Number(e.target.value) })}
          >
            {[...new Set([2048, 8192, 32768, 131072, config.contextLength])]
              .sort((a, b) => a - b)
              .map((v) => (
                <option key={v} value={v}>
                  {v.toLocaleString()}
                </option>
              ))}
          </select>
        </label>
        <label>
          Cache scalar storage
          <select
            aria-label="Cache scalar storage"
            value={config.dtypeBytes}
            onChange={(e) => onChange({ dtypeBytes: Number(e.target.value) as 1 | 2 | 4 })}
          >
            <option value={1}>1 byte / scalar (metadata excluded)</option>
            <option value={2}>2 bytes / scalar (BF16 / FP16)</option>
            <option value={4}>4 bytes / scalar (FP32)</option>
          </select>
        </label>
      </div>
      {partitions.length === 0 && <p role="alert">This width has no supported head partition.</p>}
    </>
  )
}
