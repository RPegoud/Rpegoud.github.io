import { useEffect, useMemo, useRef, useState, type ChangeEvent } from 'react';
import uPlot from 'uplot';
import 'uplot/dist/uPlot.min.css';
import {
  computeMemoryModel,
  formatBytes,
  formatCount,
  PRESETS,
  DTYPE_LABELS,
  type Dtype,
  type MemoryModelInputs,
} from '../../lib/memory-model';

const DEFAULTS: MemoryModelInputs = PRESETS[0]; // Llama 3 8B

interface FieldProps {
  label: string;
  unit?: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (n: number) => void;
}

function NumberField({ label, unit, value, min, max, step = 1, onChange }: FieldProps) {
  const id = `mc-${label.replace(/\s+/g, '-').toLowerCase()}`;
  return (
    <div>
      <label htmlFor={id} className="block font-mono text-xs text-ink-muted">
        {label}
        {unit ? ` (${unit})` : ''}
      </label>
      <input
        id={id}
        type="number"
        inputMode="numeric"
        className="mt-1 w-full border border-rule bg-bg px-2 py-1.5 font-mono text-sm text-ink focus-visible:outline-2 focus-visible:outline-accent"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e: ChangeEvent<HTMLInputElement>) => {
          const n = Number(e.target.value);
          if (!Number.isNaN(n)) onChange(n);
        }}
      />
    </div>
  );
}

interface Props {
  /** 'full' — the version embedded in the article: header blurb, wider
   * layout. 'compact' — a denser horizontal layout for the blog flagship
   * preview. Both variants expose the same inputs, chart, and stat tiles. */
  variant?: 'full' | 'compact';
}

export default function MemoryCalculator({ variant = 'full' }: Props) {
  const [inputs, setInputs] = useState<MemoryModelInputs>(DEFAULTS);
  const chartRef = useRef<HTMLDivElement>(null);
  const plotRef = useRef<uPlot | null>(null);
  const compact = variant === 'compact';

  const outputs = useMemo(() => computeMemoryModel(inputs), [inputs]);

  const set = <K extends keyof MemoryModelInputs>(key: K) => (value: MemoryModelInputs[K]) =>
    setInputs((prev) => ({ ...prev, [key]: value }));

  // Sweep vocab size from 8K to 256K at the current other settings, showing
  // how baseline (unfused) memory scales linearly with V while the fused
  // kernel's memory stays flat — the actual claim this widget exists to
  // demonstrate.
  const sweep = useMemo(() => {
    const points = 24;
    const minV = 8_000;
    const maxV = 256_000;
    const vocabs: number[] = [];
    const baseline: number[] = [];
    const fused: number[] = [];
    for (let i = 0; i < points; i++) {
      const v = Math.round(minV + ((maxV - minV) * i) / (points - 1));
      const { baselineActivationBytes, fusedActivationBytes } = computeMemoryModel({
        ...inputs,
        vocabSize: v,
      });
      vocabs.push(v);
      baseline.push(baselineActivationBytes / 1024 ** 3);
      fused.push(fusedActivationBytes / 1024 ** 3);
    }
    return { vocabs, baseline, fused };
  }, [inputs]);

  useEffect(() => {
    if (!chartRef.current) return;

    const rootStyles = getComputedStyle(document.documentElement);
    const inkMuted = rootStyles.getPropertyValue('--ink-muted').trim() || '#5b6660';
    const accent = rootStyles.getPropertyValue('--accent').trim() || '#c2410c';
    const rule = rootStyles.getPropertyValue('--rule').trim() || '#c9cdc5';
    const font = '11px "IBM Plex Mono"';

    const opts: uPlot.Options = {
      width: chartRef.current.clientWidth,
      height: compact ? 170 : 220,
      padding: [12, 12, 0, 0],
      cursor: { drag: { x: false, y: false } },
      series: [
        {},
        { label: 'Baseline (unfused)', stroke: inkMuted, width: 2 },
        { label: 'Fused kernel', stroke: accent, width: 2 },
      ],
      axes: [
        {
          stroke: inkMuted,
          grid: { stroke: rule, width: 1 },
          font,
          label: 'Vocabulary size',
          labelFont: font,
          labelSize: 22,
          values: (_u, vals) => vals.map((v) => `${Math.round(v / 1000)}K`),
        },
        {
          stroke: inkMuted,
          grid: { stroke: rule, width: 1 },
          font,
          label: 'Peak activation memory (GB)',
          labelFont: font,
          labelSize: 22,
          values: (_u, vals) => vals.map((v) => `${v.toFixed(0)}GB`),
        },
      ],
      legend: { show: !compact },
      scales: { x: { time: false } },
    };

    plotRef.current?.destroy();
    plotRef.current = new uPlot(opts, [sweep.vocabs, sweep.baseline, sweep.fused], chartRef.current);

    return () => {
      plotRef.current?.destroy();
      plotRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sweep, compact]);

  const presetRow = (
    <div className="flex flex-wrap gap-2">
      {PRESETS.map((preset) => (
        <button
          key={preset.name}
          type="button"
          onClick={() => setInputs(preset)}
          aria-pressed={inputs === preset}
          className="rounded border border-rule px-2 py-1 font-mono text-[11px] text-ink-muted transition-colors hover:border-accent hover:text-accent focus-visible:outline-2 focus-visible:outline-accent"
        >
          {preset.name}
        </button>
      ))}
    </div>
  );

  const headline = (
    <div>
      <p className="font-mono text-xs tracking-wide text-ink-muted uppercase">
        Peak memory reduction
      </p>
      <p
        className={`mt-1 font-mono leading-none font-semibold text-accent tabular-nums ${compact ? 'text-5xl' : 'text-6xl'}`}
      >
        {outputs.deltaPct.toFixed(0)}%
      </p>
      <p className="mt-2 font-mono text-xs text-ink-muted">
        {formatBytes(outputs.baselineActivationBytes)} → {formatBytes(outputs.fusedActivationBytes)}
      </p>
    </div>
  );

  const dtypeField = (
    <div>
      <label htmlFor="mc-dtype" className="block font-mono text-xs text-ink-muted">
        dtype
      </label>
      <select
        id="mc-dtype"
        className="mt-1 w-full border border-rule bg-bg px-2 py-1.5 font-mono text-sm text-ink focus-visible:outline-2 focus-visible:outline-accent"
        value={inputs.dtype}
        onChange={(e) => set('dtype')(e.target.value as Dtype)}
      >
        {(Object.keys(DTYPE_LABELS) as Dtype[]).map((d) => (
          <option key={d} value={d}>
            {DTYPE_LABELS[d]}
          </option>
        ))}
      </select>
    </div>
  );

  const inputFields = (
    <>
      <NumberField
        label="Vocabulary size"
        value={inputs.vocabSize}
        min={1000}
        max={500_000}
        step={1000}
        onChange={set('vocabSize')}
      />
      <NumberField
        label="Hidden dimension"
        value={inputs.hiddenDim}
        min={128}
        max={16384}
        step={128}
        onChange={set('hiddenDim')}
      />
      <NumberField label="Layers" value={inputs.numLayers} min={1} max={200} onChange={set('numLayers')} />
      {dtypeField}
      <NumberField
        label="Batch size"
        value={inputs.batchSize}
        min={1}
        max={1024}
        onChange={set('batchSize')}
      />
      <NumberField
        label="Sequence length"
        value={inputs.seqLen}
        min={1}
        max={131_072}
        step={128}
        onChange={set('seqLen')}
      />
    </>
  );

  const statTiles = (
    <dl className="grid grid-cols-2 gap-px border-t border-rule bg-rule sm:grid-cols-4">
      {[
        ['Total parameters', formatCount(outputs.totalParams)],
        ['Embed + LM-head share', `${(outputs.embedShare * 100).toFixed(1)}%`],
        ['Baseline activation mem.', formatBytes(outputs.baselineActivationBytes)],
        ['Bytes / decode step', formatBytes(outputs.bytesPerDecodeStep)],
      ].map(([label, value]) => (
        <div key={label} className="bg-bg px-4 py-3">
          <dt className="font-mono text-[11px] text-ink-muted uppercase">{label}</dt>
          <dd className="mt-1 font-mono text-lg text-ink tabular-nums">{value}</dd>
        </div>
      ))}
    </dl>
  );

  if (compact) {
    return (
      <div className="widget-shell not-prose border border-rule">
        <div className="grid grid-cols-1 gap-6 p-5 lg:grid-cols-[auto_1fr] lg:items-start">
          <div className="lg:w-64">
            {headline}
            <div className="mt-4">{presetRow}</div>
          </div>
          <div ref={chartRef} className="w-full" />
        </div>
        <div className="grid grid-cols-2 gap-3 border-t border-rule p-5 sm:grid-cols-3 lg:grid-cols-6">
          {inputFields}
        </div>
        {statTiles}
      </div>
    );
  }

  return (
    <div className="widget-shell not-prose my-10 border border-rule">
      <div className="border-b border-rule bg-surface px-5 py-4">
        <p className="font-mono text-xs tracking-wide text-ink-muted uppercase">Memory calculator</p>
        <p className="mt-1 font-body text-sm text-ink-muted">
          Peak activation memory for the linear + cross-entropy computation, baseline vs. the fused
          kernel from this article.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-6 p-5 lg:grid-cols-[1fr_1fr]">
        <div>
          <div className="mb-3">{presetRow}</div>
          <div className="grid grid-cols-2 gap-3">{inputFields}</div>
        </div>

        <div className="flex flex-col justify-center border-t border-rule pt-5 lg:border-t-0 lg:border-l lg:pt-0 lg:pl-6">
          {headline}
        </div>
      </div>

      <div className="border-t border-rule px-5 py-4">
        <div ref={chartRef} className="w-full" />
      </div>

      {statTiles}

      <p className="border-t border-rule px-5 py-3 font-mono text-[11px] leading-relaxed text-ink-muted">
        Simplified model: parameter count uses the standard 12·L·D² approximation (Kaplan et al.,
        2020). Activation memory covers only the fused kernel's own tensors (X, W, dX, dW, and the
        logits Y at fp32) — not full end-to-end training memory (optimizer states, other layers'
        activations, etc.).
      </p>
    </div>
  );
}
