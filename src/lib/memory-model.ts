/**
 * Memory model backing the vocabulary/logit memory calculator widget
 * (SPEC §7, embedded in the Fused LinearCE article).
 *
 * Deliberately scoped to exactly what that article covers — the linear
 * projection + cross-entropy computation (X, W, the logits Y, and their
 * gradients) — not full end-to-end training memory. Adding optimizer
 * states, other transformer layers' activations, etc. would let the widget
 * claim to reproduce the article's specific profiled benchmark numbers,
 * which it can't do honestly without replicating PyTorch's allocator
 * behavior. This computes a correct simplified model instead.
 */

export type Dtype = 'bfloat16' | 'float16' | 'float32' | 'int8';

export const DTYPE_BYTES: Record<Dtype, number> = {
  bfloat16: 2,
  float16: 2,
  float32: 4,
  int8: 1,
};

export const DTYPE_LABELS: Record<Dtype, string> = {
  bfloat16: 'bfloat16',
  float16: 'float16',
  float32: 'float32',
  int8: 'int8',
};

export interface MemoryModelInputs {
  vocabSize: number;
  hiddenDim: number;
  numLayers: number;
  dtype: Dtype;
  batchSize: number;
  seqLen: number;
}

export interface MemoryModelOutputs {
  totalParams: number;
  embedShare: number; // fraction of total params in embedding + LM head
  baselineActivationBytes: number; // unfused: materializes the full [N,V] logits (+ grad) in fp32
  fusedActivationBytes: number; // fused/tiled: X, W, dX, dW only — no full [N,V] buffer
  deltaPct: number; // reduction from baseline to fused
  bytesPerDecodeStep: number; // reading the LM-head weight matrix once per generated token
}

const LOSS_DTYPE_BYTES = 4; // fp32 — numerical stability for the loss, per the article's own "best practice" note

export function computeMemoryModel(inputs: MemoryModelInputs): MemoryModelOutputs {
  const { vocabSize: V, hiddenDim: D, numLayers: L, dtype, batchSize: B, seqLen: S } = inputs;
  const bytes = DTYPE_BYTES[dtype];
  const N = B * S;

  // Standard order-of-magnitude transformer parameter count (Kaplan et al.,
  // "Scaling Laws for Neural Language Models", 2020): N_params ≈ 12·L·D².
  const transformerParams = 12 * L * D * D;
  const embedParams = V * D;
  const lmHeadParams = V * D; // untied, as in Llama 3
  const totalParams = transformerParams + embedParams + lmHeadParams;
  const embedShare = (embedParams + lmHeadParams) / totalParams;

  // X, W, dX, dW — present in both the baseline and fused case, at the
  // model's working dtype.
  const xBytes = N * D * bytes;
  const wBytes = D * V * bytes;
  const sharedBytes = 2 * xBytes + 2 * wBytes; // X + dX + W + dW

  // Baseline: materializes the full logits Y and its gradient dY as
  // [N, V] tensors in fp32.
  const logitsBytes = N * V * LOSS_DTYPE_BYTES;
  const baselineActivationBytes = sharedBytes + 2 * logitsBytes;

  // Fused: only ever holds a D_BLOCK × V_BLOCK tile of logits — negligible
  // at any realistic block size, so it's dropped from the total here.
  const fusedActivationBytes = sharedBytes;

  const deltaPct =
    baselineActivationBytes > 0
      ? ((baselineActivationBytes - fusedActivationBytes) / baselineActivationBytes) * 100
      : 0;

  // Autoregressive decoding at batch 1 is memory-bandwidth bound: each new
  // token requires streaming the LM-head weight matrix from HBM once.
  const bytesPerDecodeStep = D * V * bytes;

  return {
    totalParams,
    embedShare,
    baselineActivationBytes,
    fusedActivationBytes,
    deltaPct,
    bytesPerDecodeStep,
  };
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes.toFixed(0)} B`;
  const units = ['KB', 'MB', 'GB', 'TB'];
  let value = bytes;
  let unitIndex = -1;
  do {
    value /= 1024;
    unitIndex++;
  } while (value >= 1024 && unitIndex < units.length - 1);
  return `${value.toFixed(value >= 100 ? 0 : value >= 10 ? 1 : 2)} ${units[unitIndex]}`;
}

export function formatCount(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return n.toFixed(0);
}

/** A handful of representative real-model configurations, used for the
 * no-JS static fallback table and as widget presets. */
export const PRESETS: Array<{ name: string } & MemoryModelInputs> = [
  {
    name: 'Llama 3 8B',
    vocabSize: 128_256,
    hiddenDim: 4096,
    numLayers: 32,
    dtype: 'bfloat16',
    batchSize: 8,
    seqLen: 8192,
  },
  {
    name: 'Llama 3 70B',
    vocabSize: 128_256,
    hiddenDim: 8192,
    numLayers: 80,
    dtype: 'bfloat16',
    batchSize: 4,
    seqLen: 8192,
  },
  {
    name: 'GPT-2 Small',
    vocabSize: 50_257,
    hiddenDim: 768,
    numLayers: 12,
    dtype: 'float32',
    batchSize: 16,
    seqLen: 1024,
  },
  {
    name: 'Mistral 7B',
    vocabSize: 32_000,
    hiddenDim: 4096,
    numLayers: 32,
    dtype: 'bfloat16',
    batchSize: 8,
    seqLen: 8192,
  },
];
