# Kimi-K3 native MoE: single-B200 acceptance

[English](kimi_k3_moe_b200.md) | [中文](kimi_k3_moe_b200_zh.md)

This is a **single-layer, single-rank diagnostic**, not a Kimi-K3 serving
benchmark. OperatorX invokes vLLM 0.29.0's native `KimiMoE` at layer 1 with
deterministic synthetic BF16 inputs and valid packed MXFP4 routed weights. The
timed boundary is BF16 hidden states through routing, latent projections,
routed SiTU experts, routed normalization, the shared path, and final output.
Layer 0 is dense and is not used. No attention, residual, checkpoint weights,
or TP/EP communication is included. The requested Hugging Face model revision
is `f831ab66814297da540d832a5235f8e904f29d06`.

The six cases hold model revision, geometry, precision, fixture seeds, topology,
execution mode, and cache policy fixed. The token count and decode/prefill phase
vary. A decode token count is the number of tokens entering this module call,
not a context length or an observed InferenceX serving batch.

## B200 observations, 2026-10-07

All cases passed output shape, BF16 dtype, finite-value, exact-repeatability,
and top-16 assignment checks. Timing uses 10 warmups, 100 CUDA-event samples,
and best-effort L2 eviction before each measured invocation. Traces were
captured in separate non-authoritative passes.

| Phase | Tokens | Median (µs) | p90 (µs) | CV | Assignments | Active experts | Kernels / streams in trace |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Decode | 4 | 1,534 | 1,568 | 1.61% | 64 | 62 | 35 / 2 |
| Decode | 8 | 1,549 | 1,577 | 1.31% | 128 | 119 | 35 / 2 |
| Decode | 16 | 1,552 | 1,583 | 1.68% | 256 | 226 | 35 / 2 |
| Decode | 32 | 1,674 | 1,742 | 3.04% | 512 | 404 | 35 / 2 |
| Decode | 64 | 2,368 | 2,411 | 2.58% | 1,024 | 608 | 33 / 2 |
| Prefill | 32,768 | 44,239 | 44,988 | 1.11% | 524,288 | 896 | 34 / 1 |

Every case recorded zero dropped assignments. The backend did **not** expose
the token block size needed to calculate actual post-padding assignment count;
`padding.status=unavailable` is intentional, not a claim of zero padding.

vLLM resolved `CutlassExpertsMxfp4` with an MXFP4 activation operand and MXFP4
weights. The trace independently shows two grouped CUTLASS expert GEMMs,
MXFP4 activation conversion before each, `situ_and_mul`, top-k selection,
sorting/shuffling, normalization, ordinary linear projections, and output
combination. The decode traces use two streams, but none of their GPU kernel
intervals overlap; two streams alone do not prove concurrent execution. The
prefill trace uses one stream.
Kernel names alone do not identify every projection's operand shape, and
summed kernel durations or profiler CPU time are **not** authoritative module
latency because profiling changes scheduling and may include overlap.

## Provenance and limits

The rented B200 had 183,359 MiB VRAM, NVIDIA driver 580.126.09, PyTorch
2.13.0, and vLLM 0.29.0. It used a copied Python environment, **not** the
container digest configured in `containers.toml`. The run JSON recorded base
Git SHA `c3c4dd8ed` because the adapter files were copied into a dirty remote
checkout. The native adapter was subsequently committed as `a3148560b`; the
run should not be represented as an immutable-image or clean-checkout result.
The raw JSON and six compressed traces are preserved locally under the ignored
`operatorx/results/kimi_k3_b200_20261007/` directory; all trace files matched
their recorded SHA-256 hashes after transfer.

The K3 paper describes MXFP4 expert weights with MXFP8 activations, whereas
this pinned vLLM runtime selected an MXFP4 activation operand. This is a
measured implementation choice, not evidence that K3 universally uses W4A4
or that this fixture matches production serving. Synthetic weights and a
zero correction bias do not reproduce checkpoint routing. The correctness
gate checks repeatability and structural invariants; an independent numerical
reference for the complete K3 layer remains future work. The prefill case is
one 32,768-token module call; it does not model server chunking or scheduling.

The DeepSeek-R1-0528 decode-4 regression also passed on this upgraded runtime,
with a 1,068 µs median after first-use FlashInfer compilation. Do not compare
that number with K3 as a model or serving-efficiency ranking: the architectures,
expert counts, and precision paths differ.
