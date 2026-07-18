#ifndef STILES_MULTIVERSION_HPP
#define STILES_MULTIVERSION_HPP
// ─────────────────────────────────────────────────────────────────────────────
// Function multi-versioning (FMV) for hot, compute-bound sTiles-own kernels.
//
// When STILES_FMV is defined (x86-64 + GCC), a function marked STILES_MULTIVERSION
// is compiled into TWO variants — the baseline "default" (== the -march=haswell/
// AVX2 code we already ship) and an "avx512f" clone — plus a one-time ifunc
// resolver that binds the right variant at load time from the actual CPU. Result:
// ONE portable binary that uses AVX-512 on capable CPUs and the unchanged AVX2
// code everywhere else. This lifts the sTiles-own C++ off the fixed AVX2 baseline
// WITHOUT giving up portability (the FLOPs are already in MKL/libxsmm, which
// runtime-dispatch on their own; this covers our own numeric loops).
//
// "Can't hurt" by construction:
//   * Default build (STILES_FMV undefined) -> macro expands to nothing ->
//     byte-identical to today.
//   * With STILES_FMV, the "default" clone is the SAME AVX2 codegen as today, so
//     an AVX2-only machine runs exactly what it runs now. The resolver is ifunc:
//     it fires ONCE at load and patches the PLT, so there is no per-call cost.
//   * The avx512f clone only ever runs on CPUs that report AVX-512.
//
// Apply ONLY to coarse, compute-bound leaf kernels: target_clones blocks inlining
// of the annotated function, so tagging a tiny hot helper would REGRESS it (the
// inline helpers get inlined INTO an annotated coarse kernel and inherit its ISA,
// which is exactly what we want — annotate the driver, not the helper).
//
// Gated to GCC/x86-64: clang spells multiversioning differently, and ARM (Apple/
// Linux arm64) has no AVX-512.  Toggle at build time with STILES_FMV=1 (make.inc).
// ─────────────────────────────────────────────────────────────────────────────
#if defined(STILES_FMV) && defined(__x86_64__) && defined(__GNUC__) && !defined(__clang__)
#  define STILES_MULTIVERSION __attribute__((target_clones("default", "avx512f")))
#else
#  define STILES_MULTIVERSION
#endif

#endif  // STILES_MULTIVERSION_HPP
