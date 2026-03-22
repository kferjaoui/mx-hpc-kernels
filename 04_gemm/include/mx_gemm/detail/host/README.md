# Optimizing GEMM on x86: From Naive to Near-Peak on Skylake-X

This article walks through my sequential optimization of General Matrix-Matrix Multiplication (GEMM) on my specific CPU architecture, progressing from a naive triple loop to a near-peak vectorized microkernel. I ground each optimization in the hardware characteristics of my target machine and justify every choice through cache behavior, arithmetic intensity analysis, and roofline reasoning.

The GEMM operation computed is:

$$C \leftarrow \alpha \cdot A \times B + \beta \cdot C$$

where $A$ is $N \times K$, $B$ is $K \times M$, and $C$ is $N \times M$, with scalars $\alpha$ and $\beta$.

All implementations support both **RowMajor** and **ColMajor** memory layouts.

> **Notation.** Throughout this document, I use N for the number of rows of A (and C), M for the number of columns of B (and C), and K for the shared inner dimension (columns of A = rows of B). This N/M convention, rather than the more common BLAS M/N/K ordering, reflects the rows-first matrix notation I was taught and personally find more natural..



> **Scope.** This analysis covers CPU-side GEMM only. A companion analysis of GPU GEMM on my RTX 4070 is planned as future work.

---

## Target Architecture: Intel Core i9-7940X (Skylake-X)

Understanding the hardware is not optional for writing high-performance GEMM as every algorithmic decision is a direct response to a specific hardware constraint. I tuned all optimizations for my personal machine with the following specifications:

| Resource | Specification |
|---|---|
| Microarchitecture | Skylake-X (Server/HEDT) |
| Cores / Threads | 14 physical / 28 logical (Hyperthreading) |
| Base Clock | 3.1 GHz |
| ISA Extensions | AVX-512F, AVX-512BW, AVX-512DQ, AVX-512CD, AVX-512VL |
| FMA Units | 2 × 512-bit FMA ports (port 0 and port 5) |
| Vector Registers | 32 × ZMM (512-bit), each holding 8 `double` or 16 `float` |
| Mask Registers | 8 × k-registers (k0–k7), used for AVX-512 predicated operations |
| General-Purpose Registers | 16 × 64-bit (RAX, RBX, ..., R8–R15) |

### Cache Hierarchy

| Level | Size | Line Size | Latency | Bandwidth (per core) | Inclusion Policy |
|---|---|---|---|---|---|
| L1 Data | 32 KB | 64 B | 4 to 5 cycles | ~128 B/cycle (2 loads + 1 store at 512-bit) | n/a |
| L2 Unified | 1024 KB | 64 B | ~12 cycles | ~64 B/cycle (1 line/cycle) | Inclusive of L1 |
| L3 Shared | 19.25 MB | 64 B | ~40 cycles | ~32 B/cycle (mesh interconnect) | Non-inclusive (NINE) |
| DRAM | n/a | n/a | ~200+ cycles | ~80 GB/s total (quad-channel DDR4) | n/a |

All cache levels use **64-byte cache lines** (8 `double`-precision values per line).

### Compute Ceiling

Peak double-precision throughput per core:

> 2 FMA ports × 8 doubles/FMA × 2 FLOPs/FMA (1 multiply + 1 add) = **32 FLOPs/cycle**

At 3.1 GHz base clock: **~99 GFLOP/s per core**.

A note on clock cycles: one cycle at 3.1 GHz is 1/(3.1 x 10^9), approximately 0.32 nanoseconds. This is the same fundamental time unit for all operations, whether ALU execution, cache lookups, or pipeline stages. All latency and bandwidth figures above are measured in this same clock cycle.

### Register File

Skylake-X provides 32 ZMM registers, each 512 bits wide (8 doubles). These are the same physical storage as the YMM (256-bit, AVX2) and XMM (128-bit, SSE) registers. ZMM0 contains YMM0 in its lower half, which contains XMM0 in its lower quarter. Scalar `double` operations use the low 64 bits of an XMM register; there is no separate scalar floating-point register file on x86 [1].

In addition, 16 general-purpose 64-bit registers (RAX through R15) hold integers, pointers, and loop counters. Eight AVX-512 mask registers (k0 through k7) support predicated vector operations.

### Memory Hierarchy Behavior

When data is first loaded (cold miss), the cache line is fetched from DRAM and installed in L3, L2, and L1 simultaneously. When L1 evicts a line under capacity pressure, the copy in L2 remains valid because no data movement occurs (L2 is inclusive of L1 on Skylake). When L2 evicts, the line is pushed to L3 as a victim. On Skylake-X, the L3 is non-inclusive, sometimes referred to as NINE (Non-Inclusive, Non-Exclusive): a line in L1/L2 is not guaranteed to have a copy in L3. This was a departure from the inclusive L3 of previous Intel architectures (Sandy Bridge through Broadwell) [2].

A critical but often overlooked detail: dirty cache lines (modified data not yet written back) exist only in the cache that last wrote them. The DRAM copy is stale until write-back occurs, triggered by eviction, coherence requests from another core, or explicit flush instructions. For GEMM, the C matrix accumulator lines are repeatedly modified in L1 and do not touch DRAM until much later.

### Hardware Prefetching

Skylake-X includes multiple hardware prefetchers per core [3]. The most relevant for GEMM are the **L1 data streamer** and the **L2 spatial prefetcher**. The L1 prefetcher (DCU streamer) detects sequential access patterns: accesses to cache line N, then N+1 triggers speculative fetches of N+2, N+3, and so on. It can track 2 to 4 independent forward or backward streams simultaneously. The L2 streamer is more aggressive and looks further ahead.

Sequential (stride-1) access patterns are the ideal case: the prefetcher runs ahead by several cache lines, hiding memory latency behind computation. Large-stride access patterns (stride much greater than 1 cache line) defeat the prefetcher, and accesses appear essentially random. Every load then pays the full latency penalty to whichever cache level (or DRAM) holds the data.

---

## The Roofline Framework

Every optimization in this document is understood through the roofline model [4]: **attainable performance = min(peak compute, arithmetic intensity x bandwidth)**.

There is a single compute ceiling per core (approximately 99 GFLOP/s at base clock) but a different bandwidth at every level of the memory hierarchy. The arithmetic intensity (AI) required to be compute-bound therefore varies dramatically depending on which boundary we measure:

| Boundary | Bandwidth (per core) | AI Threshold (compute-bound) |
|---|---|---|
| L1 to Registers | ~397 GB/s | > 0.25 FLOP/byte |
| L2 to L1 | ~198 GB/s | > 0.50 FLOP/byte |
| L3 to L2 | ~99 GB/s | > 1.0 FLOP/byte |
| DRAM to L3 | ~5.7 GB/s | > 17 FLOP/byte |

GEMM has O(N^3) FLOPs on O(N^2) data, giving a theoretical DRAM-level AI of approximately 2n/3 FLOPs/byte for square n x n matrices. For n = 1024, that is approximately 680 FLOPs/byte, massively compute-bound in theory. The challenge is realizing this potential at every cache level, all the way down to the register file.

Each of the optimizations I present, attempts to push the bottleneck one level deeper into the hierarchy, until the only remaining constraint is the FMA execution units themselves.

---

## Optimization 1: Transposing B (RowMajor) / A (ColMajor)

### The Problem: Stride-N Access on B

The naive GEMM computes `C(i,j) += A(i,k) * B(k,j)` with loop order {i, j, k}. In RowMajor layout, elements within the same row are contiguous in memory. `A(i,k)` and `A(i,k+1)` are adjacent, so the inner k loop streams along A's row with stride-1 access, and each 64-byte cache line fetch delivers 8 useful `double` values.

For B, the picture is reversed. `B(k,j)` and `B(k+1,j)` are M elements apart, one full row stride. Every increment of k lands on a completely different cache line. Each cache line fetch brings 8 doubles into L1, but only 1 is used before moving on: a 12.5% utilization rate. For a matrix with M = 1024 columns, the stride between consecutive B accesses is 1024 x 8 = 8192 bytes = 128 cache lines. This large-stride pattern defeats the L1 hardware prefetcher, which tracks strides of only 1 to 2 cache lines, causing stalls that propagate through L2, L3, and potentially all the way to DRAM.

### The Fix: Materialize BT = transpose(B)

Replacing `B(k,j)` with `BT(j,k)` converts B's access pattern from stride-M to stride-1. Now both operands stream contiguously along k, and the hardware prefetcher handles both streams efficiently.

**Cache line fetch reduction for B:** before transposition, the inner loop issues K cache line loads for B (one per k increment, each touching a new line). After transposition, it issues only ceil(K/8) loads, an **8x reduction**. More importantly, the sequential pattern enables speculative prefetching, so the remaining loads are largely hidden behind computation rather than stalling the pipeline.

### Data Reuse Across Loops

With loop order {i, j, k}, fixing i and sweeping j, A's row i is reused across all M iterations of j. If K is at most approximately 2000 (so the row fits in about 16 KB, leaving room in the 32 KB L1 for BT's streaming row), row i of A stays L1-resident for the entire j sweep. For larger K, the row spills to L2 (1 MB on my chip), but the sequential access pattern still benefits from L2 prefetching at roughly 12-cycle latency instead of L1's 4 to 5 cycles.

### Cost of the Transposition

The transpose is O(K x M), one read and one write per element. The GEMM itself performs O(N x K x M) multiply-accumulate operations. The overhead is amortized by a factor of N and is negligible for any practically sized matrix. Transposing a 1024 x 1024 matrix moves approximately 8 MB of data; the GEMM performs approximately 2 billion FLOPs.

I perform the transposition using a tiled transpose with a tile size of 32, which keeps the transpose itself L1-friendly and avoids TLB (Translation Lookaside Buffer) thrashing during the copy.

### ColMajor Symmetry

For ColMajor layout, columns are contiguous. B's access `B(k,j)` at fixed j is already stride-1 (good), but A's access `A(i,k)` at fixed i is stride-N (bad). The symmetric fix is to transpose A into AT, making `AT(k,i)` contiguous in k. I also swap the i/j loop ordering to prioritize column-contiguous writes to C.

### Arithmetic Intensity

At the L1 boundary, the inner loop's AI remains low: for each (i,j) pair, the k loop loads K elements of A and K elements of BT, performing 2K FLOPs. That is 2K FLOPs / (2 x K x 8 bytes) = ** 0.125 FLOPs/byte**, below the 0.25 threshold for compute-bound operation at L1. Spatial locality is fixed; arithmetic intensity is not.

### Bottleneck After This Optimization

Even without explicit tiling, the cache hierarchy provides implicit buffering. Evicted L1 lines land in L2 (approximately 12 cycles), and L3 (approximately 40 cycles) acts as a last defense before DRAM (200+ cycles). The pre-transposition access pattern on B was so pathological that it thrashed all cache levels for large M. Post-transposition, every level of the hierarchy handles the sequential streams efficiently. But the fundamental problem, low AI at the register/L1 boundary, remains.

---

## Optimization 2: Cache Blocking (Tiling)

### The Idea

Instead of operating on entire rows/columns of A and B, I partition the computation into tiles: `C_tile(I,J) += A_tile(I,L) x B_tile(L,J)` for each tile index triplet (I, J, L). The tile sizes, nc (rows), kc (depth), and mc (columns), are chosen so that the most-reused operand fits in a specific cache level.

### Tile Loop Ordering

Six orderings of the tile indices (I, J, L) are possible. The choice determines which operand stays "hot" in cache and which streams through.

**{I,J,L} and {J,I,L}, L innermost (depth scan per C-tile):**
For each (I,J) pair, the L loop sweeps through all depth tiles, computing partial contributions to C_tile(I,J). The same C_tile is updated ceil(K/kc) times, requiring it to stay cache-resident across L iterations. Neither A_tile nor B_tile is reused, as each L iteration brings a fresh pair. The AI does not improve meaningfully because no input operand benefits from temporal reuse across the innermost loop.

**{I,L,J}, J innermost (column scan of B at fixed A-tile):**
For a fixed (I,L) pair, the J loop sweeps through all column tiles of B. A_tile(I,L) is loaded once into L2 and reused across all ceil(M/mc) iterations of J. This is the **optimal ordering for RowMajor**: A-tile reuse drives the AI up.

**{J,L,I}, I innermost (row scan of A at fixed B-tile):**
Symmetric to {I,L,J}: the B_tile is reused across successive I-tiles, while A_tile is streamed instead. For ColMajor layouts this can be attractive because it aligns reuse with the right-hand operand and can pair naturally with contiguous updates along I inside C. In my code, I kept {I,L,J} for both layouts for simplicity; with layout-aware tile sizes, the performance difference may be small and is ultimately a benchmarking question.

**{L,I,J} and {L,J,I}, L outermost (depth strip by depth strip):**
Every C_tile is visited ceil(K/kc) times, once per L value. The entire C matrix is rescanned each L iteration. Neither A-tile nor B-tile gets meaningful reuse within a single L sweep. The repeated read-modify-write traffic on C dominates, and the AI is limited to approximately kc/8 FLOPs/byte, roughly 4x worse than {I,L,J}.

**My chosen ordering: {I,L,J} (IKJ tile ordering in the code):**
The left operand tile (A_tile or A^T_tile, depending on layout) is reused across the sweep in J, while right operand tiles stream through. This keeps the reused left tile hot in L2 and leaves the layout-specific locality mainly to the tile-internal loop order.

### Arithmetic Intensity at L2

For a fixed (I,L) pair sweeping J across all column tiles, I can quantify the data movement:

- **A_tile:** loaded once, size nc x kc elements, reused across all J iterations
- **B_tile:** for each J, a fresh tile of kc x mc elements; total over the J sweep: kc x M elements
- **C_tile:** for each J, read-modify-write of nc x mc elements; total: 2 x nc x M elements

FLOPs: 2 x nc x kc x M.

The arithmetic intensity at the L2 boundary is:

**AI_L2 = (2 x nc x kc x M) / (8 x (nc x kc + kc x M + 2 x nc x M))**

For large M, the (kc + 2 x nc) x M term dominates the denominator The asymptotic limit is therefore:

**AI_L2 ≈ (kc x nc) / 4 x (kc + 2 x nc) FLOPs/byte**

With nc = 256 and kc = 128: **AI_L2 ≈ 12.8 FLOPs/byte**. The L2 boundary requires only 0.5 FLOPs/byte to be compute-bound. My tiled version exceeds this by over 20x, making it **solidly compute-bound at L2**. In other words, cache blocking solves the macro-level bandwidth problem; the remaining bottleneck is deeper in the hierarchy, at the L1-to-register boundary inside the scalar inner loop.

### Tile Sizing for My i9-7940X

The strategy for the {I,L,J} ordering is to assign each operand to the appropriate cache level:

- **A_tile (nc x kc):** the reused operand, which must fit in L2 (1 MB)
- **B_tile (kc x mc):** streams through, benefits from L3 residency or prefetch
- **C_tile (nc x mc):** the inner loop touches one row at a time, so rows must be L1-friendly

**My chosen parameters: nc = 256, kc = 128, mc = 160.**

| Tile | Dimensions | Memory Footprint | Target Residence |
|---|---|---|---|
| A_tile | 256 x 128 | 256 KB | L2 (1 MB), fits with ample headroom |
| B_tile | 128 x 160 | 160 KB | Streams through; fits in L2 alongside A_tile if needed |
| C_tile | 256 x 160 | 320 KB total, but 1.28 KB per active row | L1 for active row; L2/L3 for full tile |

The total hot working set in L2 is A_tile (256 KB) plus streaming B lines plus active C rows, approximately 300 to 420 KB, targeting about 60 to 70% of the 1 MB L2 capacity. I leave headroom for B's streaming lines and other transient data.

### Bottleneck After This Optimization

The tiling makes the computation beautifully compute-bound at L2. But zooming into the innermost loop: for each (i,j) element pair, the k loop loads one element of A and one of BT, performing 2 FLOPs. That is 16 bytes for 2 FLOPs = **0.125 FLOPs/byte at the L1-to-register boundary**, still below the 0.25 threshold. The cache-blocked version is **L1-bandwidth-bound**. The data is close (in L2 or L1), but I am not extracting enough computation per byte crossing into registers.

---

## Optimization 3: Microkernel (Register-Level Tiling)

### The Problem: Low AI at L1 to Registers

My cache-blocked inner loop computes one scalar multiply-add per iteration, loading two 8-byte values for 2 FLOPs: AI = 0.125 FLOPs/byte. With L1 delivering 128 bytes/cycle and peak compute at 32 FLOPs/cycle, the threshold for compute-bound operation is 0.25 FLOPs/byte. The inner loop falls short by 2x.

### The Idea: Outer-Product Accumulation in Registers

Instead of computing C one element at a time, the idea is to partition each tile into **micro-panels**: A is divided into panels of nr rows (across the full kc depth), and B into panels of mr columns (across the full kc depth). The innermost computation accumulates an nr x mr block of C entirely in the register file.

At each step of the k loop, the microkernel loads nr elements from A's micro-panel and mr elements from B's micro-panel, then computes their **outer product**: nr x mr multiply-adds updating the full C micro-tile simultaneously. Each A element is reused mr times (across B's columns), and each B element is reused nr times (across A's rows). This **multiplicative two-dimensional reuse** is what drives the AI up.

### Why Micro-Panels, Not Micro-Tiles (i.e. Do Not Partition kc)

One might consider partitioning the depth kc into smaller kr blocks, creating true micro-tiles of dimensions (nr, kr) and (kr, mr). I chose not to do this. The entire benefit of the microkernel rests on the C micro-tile entering registers once and staying for the full kc sweep. Breaking kc into kr chunks would require writing back and reloading the C accumulator between chunks, destroying the amortization of C's register residency.

Additionally, short inner loops (kr = 8 instead of kc = 128) increase the fraction of cycles spent on loop control, including branch instructions, index increments, and comparisons, by 16x, with no compensating gain.

### Microkernel Loop Ordering

The choice of loop ordering within the micro-panel is critical and constrained by register pressure:

**{i,j,k}, k innermost:** for each (i,j) pair, the code sweeps k through kc iterations. This computes one C element at a time with no register-level reuse of A across j or B across i. AI remains 0.125 FLOPs/byte: identical to the non-microtiled version. No improvement.

**{i,k,j}, j innermost:** at fixed (i,k), A's element is loaded once and reused across mr columns of B, which is good. But when i increments, the same B elements at that k must be reloaded. The reuse is one-dimensional: A is reused across j, but B is not reused across i.

**{k,i,j}, k outermost (outer product, my chosen ordering):** the C micro-tile (nr x mr values) enters registers once and stays for all kc iterations. Each k step loads nr values from A and mr values from B, performing nr x mr FMAs. Both A and B enjoy reuse: each A element feeds mr FMAs, each B element feeds nr FMAs. This is the **only ordering that achieves full two-dimensional register reuse**.

A subtle but important detail is the role of `BT`. In the RowMajor scalar microkernel I transpose `B` once, then fetch the current `mr` values from `BT` at fixed `k`. That immediate `mr`-wide fetch is still strided across `j`, but each accessed row of `BT` is contiguous in `k`. As `k` advances, those accesses become sequential streams through `BT`, which is much more cache- and prefetch-friendly than walking the original RowMajor `B` down a column.

Another subtlety is that, in this **scalar** microkernel, the accumulation order and the write-back order happen to coincide nicely: the same local traversal that gives good register reuse also gives contiguous RowMajor stores into `C`.

That should not be read as a general principle. Accumulation order is chosen for **operand reuse in registers**, while write-back order is chosen for **store locality in `C`**. In the scalar kernel, both happen to favor the same local ordering. In the SIMD kernel, these two concerns become more visibly distinct: the accumulation order is driven by reuse of the loaded SIMD vector from `B`, whereas the write-back step is still governed by the layout of `C` (see the SIMD section).  

### Arithmetic Intensity at L1 to Registers

The C micro-tile (nr x mr = 6 x 8 = 48 doubles) resides entirely in the register file throughout the k loop. It is never read from or written to L1 during the accumulation. Only a single write-back occurs at the end, contributing 2 x nr x mr x 8 / kc ≈ 6 bytes per k iteration, versus approximately 112 bytes/iteration for A and B. This is negligible (about 5% of total traffic).

Per k iteration, the sustained traffic across the L1-to-register boundary is:

- **A:** nr elements x 8 bytes = nr x 8 bytes, each reused mr times
- **B:** mr elements x 8 bytes = mr x 8 bytes, each reused nr times

FLOPs per k iteration: nr x mr x 2.

**AI_registers = (2 x nr x mr) / (8 x (nr + mr)) = (2 x 6 x 8) / (8 x 14) = 96 / 112 ≈ 0.86 FLOPs/byte**

This exceeds the 0.25 threshold. The microkernel is **compute-bound at L1**. The bottleneck has shifted from memory bandwidth to the FMA execution units.

### Register Allocation: Why nr = 6, mr = 8

In the current scalar microkernel, the `C` accumulator consists of `nr x mr` **live scalar partial sums**. With `nr = 6` and `mr = 8`, it is 48 partials of `C`, together with the current `A` scalar, the buffered `B` values, loop counters, pointers, and temporaries, all of which contribute to register pressure. 

Another useful way to think about why `6 x 8` is a natural shape is that it already matches the **forthcoming SIMD mapping** (below). With AVX-512 in double precision, the SIMD width is `W = 8`, so the same logical `6 x 8` tile would later correspond to

`nr x ceil(mr / W) = 6 x 1 = 6`

vector accumulator registers for `C`. That is comfortably below the architectural limit of 32 ZMM registers.

If `nr x mr` grows too large (for example `nr = 16`, `mr = 16`, giving 256 accumulator values), then in the SIMD picture the accumulator alone would already require 32 ZMM registers. In practice this leaves no room for the rest of the live state, so the compiler is very likely to emit **spill instructions** or choose a less efficient schedule. These spills appear in assembly as stores to stack-backed memory and later reloads, for example `vmovapd [rsp+offset], zmmN` and `vmovapd zmmN, [rsp+offset]` in a vectorized kernel. Once that happens, part of the accumulator is no longer effectively “free” in registers, and the arithmetic-intensity advantage of the microkernel starts to erode.

Unlike caches, registers have **no hardware eviction policy** (no LRU, no victim buffer). The compiler statically assigns variables to registers at compile time. When live values exceed physical registers, spilling is the only option, and it is entirely the compiler's decision, visible in the generated assembly.

### Three-Level Tiling Summary

| Level | What Resides There | Size | Reuse Pattern |
|---|---|---|---|
| **Registers** | C micro-tile (nr x mr) | 48 doubles = 6 ZMM regs | Accumulated over kc iterations, written back once |
| **L1** | Active micro-panel rows of A (kc) and B (kc) | ~2 KB streaming | Sequential, prefetcher-friendly |
| **L2** | A macro-tile (nc x kc) | 256 KB | Reused across all J (column) tiles |

### Bottleneck After This Optimization

The microkernel removes the obvious L1-bandwidth bottleneck of the cache-blocked scalar inner loop by increasing the amount of work done per value loaded from L1. At that point, the kernel becomes much more **compute-oriented**: the limiting factor is no longer “too little reuse of A and B”, but how efficiently the core can sustain the fused multiply-add stream.

That said, the current implementation is a **scalar** microkernel, and as a consequence, it uses only 64 bits of the 512-bit ZMM datapath per FMA instruction, wasting **87.5% of the execution unit width**. Peak scalar throughput is 2 FMAs/cycle x 1 double/FMA x 2 FLOPs = 4 FLOPs/cycle, versus the 32 FLOPs/cycle the hardware delivers with full AVX-512 vectors. So the next optimization step is not more cache blocking, but **vectorization of the microkernel itself**.

---

## Optimization 4: SIMD Vectorization (AVX-512)

### The Gain: Filling the Full Datapath

The microkernel established compute-bound operation where the FMA units are the bottleneck, not any cache level's bandwidth. But scalar FMA instructions process one double per lane. AVX-512 FMA instructions process 8 doubles simultaneously in a single ZMM register. Vectorization does not change the arithmetic intensity (the ratio of FLOPs to bytes transferred is the same); it increases the **throughput of FLOPs per cycle** by filling all 8 lanes of the 512-bit datapath.

The key instruction is `vfmadd231pd zmm_acc, zmm_a_broadcast, zmm_b`: a fused multiply-add where `zmm_a_broadcast` contains a_val replicated across all 8 lanes (via `vbroadcastsd`), `zmm_b` contains 8 contiguous elements from B's row, and `zmm_acc` is one row of the C micro-tile. **One instruction, 16 FLOPs.** With 2 FMA ports: **32 FLOPs/cycle**, the full peak of the chip.

### Why the SIMD Kernel Changes the Dataflow

The SIMD kernel keeps the same macro-level GEMM structure, but it changes the preferred inner-kernel dataflow completely.

In the scalar RowMajor microkernel, I transpose `B` to `BT` because the useful reuse happens along the `k` dimension: I want the right-hand operand to behave as a sequential stream across the full depth sweep. In the SIMD RowMajor kernel, that is no longer the dominant concern. The expensive object to reuse is now the **SIMD vector load** itself.

At fixed `k`, the RowMajor SIMD path loads a full vector of contiguous elements

`B(k, j : j + W - 1)`

where `W = 8` for double precision on AVX-512 (and `mr_simd = 1` for `mr = W`). That vector is then reused across all `nr` rows of the current `A` micro-panel. So the useful hot-loop structure becomes effectively **`{k,j,i}`**:

```cpp
using vT = stdx::native_simd<T>;              // 8-wide for double on AVX-512
constexpr index_t W = vT::size();              // = 8
constexpr index_t mr_simd = (mr + W - 1) / W; // ZMM vectors per C micro-tile row

vT acc[nr * mr_simd]{};  // C micro-tile in registers

for (index_t k = k0; k < k_end; ++k) {
    for (index_t j_simd = 0; j_simd < mr_simd; ++j_simd) {
        vT b_vec{};                                      // one SIMD load, reused for all rows `il` in the A micro-panel
        stdx::where(mask[j_simd], b_vec)
            .copy_from(&B(k, j + j_simd*W), stdx::element_aligned);
    for (index_t il = 0; il < nr_actual; ++il) {         // `l` refers here to the local indexing within the micro-panel
        const T a_val = A(i + il, k);                    // scalar load, broadcast by FMA
            acc[il*mr_simd + j_simd] += a_val * b_vec;   // vector FMA: 16 FLOPs
        }
    }
}
```

- `k` stays outermost so the `C` micro-tile remains live across the full depth sweep
- at fixed `k`, one SIMD vector `b_vec` is loaded from contiguous elements of `B`
- that same `b_vec` is reused across all rows `il` of the micro-panel
- the cheaper scalar `a_val` values stream through and are broadcast into the FMAs

This is the key reason the SIMD RowMajor kernel **does not need `BT`**. Once the vectorization axis is `j`, the natural contiguous access direction is already present in RowMajor `B`. Transposing `B` would no longer help the hot SIMD load; it would only add preprocessing cost and extra memory footprint.

This optimization changes neither the mathematical GEMM nor the arithmetic intensity at a high level;; instead, it changes the instruction mix and the throughput per cycle by utilizing the AVX-512 datapath much more effectively. In the common case `mr = W = 8`, (`mr_simd = 1`, will an all-true mask), each `k` iteration reduces to `nr = 6` broadcast-load-FMA sequences. That corresponds to 6 vector FMAs per `k` step, or **96 FLOPs**.

### Handling Edge Tiles (mr not equal to W)

When M is not a multiple of mr, the last micro-tile has fewer than mr valid columns. AVX-512 mask registers (k0 through k7) handle this: lanes beyond the valid count are masked off during loads and stores, preventing out-of-bounds access and keeping masked accumulator lanes at zero.

The `std::experimental::simd` abstraction via `stdx::where(mask, vec)` compiles to native AVX-512 masked operations such as `vmovapd zmm {k1}, [mem]`, keeping the source portable while generating optimal instructions for my target.

### Register Allocation Under Vectorization

The C micro-tile maps to **6 ZMM registers** (one per row, 8 doubles each). Each k iteration uses 1 ZMM for the broadcast A value and 1 ZMM for the B vector load. Total active ZMM usage: approximately 8 to 10 out of 32. No pressure, no spilling.

### Instruction Count Reduction

Beyond FLOP throughput, vectorization reduces the total instruction count by a factor of W = 8. Fewer instructions means less pressure on the instruction decoder (4-wide on Skylake), the reorder buffer (224 entries), and the branch predictor. These second-order effects contribute to actually reaching peak throughput in practice.

### AVX-512 Frequency Throttling

A practical consideration specific to Skylake-X: sustained AVX-512 workloads cause the CPU to **reduce its clock frequency** by approximately 15 to 20%, sometimes more under heavy multi-core load. The 512-bit execution units draw significantly more power, and the chip throttles to stay within its thermal envelope [5]. The effective peak under AVX-512 is roughly **80 to 85 GFLOP/s per core** rather than the nominal 99 GFLOP/s.

A well-tuned GEMM kernel typically achieves 85 to 92% of this effective peak. The gap comes from loop overhead, edge-tile cleanup, tile-transition L1 misses, and AVX-512 frequency ramp-up/ramp-down transients.

### ColMajor Symmetry

For ColMajor, columns are contiguous (along i, not j). The vectorization axis flips: I load a vector of nr elements from A's column (contiguous), broadcast one element of B, and FMA into a column of the C micro-tile. The loop order becomes {k,j,i} with the i dimension vectorized.

---

## The Full Optimization Stack

| Optimization | What It Fixes | Bottleneck Before | Bottleneck After | AI at Critical Boundary |
|---|---|---|---|---|
| **Transposition** | Spatial locality on B | Prefetcher-defeating stride-M access | L1 bandwidth (low AI) | 0.125 FLOP/B at L1 |
| **Cache Blocking** | DRAM/L3 traffic via A-tile reuse | A fetched repeatedly from DRAM | L1 bandwidth (AI still low at L1) | 64 FLOP/B at L2 |
| **Microkernel** | Register-level reuse (outer product) | L1-to-register bandwidth | Scalar FMA throughput (1/8 peak) | 0.86 FLOP/B at registers |
| **SIMD (AVX-512)** | Full datapath utilization (8 lanes) | 87.5% of FMA width wasted | **Peak FMA throughput** | 0.86 FLOP/B at registers |

Each optimization addresses a different level of the hardware hierarchy. They are not interchangeable alternatives but a **cumulative stack** where each layer builds on the one below it. Skipping any one leaves significant performance on the table.

---

## Note on Threading (Ongoing work)

For the current `{I,L,J}` macro-tile ordering, the natural parallelization strategy is over the **`I` dimension**: each thread gets a contiguous block of row-tiles of `A` and the corresponding rows of `C`. That gives each thread exclusive ownership of its output rows, so the computation phase requires **no synchronization**, no atomics, and no reductions. The only shared operand is `B` (or `BT` when used), which is read-only.

Key architectural considerations for my i9-7940X:

- **L2 is per-core.** Each thread's A-tile stays in its own 1 MB L2 with no contention.
- **L3 is shared.** B (or BT) is read by every thread and naturally becomes shared in L3. One core's fetch populates L3 for the others.
- **DRAM bandwidth is shared.** The approximately 80 GB/s is divided across all active cores (approximately 5.7 GB/s each at full occupancy). Per-core tiling keeps this from becoming a bottleneck for well-sized matrices.
- **Thread count = physical cores.** 14 threads (one per physical core) is optimal. All 28 hyperthreads would contend on the same FMA units, L1, and L2 with no additional compute throughput. Hyperthreading helps latency-bound workloads; GEMM is throughput-bound.
- **Transpose once before spawning threads.** BT (when needed) is shared read-only data.

## Next Steps to Close the Gap to BLAS

The remaining gap to production GEMM is no longer about changing the high-level algorithm, but about removing hot-loop overhead and making the fast path more BLAS-like. The upgrades below are cumulative.

### Common to all three kernels

- **Adaptive macro-tiles** for small, medium, and large matrices
- **Thread cutoff** for small problems, to avoid parallel overhead dominating useful work

### Cache-blocked kernel

- Replace the current scalar inner loop with a **pointer-based inner dot product**
- **Unroll the hot `k` loop by 4** to reduce loop overhead and expose more instruction-level parallelism

### Scalar microkernel

- Move to a **true pointer-based microkernel**, passing pointers already offset to the current `A`/`BT`/`C` micro-panels
- Remove as much **global index arithmetic** as possible from the hot loop
- **Unroll the `k` loop by 4** inside the microkernel

### SIMD microkernel

- **Pack the streamed operand**: pack `B` for RowMajor, and symmetrically pack `A` for ColMajor
- Split the kernel into a **full-tile fast path** and a **separate tail path**
- **Unroll the hot `k` loop by 4**
- Keep the implementation in the current **`std::experimental::simd` style**

---

## Acknowledgment

I used LLMs as a brainstorming partner during the writing of this document. The technical analysis, implementation choices, and architectural reasoning are my own; the LLM helped me stress-test my understanding and clean up the text.

---

## References

[1] Intel Corporation. *Intel 64 and IA-32 Architectures Software Developer's Manual, Volume 1: Basic Architecture*. Chapter 14: Programming with AVX-512. Available at https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html

[2] Wikichip. *Skylake (server) - Microarchitectures - Intel*. Section on cache hierarchy and non-inclusive L3. Available at https://en.wikichip.org/wiki/intel/microarchitectures/skylake_(server)

[3] Intel Corporation. *Intel 64 and IA-32 Architectures Optimization Reference Manual*. Chapter 2: Intel 64 and IA-32 Architectures, section on hardware prefetching. Available at https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html

[4] Williams, S., Waterman, A., and Patterson, D. *Roofline: An Insightful Visual Performance Model for Multicore Architectures*. Communications of the ACM, Vol. 52, No. 4, April 2009.

[5] Lemire, D. *AVX-512: when and how to use these new instructions*. Blog post discussing frequency throttling behavior on Skylake-X. Available at https://lemire.me/blog/

[6] Goto, K. and van de Geijn, R. *Anatomy of High-Performance Matrix Multiplication*. ACM Transactions on Mathematical Software, Vol. 34, No. 3, Article 12, May 2008. (The foundational paper for the GotoBLAS approach to tiling and microkernel design.)

[7] Low, T.M., Igual, F.D., Smith, T.M., and Quintana-Orti, E.S. *Analytical Modeling Is Enough for High-Performance BLIS*. ACM Transactions on Mathematical Software, Vol. 43, No. 2, Article 12, August 2016. (Extends the Goto approach with analytical cache modeling for tile size selection.)

[8] Van Zee, F.G. and van de Geijn, R. *BLIS: A Framework for Rapidly Instantiating BLAS Functionality*. ACM Transactions on Mathematical Software, Vol. 41, No. 3, Article 14, June 2015. (The BLIS framework that formalized the five-loop GEMM structure with explicit microkernel.)
