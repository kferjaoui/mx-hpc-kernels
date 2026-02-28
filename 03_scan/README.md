# Scan (Prefix Sum) — Literature (mx_scan)

List of the references used to implement inclusive/exclusive scans on CPU and NVIDIA GPUs.

## Theory / Foundations

- **Guy E. Blelloch — Prefix Sums and Their Applications**  
  Classic work-efficient scan (upsweep/downsweep), “prescan” and terminology.  
  https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf

## GPU (CUDA) — Implemented (Hillis–Steele, Blelloch)

- **Mark Harris (GPU Gems 3) — Parallel Prefix Sum (Scan) with CUDA**  
  Practical CUDA implementation guide and background.  
  https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda

- **NVIDIA CUB / CCCL — `cub::DeviceScan` (production baseline + API reference)**  
  Useful as a performance baseline and to understand high-performance scan interfaces.  
  https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceScan.html

- **CUDA Samples — `shfl_scan` (warp shuffle scan idioms)**  
  Minimal example of warp-level scan via shuffle intrinsics.  
  https://github.com/NVIDIA/cuda-samples/tree/master/Samples/2_Concepts_and_Techniques/shfl_scan

## GPU (CUDA) — Advanced / Not Implemented (yet)

- **Merrill & Garland — Single-pass Parallel Prefix Scan with Decoupled Look-back**  
  Modern scalable device-wide scan technique used by high-performance libraries (alternative to “scan block sums + uniform add”).  
  https://research.nvidia.com/sites/default/files/pubs/2016-03_Single-pass-Parallel-Prefix/nvr-2016-002.pdf

## CPU

- **C++ Standard Algorithms — `std::inclusive_scan` / `std::exclusive_scan`**  
  Useful for defining semantics and as a correctness reference.  
  https://en.cppreference.com/w/cpp/algorithm/inclusive_scan.html  
  https://en.cppreference.com/w/cpp/algorithm/exclusive_scan.html

- **oneTBB — `parallel_scan` (production CPU parallel scan structure)**  
  Two-pass style structure (local scan + carry propagation) and robust scaling model.  
  https://oneapi-spec.uxlfoundation.org/specifications/oneapi/v1.1-rev-1/elements/onetbb/source/algorithms/functions/parallel_scan_func
