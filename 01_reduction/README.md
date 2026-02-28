# Reduction — Literature (mx_reduction)

Some of the core references used to implement and optimize reduction on CPU and NVIDIA GPUs.

## GPU (CUDA)

- **Mark Harris — Optimizing Parallel Reduction in CUDA**  
  Step-by-step “ladder” of reduction kernels (addressing patterns, bank conflicts, unrolling, warp-level reductions).  
  https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

- **NVIDIA CUB / CCCL — `cub::DeviceReduce` (production baseline + API reference)**  
  Good for benchmarking and for seeing what “best practice” looks like.  
  https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceReduce.html

## CPU

- **C++ Standard Algorithms — `std::reduce` semantics**  
  Important: reduction order may be reassociated/reordered (floating-point non-determinism).  
  https://en.cppreference.com/w/cpp/algorithm/reduce

- **oneTBB — `parallel_reduce` (industrial pattern for parallel reduction)**  
  Split → local reduce → join, with good scaling structure.  
  https://www.intel.com/content/www/us/en/docs/onetbb/developer-guide-api-reference/2022-0/parallel-reduce.html

- **OpenMP — `reduction` clause**  
  Canonical CPU baseline for threaded reduction and privatization.  
  https://www.openmp.org/spec-html/5.2/openmpsu52.html
