# CI Workflows

## `ci.yml` — Encode / Decode Regression

Runs on every push and pull request to `main`. Verifies that both the C++ and Python backends can round-trip a point cloud and keep reconstruction error within spec.

### Jobs

| Job | What it does |
|-----|-------------|
| `cpp-backend` | Compiles `src/cpp/lizip.cpp` from source, then encodes and decodes each fixture with the C++ engine |
| `python-backend` | Installs PyTorch (CPU-only) and runs the same encode/decode cycle through the Python engine |

Both jobs run across **5 Python versions** (3.9, 3.10, 3.11, 3.12, 3.13) and **2 fixture files** per version, giving 20 parallel matrix runs per job (40 total).

### Fixtures (`tests/fixtures/`)

| File | Dataset | Points |
|------|---------|--------|
| `sample_nuscenes.bin` | NuScenes | ~34,752 |
| `sample_argo_315978406019574000.bin` | Argoverse | ~90,567 |

Argoverse frames were originally `.ply` and were converted to the 5-float-per-point binary format (`x, y, z, 0, 0`) expected by the C++ engine.

The full fixture set (10 NuScenes + 10 Argoverse) is committed under `tests/fixtures/` but commented out of the matrix. Uncomment to run release validation.

### Pass criteria

After decoding, `src/utils/compare.py` asserts:

- **Mean** nearest-neighbour error ≤ 0.02 mm
- **p99** nearest-neighbour error ≤ 0.05 mm

Actual errors are typically ~0.005 mm mean / ~0.009 mm p99 — the thresholds leave 4× headroom for platform variation.

### C++ compilation flags

```
g++ -O2 -fopenmp -mavx2 -mfma -o src/cpp/lizip.exe src/cpp/lizip.cpp -lz -llzma
```

- `-fopenmp` — parallel block processing
- `-mavx2 -mfma` — AVX2 + fused multiply-add SIMD (supported on GitHub Actions ubuntu-latest runners)
- `-lz -llzma` — zlib and LZMA entropy compression

### Scaling up

When ready to run the full suite, uncomment the remaining fixtures in `ci.yml`. The matrix expands to 5 × 20 × 2 = 200 jobs.
