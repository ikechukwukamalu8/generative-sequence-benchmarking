# Benchmarking Generative Sequences & Quantized Latent Spaces

An empirical evaluation framework designed to quantify multi-dimensional optimization trade-offs (compute latency, structural memory allocation, and sequence-level epistemic uncertainty boundaries) when deploying open-source autoregressive transformer architectures under varying hardware precision states.

## Research Architecture Overview
When scaling generative model systems along the edge-computing continuum, maintaining local digital sovereignty requires strict adherence to local physical resource boundaries. This research treats architecture selection as a dynamic constraint problem, exploring how structural network compression via uniform post-training quantization techniques (4-bit NF4 and 8-bit integer parameters) modifies internal token representation spaces and downstream predictive shannon entropy.

## Empirical Performance Matrix (NVIDIA T4 Profile)

| Compression Mode | Compute Latency (s) | Generation Velocity (tok/s) | Latent Space Predictive Entropy | Peak VRAM Allocation (MB) |
| :--- | :--- | :--- | :--- | :--- |
| **UNCOMPRESSED** | 2.4122 | 12.44 | 3.5003 | 981.34 |
| **8-bit Integer Quantized** | 16.7744 | 1.79 | 3.6521 | 633.47 |
| **4-bit NormalFloat (NF4)** | 2.3929 | 12.54 | 3.2117 | 483.03 |

### System Telemetry Insights
1. **Infrastructure Minimization:** Converting the weights layout configuration down to 4-bit NF4 yielded a **51% reduction in peak VRAM consumption** (from 981.34 MB down to 483.03 MB), proving viability for restricted compute continuum deployment profiles.
2. **Sequential Runtime Bottlenecks:** Standard 8-bit integer quantization layouts introduced a severe performance overhead penalty, resulting in a latency surge up to **16.7744 seconds**. This is caused by on-the-fly de-quantization cycles executing inside the sequential autoregressive decoding loops on the target hardware layer.

## System Dependency Configuration
To initialize the benchmarking workspace environments on your local device platform, execute:
```bash
pip install -r requirements.txt
python benchmark.py
```
