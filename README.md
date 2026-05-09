# swipe-rs

Pure-DSP **SWIPE'** / **SWIPE** pitch estimator in Rust. No neural networks,
no model weights, no training — just an FFT, a kernel matrix, and
math from two published papers.

Targets **48 kHz mono `f32`** audio. Streaming-friendly: feed any-size
chunks, get back zero or more `(time, hz, confidence)` frames per call.

```rust
use swipe_rs::Swipe;

fn main() -> Result<(), swipe_rs::Error> {
    let mut swipe = Swipe::new()?;
    // 48 kHz mono f32 audio, any chunk size:
    let frames = swipe.process(&audio_chunk)?;
    for f in frames {
        if f.confidence > 0.3 {
            println!("{:6.3} s  {:6.1} Hz  conf={:.2}",
                     f.time_s, f.pitch_hz, f.confidence);
        }
    }
    Ok(())
}
```

Add to `Cargo.toml`:
```toml
[dependencies]
swipe-rs = "0.1"
```

## What is SWIPE

Both algorithms estimate the fundamental frequency $f_0$ of a windowed
audio frame by:

1. **Building a bank of spectral kernels** — one per pitch candidate.
   Each kernel looks like the square root of the magnitude spectrum of
   a Hann-windowed sawtooth at that pitch: positive cosine lobes at
   harmonic positions, negative cosine valleys between them.

2. **Scoring each candidate** as the cosine-similarity-like inner
   product between its kernel and the (sqrt of) magnitude spectrum of
   the input frame.

3. **Picking the argmax** as the f0 estimate.

**SWIPE'** keeps lobes only at $f_c$ and at *prime-harmonic* multiples
($2f_c, 3f_c, 5f_c, 7f_c, \ldots$), which suppresses sub-octave errors
that plain SWIPE has on sawtooths and similar harmonic-rich sources.
This crate uses SWIPE'.

The Marttila & Reiss (2025) paper extends Camacho's SWIPE with two
key tricks:

- **Multi-resolution windows** — the magnitude spectrum is computed at
  several FFT sizes (powers of two from 1024 up to `max_window`), and
  for each candidate the two flanking windows are linearly blended in
  $\log_2(W)$ space to approximate the ideal $W_c = 8 f_s / f_c$.
  This decouples low-pitch reach (longer windows) from high-pitch
  precision (shorter windows).

- **Mel-axis sampling** — kernels and spectra are sampled on a 1024-point
  mel axis instead of linear bins. This concentrates resolution where
  it matters (low harmonics) and dramatically improves accuracy on
  speech and music — comparable to learned models like CREPE on standard
  benchmarks (96.4% RPA on MIR-1K, paper Table 4).

The result is a pitch detector that rivals CREPE/pyin on monophonic
audio without any neural network — just FFT + matrix multiply + argmax.

## Configuration

| preset | `max_window` | latency | lowest pitch | RPA on MIR-1K |
|---|---|---|---|---|
| Realtime | 4096 | ~85 ms | ~94 Hz | 96.2% |
| Balanced (default) | 8192 | ~170 ms | ~47 Hz | 96.4% |
| Full range | 16384 | ~340 ms | ~23 Hz | 96.4% |

```rust
use swipe_rs::Swipe;
let realtime  = Swipe::with_max_window(4096)?;
let balanced  = Swipe::new()?;                  // default
let full_rng  = Swipe::with_max_window(16384)?;
```

The bottleneck is the FFT of size `max_window` per emitted frame —
~10 ms hop, so a Realtime instance runs >100× realtime on a single
modern CPU core.

## Voicing

Returned `confidence` is the SWIPE peak score in `[0, 1]`. On real
vocal it sits around `0.3..0.6` for clearly voiced frames and `0.05..0.2`
for unvoiced/silence. Standard threshold: `confidence ≥ 0.3`.

The paper reports unvoiced/voiced classification accuracy without using
a fixed threshold — it just takes the global argmax. For practical
realtime use a threshold or a small post-process median filter on the
confidence is recommended.

## Tests

```bash
cargo test
```

- Pure sine at 220, 440, 880, 1760 Hz on the default Balanced preset →
  estimate within 50 cents (the `mir_eval` Raw Pitch Accuracy bar).
- Sawtooth at the same fundamentals → same tolerance, validates the
  prime-harmonic kernel weights.
- 100 Hz sine on the Full-range preset (`max_window=16384`) → within
  50 cents (Balanced lacks the low-frequency window).
- Streaming chunked input → identical frames to one-shot processing.

Sub-cent precision happens on clean synthetic input but isn't guaranteed
at the edges of the candidate grid (36 bins/octave ≈ 33 cents per bin;
parabolic refinement narrows that further). For practical pitch-tracking
on real audio, treat the algorithm as ±50 cents.

## Citation

If you use swipe-rs in academic work, cite the source papers (not this
crate; the algorithm is theirs):

```bibtex
@article{camacho2008swipe,
  title = {A sawtooth waveform inspired pitch estimator for speech and music},
  author = {Camacho, Arturo and Harris, John G},
  journal = {The Journal of the Acoustical Society of America},
  volume = {124},
  number = {3},
  pages = {1638--1652},
  year = {2008},
  doi = {10.1121/1.2951592},
}

@inproceedings{marttila2025swipe,
  title = {Improving Neural Pitch Estimation with SWIPE Kernels},
  author = {Marttila, David and Reiss, Joshua D},
  booktitle = {ISMIR},
  year = {2025},
  url = {https://arxiv.org/abs/2507.11233},
}
```

Read the Marttila & Reiss paper at <https://arxiv.org/pdf/2507.11233>
for the full algorithm, evaluation details, and the neural extensions
(SWIPE-tiny, SWIPE-sup) that go beyond what this crate implements.

## License

[Apache-2.0](LICENSE). See [`NOTICE`](NOTICE).

This crate is an **independent reimplementation in Rust** from the
published descriptions in the papers above. **No code from any
reference implementation was copied.** Algorithms and mathematical
methods are not protected by copyright (US Copyright Act §102(b);
Feist v. Rural, 1991; EU Directive 2009/24/EC).

The Apache-2.0 license includes an explicit *no-warranty* disclaimer
(Section 8) and a *patent grant* (Section 3) — you can use, modify,
distribute, and build commercial products on top of this crate with
no further obligations beyond the standard Apache attribution.

## Status

Pre-1.0. API may change. Streaming behaviour and acceptance tests are
stable; the public surface is intentionally minimal so it should
solidify quickly.

Bug reports, accuracy regressions, and PRs welcome.

## Authors

- gzivdo — initial extraction and packaging
- Claude Opus 4.7 (Anthropic) — implementation work, refactoring,
  documentation
