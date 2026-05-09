//! Pure-DSP SWIPE'/SWIPE pitch estimator in Rust.
//!
//! Implements the multi-resolution mel-axis SWIPE' described in
//! Marttila & Reiss, *"Improving Neural Pitch Estimation with SWIPE Kernels"*
//! (ISMIR 2025, [arXiv:2507.11233](https://arxiv.org/abs/2507.11233)),
//! built on top of the original Camacho & Harris (2008) formulation.
//!
//! This is an independent reimplementation in Rust from the published
//! descriptions; no source from any reference implementation was copied.
//!
//! # Pipeline (per frame)
//! 1. For each window length `W` in the bank: take the most recent `W`
//!    samples, apply a Hann window, zero-pad to `FFT_SIZE`, compute the
//!    magnitude spectrum, linearly interpolate it onto the 1024-point
//!    mel axis.
//! 2. Compute the SWIPE score for each of 295 pitch candidates against
//!    each window's mel-magnitude:
//!        `Z_W[c] = (kernel[c] · sqrt(mag_mel)) / sqrt(sum(mag_mel))`
//! 3. For each candidate `f_c`, blend the two adjacent windows' scores
//!    by linear interpolation in `log2(W)` space to approximate the
//!    ideal `W_c = 8 fs / f_c`.
//! 4. argmax + parabolic refinement on the 36-bins-per-octave log axis.
//!
//! # Streaming
//! Each emitted frame is **right-aligned**: it consumes the latest
//! `W_max` samples of the rolling buffer. The estimator emits one frame
//! per `HOP_SAMPLES = 480` samples (10 ms at 48 kHz) of new audio.
//!
//! # Quick example
//! ```no_run
//! use swipe_rs::Swipe;
//!
//! let mut swipe = Swipe::new().unwrap();
//! // feed 48 kHz mono f32 audio in any-size chunks; emitted frames
//! // accumulate as you process more input.
//! let frames = swipe.process(&[0.0_f32; 48_000]).unwrap();
//! for f in frames {
//!     if f.confidence > 0.3 {
//!         println!("{:6.3} s — {:6.1} Hz (conf {:.2})",
//!                  f.time_s, f.pitch_hz, f.confidence);
//!     }
//! }
//! ```

use realfft::num_complex::Complex32;
use realfft::{RealFftPlanner, RealToComplex};
use std::sync::Arc;

/// SWIPE always operates at 48 kHz internally. Resample your input to
/// this rate before calling [`Swipe::process`].
pub const SAMPLE_RATE: u32 = 48_000;

/// Smallest analysis window we ever use. With `max_window = 2048` the bank
/// degenerates to a single window; below 2 entries the multi-resolution
/// blend has nothing to interpolate between. `MIN_WINDOW = 1024` keeps a
/// useful spread even at the most aggressive Realtime preset.
pub const MIN_WINDOW: usize = 1024;
/// Default longest window. At 48 kHz it covers down to ~47 Hz (E1) with
/// no measurable accuracy loss vs the full 16 384 (paper Table 4).
pub const DEFAULT_MAX_WINDOW: usize = 8192;
/// 295 candidates @ 36 bins/octave from 27.5 Hz to ~8055 Hz.
const NUM_CANDIDATES: usize = 295;
const F_MIN_HZ: f32 = 27.5;
/// 1024 mel-spaced sampling frequencies, range `[0.25*f_min, 1.25*f_max]`.
const MEL_FREQS: usize = 1024;
const MEL_FMIN_HZ: f32 = 0.25 * F_MIN_HZ;
const MEL_FMAX_HZ: f32 = 1.25 * 8055.0;
/// 10 ms hop at 48 kHz.
pub const HOP_SAMPLES: usize = 480;

/// One emitted pitch frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PitchFrame {
    /// Frame index since the last [`Swipe::reset`] (or stream start).
    pub frame_index: u64,
    /// Wall time of this frame in seconds (`frame_index * HOP_SAMPLES / SAMPLE_RATE`).
    pub time_s: f32,
    /// Estimated fundamental frequency in Hz.
    pub pitch_hz: f32,
    /// Peak SWIPE score in `[0, 1]`. Treat ≥ 0.3 as "voiced" on real
    /// vocal; pure DSP noise floor is around 0.1-0.2.
    pub confidence: f32,
}

/// Errors from the estimator.
#[derive(Debug)]
pub enum Error {
    /// realfft (or our wiring around it) failed.
    Fft(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Fft(s) => write!(f, "fft error: {s}"),
        }
    }
}

impl std::error::Error for Error {}

/// Slaney-style HTK mel: linear up to 1 kHz, log above. Matches
/// `librosa.mel_frequencies(htk=False)`.
fn hz_to_mel(hz: f32) -> f32 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let log_step = (6.4_f32).ln() / 27.0;
    if hz < min_log_hz {
        hz / f_sp
    } else {
        min_log_mel + (hz / min_log_hz).ln() / log_step
    }
}
fn mel_to_hz(mel: f32) -> f32 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let log_step = (6.4_f32).ln() / 27.0;
    if mel < min_log_mel {
        f_sp * mel
    } else {
        min_log_hz * ((mel - min_log_mel) * log_step).exp()
    }
}

fn build_mel_axis() -> Vec<f32> {
    let m_lo = hz_to_mel(MEL_FMIN_HZ);
    let m_hi = hz_to_mel(MEL_FMAX_HZ);
    let step = (m_hi - m_lo) / (MEL_FREQS as f32 - 1.0);
    (0..MEL_FREQS)
        .map(|i| mel_to_hz(m_lo + i as f32 * step))
        .collect()
}

fn build_candidates() -> Vec<f32> {
    (0..NUM_CANDIDATES)
        .map(|c| F_MIN_HZ * 2.0_f32.powf(c as f32 / 36.0))
        .collect()
}

fn is_swipe_prime(h: usize) -> bool {
    if h == 1 {
        return true;
    }
    if h < 2 {
        return false;
    }
    if h % 2 == 0 {
        return h == 2;
    }
    let mut d = 3usize;
    while d * d <= h {
        if h % d == 0 {
            return false;
        }
        d += 2;
    }
    true
}

fn build_kernels(candidates: &[f32], mel_freqs: &[f32]) -> Vec<f32> {
    let nf = mel_freqs.len();
    let mut kernel = vec![0.0_f32; candidates.len() * nf];

    for (row, &fc) in candidates.iter().enumerate() {
        let half_lobe = fc / 2.0;
        let row_off = row * nf;

        let mut h: usize = 1;
        let mut last_kept_h: usize = 0;
        loop {
            let center = h as f32 * fc;
            if center > MEL_FMAX_HZ {
                break;
            }
            if is_swipe_prime(h) {
                let weight = 1.0 / (h as f32).sqrt();
                let lo = center - half_lobe;
                let hi = center + half_lobe;
                for (j, &mf) in mel_freqs.iter().enumerate() {
                    if mf >= lo && mf <= hi {
                        let lobe =
                            0.5 * (1.0 + (2.0 * std::f32::consts::PI * (mf - center) / fc).cos());
                        kernel[row_off + j] += weight * lobe;
                    }
                }
                if last_kept_h > 0 {
                    let mid = (last_kept_h as f32 + h as f32) * 0.5 * fc;
                    let lo_v = mid - half_lobe;
                    let hi_v = mid + half_lobe;
                    let v_amp = -weight * 0.5;
                    for (j, &mf) in mel_freqs.iter().enumerate() {
                        if mf >= lo_v && mf <= hi_v {
                            let lobe = 0.5
                                * (1.0
                                    + (2.0 * std::f32::consts::PI * (mf - mid) / fc).cos());
                            kernel[row_off + j] += v_amp * lobe;
                        }
                    }
                }
                last_kept_h = h;
            }
            h += 1;
        }

        let mean = kernel[row_off..row_off + nf].iter().sum::<f32>() / nf as f32;
        let mut sumsq = 0.0_f32;
        for v in kernel[row_off..row_off + nf].iter_mut() {
            *v -= mean;
            sumsq += *v * *v;
        }
        let norm = sumsq.sqrt();
        if norm > 1e-12 {
            for v in kernel[row_off..row_off + nf].iter_mut() {
                *v /= norm;
            }
        }
    }
    kernel
}

fn windows_for_max(max_window: usize) -> Vec<usize> {
    let mut out = Vec::new();
    let mut w = MIN_WINDOW;
    while w <= max_window {
        out.push(w);
        w *= 2;
    }
    if out.is_empty() {
        out.push(MIN_WINDOW);
    }
    out
}

fn build_window_interp(
    candidates: &[f32],
    windows: &[usize],
) -> (Vec<usize>, Vec<usize>, Vec<f32>) {
    let log_windows: Vec<f32> = windows.iter().map(|&w| (w as f32).log2()).collect();
    let mut idx_minus = Vec::with_capacity(candidates.len());
    let mut idx_plus = Vec::with_capacity(candidates.len());
    let mut alpha = Vec::with_capacity(candidates.len());
    for &fc in candidates {
        let wc = 8.0 * SAMPLE_RATE as f32 / fc;
        let l = wc.log2();
        let mut hi = log_windows
            .iter()
            .position(|&lw| lw >= l)
            .unwrap_or(log_windows.len() - 1);
        if hi == 0 {
            hi = 1;
        }
        let lo = hi - 1;
        let denom = (log_windows[hi] - log_windows[lo]).max(1e-6);
        let a = ((l - log_windows[lo]) / denom).clamp(0.0, 1.0);
        idx_minus.push(lo);
        idx_plus.push(hi);
        alpha.push(a);
    }
    (idx_minus, idx_plus, alpha)
}

fn build_mel_interp(mel_freqs: &[f32], fft_size: usize) -> (Vec<usize>, Vec<f32>) {
    let bin_hz = SAMPLE_RATE as f32 / fft_size as f32;
    let n_bins = fft_size / 2 + 1;
    let mut lo = Vec::with_capacity(mel_freqs.len());
    let mut alpha = Vec::with_capacity(mel_freqs.len());
    for &mf in mel_freqs {
        let pos = mf / bin_hz;
        let i = pos.floor() as isize;
        let i = i.clamp(0, n_bins as isize - 2) as usize;
        let a = (pos - i as f32).clamp(0.0, 1.0);
        lo.push(i);
        alpha.push(a);
    }
    (lo, alpha)
}

/// Streaming SWIPE pitch estimator.
///
/// Operates at [`SAMPLE_RATE`] (48 kHz) on mono `f32` audio. Use
/// [`Swipe::new`] for the default Balanced preset (max window 8192 samples)
/// or [`Swipe::with_max_window`] to trade latency for low-pitch coverage.
pub struct Swipe {
    candidates: Vec<f32>,
    /// (NUM_CANDIDATES * MEL_FREQS) row-major kernel matrix.
    kernel: Vec<f32>,
    cand_idx_minus: Vec<usize>,
    cand_idx_plus: Vec<usize>,
    cand_alpha: Vec<f32>,
    mel_lo: Vec<usize>,
    mel_alpha: Vec<f32>,
    /// Pre-computed Hann windows, one per entry in `windows`.
    hann: Vec<Vec<f32>>,
    /// Active analysis windows in samples (powers of 2 ascending).
    windows: Vec<usize>,
    /// FFT size = largest window. Sets DFT bin granularity.
    fft_size: usize,
    fft: Arc<dyn RealToComplex<f32> + Send + Sync>,
    fft_input: Vec<f32>,
    fft_output: Vec<Complex32>,

    buffer: Vec<f32>,
    next_frame_index: u64,
    /// Sample index of `buffer[0]` in the absolute estimator stream.
    buffer_origin: u64,
    /// Sample index where the next frame's right edge sits.
    next_frame_end: u64,
}

impl Swipe {
    /// Construct with a custom `max_window`. The window bank is every
    /// power of two from [`MIN_WINDOW`] up to `max_window` (rounded up
    /// to the next power of two) inclusive; the FFT size equals the
    /// largest window. Smaller `max_window` = lower latency and less
    /// compute, but the lowest detectable pitch rises to roughly
    /// `8 * SAMPLE_RATE / max_window`.
    pub fn with_max_window(max_window: usize) -> Result<Self, Error> {
        let mw = max_window.next_power_of_two();
        let mw = mw.max(MIN_WINDOW);
        let windows = windows_for_max(mw);
        let fft_size = *windows.last().unwrap();

        let candidates = build_candidates();
        let mel_freqs = build_mel_axis();
        let kernel = build_kernels(&candidates, &mel_freqs);
        let (cand_idx_minus, cand_idx_plus, cand_alpha) =
            build_window_interp(&candidates, &windows);
        let (mel_lo, mel_alpha) = build_mel_interp(&mel_freqs, fft_size);
        let hann: Vec<Vec<f32>> = windows
            .iter()
            .map(|&w| {
                (0..w)
                    .map(|i| {
                        0.5 - 0.5
                            * (2.0 * std::f32::consts::PI * i as f32 / (w as f32 - 1.0)).cos()
                    })
                    .collect()
            })
            .collect();
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(fft_size);
        let fft_input = vec![0.0_f32; fft_size];
        let fft_output = vec![Complex32::new(0.0, 0.0); fft_size / 2 + 1];
        Ok(Self {
            candidates,
            kernel,
            cand_idx_minus,
            cand_idx_plus,
            cand_alpha,
            mel_lo,
            mel_alpha,
            hann,
            windows,
            fft_size,
            fft,
            fft_input,
            fft_output,
            buffer: Vec::with_capacity(fft_size * 2),
            next_frame_index: 0,
            buffer_origin: 0,
            next_frame_end: fft_size as u64,
        })
    }

    /// Default: `max_window = DEFAULT_MAX_WINDOW`. Balanced preset that
    /// covers the human voice range with no measurable accuracy loss vs
    /// the 16 384 full-range setting (see paper Table 4).
    pub fn new() -> Result<Self, Error> {
        Self::with_max_window(DEFAULT_MAX_WINDOW)
    }

    /// Reset the streaming state: clears the buffer and frame counter.
    /// The kernel matrix and FFT planner are kept (no reallocation).
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.next_frame_index = 0;
        self.buffer_origin = 0;
        self.next_frame_end = self.fft_size as u64;
    }

    /// Feed audio. Returns every frame whose right edge has now landed
    /// inside the current buffer. May return zero frames if the input
    /// chunk was small. Drives one frame per [`HOP_SAMPLES`] of new audio.
    ///
    /// Allocates a fresh `Vec<PitchFrame>` on every call. Realtime callers
    /// that want to reuse a buffer should use [`Self::process_into`].
    pub fn process(&mut self, audio: &[f32]) -> Result<Vec<PitchFrame>, Error> {
        let mut out = Vec::new();
        self.process_into(audio, &mut out)?;
        Ok(out)
    }

    /// Same as [`Self::process`] but appends frames to a caller-provided
    /// buffer instead of allocating a new one.
    ///
    /// Frames are **appended** — `out` is *not* cleared first. This lets
    /// realtime callers keep a single long-lived `Vec` across thousands
    /// of `process_into` calls without ever hitting the allocator.
    /// `out.clear()` it yourself before the call if you only want the
    /// frames produced by this chunk.
    ///
    /// `out`'s capacity is grown in amortised-`O(1)` fashion (Rust's
    /// `Vec` doubling strategy), so after the first second of audio it
    /// will essentially never reallocate again.
    pub fn process_into(
        &mut self,
        audio: &[f32],
        out: &mut Vec<PitchFrame>,
    ) -> Result<(), Error> {
        self.buffer.extend_from_slice(audio);
        let hop_s = HOP_SAMPLES as f32 / SAMPLE_RATE as f32;

        loop {
            let buf_end = self.buffer_origin + self.buffer.len() as u64;
            if self.next_frame_end > buf_end {
                break;
            }
            let local_end = (self.next_frame_end - self.buffer_origin) as usize;
            let (conf, f0) = self.one_frame(local_end)?;
            let abs_idx = self.next_frame_index;
            out.push(PitchFrame {
                frame_index: abs_idx,
                time_s: abs_idx as f32 * hop_s,
                pitch_hz: f0,
                confidence: conf,
            });
            self.next_frame_index += 1;
            self.next_frame_end += HOP_SAMPLES as u64;
        }

        let need = self.fft_size as u64;
        if self.next_frame_end > need {
            let must_keep_from = self.next_frame_end - need;
            if must_keep_from > self.buffer_origin {
                let drop_n = (must_keep_from - self.buffer_origin) as usize;
                let drop_n = drop_n.min(self.buffer.len());
                self.buffer.drain(..drop_n);
                self.buffer_origin += drop_n as u64;
            }
        }
        Ok(())
    }

    fn mel_mag(&mut self, audio_end: usize, w_len: usize) -> Result<Vec<f32>, Error> {
        let buf = &self.buffer;
        let start_signed = audio_end as isize - w_len as isize;
        let hann_idx = self.windows.iter().position(|&w| w == w_len).unwrap();
        let hann = &self.hann[hann_idx];

        for v in self.fft_input.iter_mut() {
            *v = 0.0;
        }
        for i in 0..w_len {
            let bi = start_signed + i as isize;
            if bi >= 0 && (bi as usize) < buf.len() {
                self.fft_input[i] = buf[bi as usize] * hann[i];
            }
        }
        self.fft
            .process(&mut self.fft_input, &mut self.fft_output)
            .map_err(|e| Error::Fft(e.to_string()))?;

        let n_bins = self.fft_size / 2 + 1;
        let mut mag = vec![0.0_f32; n_bins];
        for (i, c) in self.fft_output.iter().enumerate() {
            mag[i] = (c.re * c.re + c.im * c.im).sqrt();
        }
        let mut mag_mel = vec![0.0_f32; MEL_FREQS];
        for j in 0..MEL_FREQS {
            let i = self.mel_lo[j];
            let a = self.mel_alpha[j];
            mag_mel[j] = mag[i] * (1.0 - a) + mag[i + 1] * a;
        }
        Ok(mag_mel)
    }

    fn score_window(&self, mag_mel: &[f32]) -> Vec<f32> {
        let mut sqrt_mag = vec![0.0_f32; MEL_FREQS];
        let mut mag_sum = 0.0_f32;
        for (i, &m) in mag_mel.iter().enumerate() {
            sqrt_mag[i] = m.sqrt();
            mag_sum += m;
        }
        let denom = mag_sum.sqrt().max(1e-12);
        let mut z = vec![0.0_f32; NUM_CANDIDATES];
        for c in 0..NUM_CANDIDATES {
            let row = &self.kernel[c * MEL_FREQS..(c + 1) * MEL_FREQS];
            let mut acc = 0.0_f32;
            for k in 0..MEL_FREQS {
                acc += row[k] * sqrt_mag[k];
            }
            z[c] = acc / denom;
        }
        z
    }

    fn one_frame(&mut self, audio_end: usize) -> Result<(f32, f32), Error> {
        let mut z_per_win: Vec<Vec<f32>> = Vec::with_capacity(self.windows.len());
        let windows = self.windows.clone();
        for w in windows {
            let mag = self.mel_mag(audio_end, w)?;
            z_per_win.push(self.score_window(&mag));
        }
        let mut z = vec![0.0_f32; NUM_CANDIDATES];
        for c in 0..NUM_CANDIDATES {
            let lo = self.cand_idx_minus[c];
            let hi = self.cand_idx_plus[c];
            let a = self.cand_alpha[c];
            z[c] = (1.0 - a) * z_per_win[lo][c] + a * z_per_win[hi][c];
        }
        let mut peak_c = 0usize;
        let mut peak_v = f32::NEG_INFINITY;
        for (c, &v) in z.iter().enumerate() {
            if v > peak_v {
                peak_v = v;
                peak_c = c;
            }
        }
        let f0 = if peak_c > 0 && peak_c + 1 < NUM_CANDIDATES {
            let y_m = z[peak_c - 1];
            let y_0 = z[peak_c];
            let y_p = z[peak_c + 1];
            let denom = y_m - 2.0 * y_0 + y_p;
            let shift = if denom.abs() > 1e-12 {
                0.5 * (y_m - y_p) / denom
            } else {
                0.0_f32
            }
            .clamp(-0.5, 0.5);
            self.candidates[peak_c] * 2.0_f32.powf(shift / 36.0)
        } else {
            self.candidates[peak_c]
        };
        Ok((peak_v.clamp(0.0, 1.0), f0))
    }
}
