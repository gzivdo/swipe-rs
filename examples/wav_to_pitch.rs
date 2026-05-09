//! Minimal example: load a mono WAV, run SWIPE, dump frames to a text file.
//!
//! Usage:
//!     cargo run --release --example wav_to_pitch -- input.wav output.txt
//!
//! Output format (one frame per line, voiced frames only):
//!     <time_s>  <pitch_hz>  <confidence>
//!
//! Input must be mono (or first channel will be used). Resamples to 48 kHz
//! if needed using a basic linear interpolation.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use swipe_rs::{Swipe, SAMPLE_RATE};

const VOICED_THRESHOLD: f32 = 0.3;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let in_path: PathBuf = args
        .next()
        .ok_or("usage: wav_to_pitch <input.wav> <output.txt>")?
        .into();
    let out_path: PathBuf = args
        .next()
        .ok_or("usage: wav_to_pitch <input.wav> <output.txt>")?
        .into();

    // Load WAV (any sample format hound supports). Take channel 0.
    let mut reader = hound::WavReader::open(&in_path)?;
    let spec = reader.spec();
    let src_sr = spec.sample_rate;
    let channels = spec.channels as usize;

    let raw: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().filter_map(Result::ok).collect(),
        hound::SampleFormat::Int => {
            let max = (1u64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .filter_map(Result::ok)
                .map(|s| s as f32 / max)
                .collect()
        }
    };
    let mono: Vec<f32> = if channels == 1 {
        raw
    } else {
        raw.chunks_exact(channels)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect()
    };

    let mono_len = mono.len();
    let audio_48k: Vec<f32> = if src_sr == SAMPLE_RATE {
        mono
    } else {
        resample_linear(&mono, src_sr, SAMPLE_RATE)
    };

    eprintln!(
        "[swipe-rs example] loaded {} ({:.1}s @ {} Hz, {} ch) → {} samples @ {} Hz",
        in_path.display(),
        mono_len as f32 / src_sr as f32,
        src_sr,
        channels,
        audio_48k.len(),
        SAMPLE_RATE,
    );

    let mut swipe = Swipe::new()?;
    let frames = swipe.process(&audio_48k)?;

    let f = File::create(&out_path)?;
    let mut w = BufWriter::new(f);
    writeln!(w, "# time_s  pitch_hz  confidence")?;
    let mut voiced_count = 0;
    for fr in &frames {
        if fr.confidence >= VOICED_THRESHOLD {
            voiced_count += 1;
            writeln!(w, "{:.4}\t{:.3}\t{:.3}", fr.time_s, fr.pitch_hz, fr.confidence)?;
        }
    }
    w.flush()?;

    eprintln!(
        "[swipe-rs example] wrote {} ({} voiced of {} frames, threshold {:.2})",
        out_path.display(),
        voiced_count,
        frames.len(),
        VOICED_THRESHOLD,
    );

    Ok(())
}

/// Linear-interpolation resampler — minimal, dependency-free. Good enough
/// for an example; real apps should use rubato or libsamplerate.
fn resample_linear(audio: &[f32], src_sr: u32, dst_sr: u32) -> Vec<f32> {
    if src_sr == dst_sr {
        return audio.to_vec();
    }
    let ratio = src_sr as f64 / dst_sr as f64;
    let n_out = (audio.len() as f64 / ratio).floor() as usize;
    let mut out = Vec::with_capacity(n_out);
    for i in 0..n_out {
        let pos = i as f64 * ratio;
        let lo = pos.floor() as usize;
        let hi = (lo + 1).min(audio.len() - 1);
        let frac = (pos - lo as f64) as f32;
        out.push(audio[lo] * (1.0 - frac) + audio[hi] * frac);
    }
    out
}
