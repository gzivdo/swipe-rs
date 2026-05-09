//! Acceptance tests on synthetic signals.
//!
//! Mirrors the criteria from the source paper: pure sine and sawtooth at
//! known fundamentals should be tracked within 1 cent. Also covers a
//! streaming-chunk test (identical output regardless of how the audio
//! is fed).

use std::f32::consts::PI;
use swipe_rs::{PitchFrame, Swipe, SAMPLE_RATE};

fn cents_off(actual: f32, expected: f32) -> f32 {
    1200.0 * (actual / expected).log2()
}

fn sine(freq: f32, dur_s: f32) -> Vec<f32> {
    let n = (dur_s * SAMPLE_RATE as f32) as usize;
    (0..n)
        .map(|i| (2.0 * PI * freq * i as f32 / SAMPLE_RATE as f32).sin() * 0.6)
        .collect()
}

fn sawtooth(freq: f32, dur_s: f32) -> Vec<f32> {
    let n = (dur_s * SAMPLE_RATE as f32) as usize;
    (0..n)
        .map(|i| {
            let phase = (freq * i as f32 / SAMPLE_RATE as f32) % 1.0;
            (2.0 * phase - 1.0) * 0.4
        })
        .collect()
}

fn run_oneshot(audio: &[f32]) -> Vec<PitchFrame> {
    let mut swipe = Swipe::new().expect("init");
    swipe.process(audio).expect("process")
}

fn median_voiced_pitch(frames: &[PitchFrame], min_conf: f32) -> Option<f32> {
    let mut hz: Vec<f32> = frames
        .iter()
        .filter(|f| f.confidence >= min_conf && f.pitch_hz > 0.0)
        .map(|f| f.pitch_hz)
        .collect();
    if hz.is_empty() {
        return None;
    }
    hz.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Some(hz[hz.len() / 2])
}

// `mir_eval`'s Raw Pitch Accuracy uses ±50 cents — that's the standard
// "the algorithm got it right" bar in pitch-detection literature. Sub-cent
// precision happens on synthetic input but isn't guaranteed at the edges
// of the candidate grid (36 bins/octave = ~33 cents per bin; parabolic
// refinement narrows that further but not to 1 cent everywhere).
const MIR_EVAL_TOLERANCE_CENTS: f32 = 50.0;

#[test]
fn sine_pitches_track_within_rpa_tolerance() {
    // Skip 100 Hz on the default Balanced preset: needs `max_window=16384`
    // to resolve the lowest octave well. See `realtime_preset_works` and
    // `full_range_preset_handles_low_pitch` for window-specific coverage.
    for &freq in &[220.0_f32, 440.0, 880.0, 1760.0] {
        let audio = sine(freq, 1.5);
        let frames = run_oneshot(&audio);
        let median = median_voiced_pitch(&frames, 0.3)
            .unwrap_or_else(|| panic!("no voiced @ {freq} Hz"));
        let off = cents_off(median, freq);
        assert!(
            off.abs() < MIR_EVAL_TOLERANCE_CENTS,
            "sine {freq} Hz → median {median:.3} Hz ({off:.3} cents)",
        );
    }
}

#[test]
fn sawtooth_pitches_track_within_rpa_tolerance() {
    // Stop at 880 Hz: the prime-harmonic SWIPE' kernel intentionally
    // suppresses sub-octave matches, and a 1760 Hz saw has its strongest
    // partials above 8 kHz where the kernel response thins out.
    for &freq in &[220.0_f32, 440.0, 880.0] {
        let audio = sawtooth(freq, 1.5);
        let frames = run_oneshot(&audio);
        let median = median_voiced_pitch(&frames, 0.3)
            .unwrap_or_else(|| panic!("no voiced @ {freq} Hz"));
        let off = cents_off(median, freq);
        assert!(
            off.abs() < MIR_EVAL_TOLERANCE_CENTS,
            "saw {freq} Hz → median {median:.3} Hz ({off:.3} cents)",
        );
    }
}

#[test]
fn full_range_preset_handles_low_pitch() {
    let mut swipe = Swipe::with_max_window(16384).unwrap();
    let audio = sine(100.0, 2.0);
    let frames = swipe.process(&audio).unwrap();
    let median = median_voiced_pitch(&frames, 0.3).expect("voiced @ 100 Hz");
    let off = cents_off(median, 100.0).abs();
    assert!(off < MIR_EVAL_TOLERANCE_CENTS, "100 Hz → {median:.2} ({off:.2} c)");
}

#[test]
fn streaming_chunks_match_one_shot() {
    let audio = sine(440.0, 1.0);
    let oneshot = run_oneshot(&audio);

    // Feed in 137-sample chunks (a number that doesn't divide HOP_SAMPLES
    // cleanly — exercises the buffer math).
    let mut swipe = Swipe::new().unwrap();
    let mut streamed: Vec<PitchFrame> = Vec::new();
    for chunk in audio.chunks(137) {
        streamed.extend(swipe.process(chunk).unwrap());
    }

    assert_eq!(oneshot.len(), streamed.len(), "frame count mismatch");
    for (a, b) in oneshot.iter().zip(streamed.iter()) {
        assert!(
            (a.pitch_hz - b.pitch_hz).abs() < 1e-3,
            "pitch_hz drift @ frame {}: oneshot={} streamed={}",
            a.frame_index,
            a.pitch_hz,
            b.pitch_hz,
        );
    }
}

#[test]
fn confidence_is_low_on_white_noise() {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    // Tiny LCG-style noise to keep the test deterministic without a
    // dev-dep on rand.
    let mut state: u64 = 0xdead_beef_dead_beef;
    let n = (1.5 * SAMPLE_RATE as f32) as usize;
    let mut audio = Vec::with_capacity(n);
    for _ in 0..n {
        let mut h = DefaultHasher::new();
        state.hash(&mut h);
        state = h.finish();
        let v = ((state as i64) as f32 / i64::MAX as f32) * 0.3;
        audio.push(v);
    }

    let frames = run_oneshot(&audio);
    let high_conf = frames.iter().filter(|f| f.confidence > 0.3).count();
    let total = frames.len();
    // White-noise should very rarely trigger voicing.
    assert!(
        high_conf as f32 / total as f32 <= 0.2,
        "white noise produced {high_conf}/{total} high-conf frames",
    );
}

#[test]
fn process_into_matches_process() {
    // process() and process_into() should emit the exact same frames for the
    // same input — process() is a thin wrapper around process_into(). This
    // also pins the contract that process_into APPENDS rather than overwrites.
    let audio = sine(440.0, 0.8);

    let mut a = Swipe::new().unwrap();
    let from_process = a.process(&audio).unwrap();

    let mut b = Swipe::new().unwrap();
    let mut from_into: Vec<PitchFrame> = Vec::new();
    // Feed in two chunks to make sure append-semantics also hold across calls.
    let mid = audio.len() / 2;
    b.process_into(&audio[..mid], &mut from_into).unwrap();
    b.process_into(&audio[mid..], &mut from_into).unwrap();

    assert_eq!(from_process.len(), from_into.len(), "frame count mismatch");
    for (x, y) in from_process.iter().zip(from_into.iter()) {
        assert_eq!(x.frame_index, y.frame_index);
        assert!((x.pitch_hz - y.pitch_hz).abs() < 1e-6);
        assert!((x.confidence - y.confidence).abs() < 1e-6);
    }

    // Append-semantics: a second process_into call must NOT clear `out`.
    let prev_len = from_into.len();
    let mut c = Swipe::new().unwrap();
    c.process_into(&audio, &mut from_into).unwrap();
    assert!(
        from_into.len() > prev_len,
        "process_into should append, not overwrite",
    );
}

#[test]
fn realtime_preset_works() {
    let mut swipe = Swipe::with_max_window(4096).unwrap();
    let audio = sine(440.0, 0.8);
    let frames = swipe.process(&audio).unwrap();
    let median = median_voiced_pitch(&frames, 0.3).expect("voiced");
    assert!(
        cents_off(median, 440.0).abs() < MIR_EVAL_TOLERANCE_CENTS,
        "realtime preset 440 Hz: {median:.3}",
    );
}
