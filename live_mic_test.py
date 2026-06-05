"""Layer 4 (live) — speak a cue word into your mic and watch Stage 2 fire.

Drives the REAL Vosk-backed CueWordDetector, so this validates the actual
Stage 2 path. No API key, no signup — just the Vosk model on disk.

Setup:
  1) download a Vosk model to ./models (see download_vosk_model.py or README)
  2) python live_mic_test.py

Say a word from config.CUE_WORDS (default "help"). Ctrl+C to stop.
"""
import config
from pvrecorder import PvRecorder
from stage2_cueword import CueWordDetector, CueWordError


def main() -> None:
    try:
        detector = CueWordDetector()
    except CueWordError as e:
        raise SystemExit(f"Stage 2 init failed: {e}")

    rec = PvRecorder(frame_length=512, device_index=-1)
    print(f"Listening on '{rec.selected_device}'. "
          f"Say one of {config.CUE_WORDS}. Ctrl+C to stop.")

    rec.start()
    frames = 0
    try:
        while True:
            pcm = rec.read()                       # list[int16], len 512
            ts = frames * 512 / config.SAMPLE_RATE
            frames += 1
            event = detector.process(pcm, ts)
            if event is not None:
                print(f"  >>> DETECTED '{event.keyword}' at t={event.ts:.1f}s "
                      f"(conf={event.confidence_proxy:.2f})")
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        rec.stop()
        final = detector.flush(frames * 512 / config.SAMPLE_RATE)  # drain last utterance
        if final is not None:
            print(f"  >>> DETECTED '{final.keyword}' (flush) conf={final.confidence_proxy:.2f}")
        rec.delete()
        detector.close()


if __name__ == "__main__":
    main()
