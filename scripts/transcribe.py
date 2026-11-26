"""Command-line transcription assistant: audio file(s) -> transcript with flagged words, as TXT / SRT / JSON.

Usage:
  python scripts/transcribe.py recording.wav                       # prints the transcript, flags marked [WORD?]
  python scripts/transcribe.py a.wav b.mp3 --out results/ --threshold 0.4 --lm 3gram --threads 4 [--int8]
Outputs (with --out): <name>.txt, <name>.srt, <name>.json (all words with times, confidence, risk, replay window)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from group9_ds_ai_lab.assistant.engine import Assistant  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("audio", nargs="+")
    ap.add_argument("--out", help="folder for .txt/.srt/.json (default: print only)")
    ap.add_argument("--threshold", type=float, default=0.5, help="risk above which a word is flagged")
    ap.add_argument("--lm", choices=["3gram", "4gram", "none"], default="3gram")
    ap.add_argument("--model", default=None, help="local folder or Hugging Face id (default: B960_rev)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--int8", action="store_true", help="int8 dynamic quantisation (faster on CPU)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    asst = Assistant(model=args.model, lm=None if args.lm == "none" else args.lm, threshold=args.threshold,
                     threads=args.threads, quantize=args.int8, device=args.device)
    for f in args.audio:
        tr = asst.transcribe(f)
        n = sum(w.flagged for w in tr.words)
        print(f"\n== {f}  ({tr.duration_s:.1f} s, {len(tr.words)} words, {n} flagged, RTF {tr.rtf:.3f})")
        print(tr.to_txt())
        if args.out:
            out = Path(args.out)
            out.mkdir(parents=True, exist_ok=True)
            stem = Path(f).stem
            (out / f"{stem}.txt").write_text(tr.to_txt() + "\n")
            (out / f"{stem}.srt").write_text(tr.to_srt())
            (out / f"{stem}.json").write_text(tr.to_json())
            print(f"-> {out}/{stem}.txt/.srt/.json")


if __name__ == "__main__":
    main()
