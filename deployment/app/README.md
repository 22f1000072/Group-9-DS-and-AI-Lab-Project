# Noise-robust transcription assistant (demo app)

Upload or record English speech → transcript with the **words most likely to be wrong highlighted** →
click a word to **replay it (±1 s)** → type corrections → **export** TXT / SRT (flags marked `?`) / JSON
(every word with start/end time, confidence, error risk and replay window).

| Component | What runs |
|---|---|
| Acoustic model | wav2vec 2.0 base + CTC, fine-tuned on 960 h LibriSpeech with noise, reverberation, speed and telephone augmentation ([B960_rev on Hugging Face](https://huggingface.co/kumargulshan/wav2vec2-base-ls960-noise-robust)) |
| Long audio | overlapping 30 s windows (2 s context each side), log-probabilities stitched before decoding |
| Decoder | beam search (width 50) + pruned 3-gram KenLM (α 0.8, β 2.0); 4-gram optional |
| Error detector | gradient-boosted trees on 14 word/utterance features (AUROC 0.895, ECE 0.007 on the test grid) |
| Replay window | flagged word start − 1 s … end + 1 s (99 % of flags contain the true word, objective O4) |

Measured on CPU (4 threads, 2-minute recordings): real-time factor **0.05** (fp32) / **0.034** (int8) — see
`milestones/M5_evaluation/tables/assistant_benchmark.csv`.

## Run locally

From the repository root, with the project environment (see the main README):

```bash
G9_VENV=/path/to/venv source scripts/env.sh
python deployment/app/app.py                    # open http://127.0.0.1:7860
```

Needs `data/raw/librispeech_lm/3-gram.pruned.1e-7.arpa` (OpenSLR 11, `3-gram.pruned.1e-7.arpa.gz`, gunzipped) and
the detector in `experiments/results/B960_rev_lm3/error_detector/model.joblib` (in git). The acoustic model is
read from `experiments/runs/B960_rev/best` if present, otherwise downloaded from Hugging Face.

Options (environment variables): `APP_THREADS=4`, `APP_LM=3gram|4gram|none`, `APP_MODEL=<folder or HF id>`,
`APP_HOST=0.0.0.0`, `APP_PORT=7860`, `APP_SHARE=1` (temporary public gradio.live link).

On a remote server, forward the port from your PC: `ssh -L 7860:127.0.0.1:7860 <user>@<server>`, then open
http://127.0.0.1:7860 in your browser.

## Command line (no web UI)

```bash
python scripts/transcribe.py recording.wav --out results/      # writes recording.txt / .srt / .json
```

## Container

```bash
docker build -f deployment/app/Dockerfile -t g9-asr-assistant .
docker run -p 7860:7860 g9-asr-assistant
```
The image downloads the model from Hugging Face and the 3-gram LM from OpenSLR at build time.

## Examples

`examples/` holds a 34 s excerpt of LibriSpeech test-other (CC BY 4.0), clean and with white noise at 5 dB SNR;
`examples/reference.txt` is its reference transcript.
