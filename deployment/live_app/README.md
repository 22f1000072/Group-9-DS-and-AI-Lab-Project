# Live transcription assistant with an interview glossary (Group 9)

The live app. It runs alongside the original upload-only app in `deployment/app/`.

| Feature | How it works |
|---|---|
| **Live mode** | Record from the microphone. The audio is cut into 6–15 s segments at the gaps between words. Each segment is transcribed as soon as it ends, while the speaker keeps talking. The part not transcribed yet shows as a grey preview ("listening…"). |
| **Upload mode** | Any audio file (wav, flac, mp3, …). The transcript appears segment by segment. |
| **Model** | wav2vec 2.0 base fine-tuned on 960 h of LibriSpeech with noise, reverb, speed and telephone augmentation (`B960_rev`), plus a 3-gram LM and the word-error detector. |
| **Highlights** | Red = likely wrong, amber = check, blue = glossary suggestion, green = accepted suggestion, teal = edited by you, purple = a glossary term the model already wrote. |
| **Edit after listening** | Click any word to hear it (±1 s). Type a correction and press Replace or Enter. **Delete** removes the word. **Revert / undo** restores the original. |
| **Review queue** | Glossary suggestions and flagged words, in time order. Click a row to replay that part. |
| **Glossary** (novelty) | Up to 20 names, places or terms you expect to hear. Where the glossary-assisted decoding differs on a term, a suggestion goes to the queue. It changes the transcript **only if you accept it**. |
| **Export** | TXT, SRT and JSON (every word, your edits, every suggestion and your decision), plus the recording as WAV in live mode. |

## Run

From the repository root, with the project venv:

```bash
python deployment/live_app/app.py            # http://127.0.0.1:7861
```

Optional environment variables:
- `APP_DEVICE=cuda|cpu` (default: `cuda` if available)
- `APP_THREADS=4`
- `APP_LM=3gram|4gram|none`
- `APP_MODEL=<path or HF id>` (default: `experiments/runs/B960_rev/best`, else `kumargulshan/wav2vec2-base-ls960-noise-robust`)
- `APP_HOST`, `APP_PORT=7861`
- `APP_SHARE=1` for a temporary public https link

**Microphone from another computer:** browsers allow the microphone only on `https` or on `localhost`. Use an SSH tunnel, then open `http://localhost:7861` on your own computer:

```bash
ssh -L 7861:localhost:7861 user1@<server>
```

## Files

| File | What it does |
|---|---|
| `app.py` | The Gradio interface: live and upload tabs, glossary box, highlighted transcript, replay, editing, review queue, export. |
| `session.py` | The session logic, with no UI code: the audio buffer, segment cuts, transcription of each segment with context, the live preview, and edit / accept / reject / undo. |
| `../../src/group9_ds_ai_lab/assistant/glossary.py` | The glossary method (see below). |
| `../../src/group9_ds_ai_lab/assistant/engine.py` | `Assistant.transcribe(audio, glossary=...)`. Without a glossary it gives exactly the same output as before. |
| `../../scripts/glossary_eval.py` | The evaluation: glossary off vs on. |
| `../../tests/test_glossary.py` | Unit tests for the glossary and the session logic. |

## How the glossary works

No retraining is needed, and the acoustic model runs once per segment.

1. **Clean up the input.** At most 20 terms, each up to 4 words / 40 characters. They are converted to the model's alphabet (A–Z and apostrophe, upper case; hyphens become spaces).
2. **Decode twice from the same CTC log-probabilities:**
   - the **standard** beam search (beam 50, 3-gram LM, α 0.8, β 2.0), which is the evaluated system;
   - a **glossary-assisted** beam search, where:
     - only the *rare* words of each term get a bonus (weight 8), so in "Neeladri Road" only NEELADRI is boosted and ROAD is not pushed into the rest of the transcript;
     - glossary words the LM has never seen are scored like a rare word (log10 p = −5) instead of getting the unknown-word penalty.
3. **Align the two transcripts.** Wherever they differ *and* the assisted side contains a complete glossary term, that becomes a suggestion with its time.
4. **Check against the audio.** A suggestion is kept only if the CTC log-likelihood of the suggested words, on the same frames, is at most 8 nats below that of the standard words. This keeps unspoken terms such as "Aniruddha" out.
5. **Review.** The reviewer replays the region and accepts or rejects. Nothing changes without approval.

The weight and the audio-check limit were chosen on the dev set (`experiments/results/glossary/B960_rev_lm3_dev_grid.csv`).

## Results

Held-out LibriSpeech test (5,559 utterances per condition), glossary off vs on:

| Condition | Spoken glossary terms correct: off | on, all suggestions accepted | Unspoken terms in output: off | on | WER: off | on |
|---|---|---|---|---|---|---|
| Unseen noise (ESC-50 + brown, 0–10 dB) | 58.5 % | **81.1 %** | 0.001 % | 0.020 % | 7.89 % | 6.92 % |
| ESC-50 0 dB (hardest) | 47.0 % | 71.4 % | 0.002 % | 0.074 % | 15.06 % | 14.01 % |
| Clean | 63.9 % | 85.2 % | 0.001 % | 0.006 % | 5.22 % | 4.30 % |

- **Glossary sizes.** Every clip had a 20-term glossary: up to 2 rare words actually spoken, plus unspoken rare words from other chapters and Indian names and places that never occur in LibriSpeech.
- **Gain.** The gain in spoken-term accuracy on unseen noise is +22.6 points (95 % paired-bootstrap CI: 21.5 to 23.8).
- **Suggestion precision.** 96.9 % of suggestions were right.
- **Reviewer load.** About 1.3 suggestions per minute of audio.
- **Unknown names.** Terms the LM does not know at all (the realistic case for Indian names) go from 25.6 % to 70.9 % correct.
- **Unspoken terms.** They appeared 7 times without the glossary and 126 times with it (accepting everything), out of 645,642 clip-term pairs.
- **Live vs whole-file.** On 20 one-minute noisy recordings (ESC-50 / white at 5 dB, from test-other), live segmentation with 4 s left and 1 s right context gives 13.97 % WER, against 14.26 % when the whole file is decoded at once.
- **Speed.** With the GPU, each 1 s microphone chunk is processed in about 30 ms on average (at most about 0.25 s when a segment is finalised). The glossary adds about 25 ms of decoding per utterance.
