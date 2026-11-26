# Novelty: live transcription assistant with an interview glossary (Group 9)

The glossary **method** lives in this folder; the **app** that uses it is `app2/app_glossary.py` (built on this folder and the repository code) or `app1/app_glossary.py` (a self-contained copy). The project code (`src/group9_ds_ai_lab/`, the upload app in `deployment/app/`, the training and evaluation scripts) is used as is and not changed. **No retraining:** the app uses the trained model from Hugging Face (`kumargulshan/wav2vec2-base-ls960-noise-robust`), the project's 3-gram LM and word-error detector.

| Feature | How it works |
|---|---|
| **Live mode** | Record from the microphone. The audio is cut into 6–15 s segments at the gaps between words. Each segment is transcribed as soon as it ends, while the speaker keeps talking. The part not transcribed yet shows as a grey preview ("listening…"). |
| **Upload mode** | Any audio file (wav, flac, mp3, …). The transcript appears segment by segment. |
| **Model** | wav2vec 2.0 base fine-tuned on 960 h of LibriSpeech with noise, reverb, speed and telephone augmentation (`B960_rev`), plus a 3-gram LM and the word-error detector. |
| **Highlights** | Red = likely wrong, amber = check, blue = glossary suggestion, green = accepted suggestion, teal = edited by you, purple = a glossary term the model already wrote. |
| **Edit after listening** | Click any word to hear it (±1 s). Type a correction and press Replace or Enter. **Delete** removes the word. **Revert / undo** restores the original. |
| **Review queue** | Glossary suggestions and flagged words, in time order. Click a row to replay that part. |
| **Glossary** (novelty) | Up to 20 names, places or terms you expect to hear, entered fresh for each recording (nothing is stored between recordings). The audio is decoded twice from the same model output, and **both transcripts are shown**: **A**, the standard one, which is your working copy, and **B**, with a small boost for the glossary terms. Where they disagree on a term, a suggestion goes to the queue. It changes the transcript **only if you accept it**. |
| **Weak suggestions** | Differences on a glossary term that fail the audio check stay out of the queue. They are shown in light blue in transcript B. Click one to replay it, and accept it if the audio clearly says the term. |
| **Glossary status** | For each term: already in A, suggested (time and status), weak, partly heard in B (some words of the term but not all), or not heard. |
| **Export** | TXT, SRT and JSON (every word, your edits, every suggestion and your decision, both transcripts and the glossary status), plus transcripts A and B as text, and the recording as WAV in live mode. |

## Run

From the repository root, with the project venv:

```bash
python app2/app_glossary.py      # http://127.0.0.1:7861  (uses this folder)
python app1/app_glossary.py      # the same app, self-contained (see app1/README.md)
```

Optional environment variables:
- `APP_DEVICE=cuda|cpu` (default: `cuda` if available)
- `APP_THREADS=4`
- `APP_LM=3gram|4gram|none`
- `APP_MODEL=<path or HF id>` (default: the trained model on Hugging Face, `kumargulshan/wav2vec2-base-ls960-noise-robust`)
- `APP_HOST`, `APP_PORT=7861`
- `APP_SHARE=1` for a temporary public https link

**Microphone from another computer:** browsers allow the microphone only on `https` or on `localhost`. Use an SSH tunnel, then open `http://localhost:7861` on your own computer:

```bash
ssh -L 7861:127.0.0.1:7861 user1@<server>
```

## Files

| File | What it does |
|---|---|
| `glossary_assistant.py` | `GlossaryAssistant`: extends the project's `Assistant` (not modified). Runs the model once, decodes transcript A (standard) and B (glossary boost), finds the suggestions. |
| `glossary.py` | The glossary method: term clean-up, boosted beam search, diff and close-spelling candidates, audio check. |
| `app2/app_glossary.py` (outside this folder) | The Gradio interface: live and upload tabs, glossary box, highlighted transcript, replay, editing, review queue, export. |
| `app2/core/session.py` + `app2/glossary_ext/session.py` | The session logic, with no UI code: the audio buffer, segment cuts, transcription of each segment with context, the live preview (shared with the standard app), plus the glossary suggestions and accept / reject / undo. |
| `glossary_eval.py` | The evaluation: glossary off vs on (results in `experiments/results/glossary/`). |
| `tests/test_glossary.py` | Unit tests for the glossary and the app's session logic (run: `python -m pytest novelty/interview_glossary/tests`). |

## How the glossary works

No retraining is needed, and the acoustic model runs once per segment.

1. **Clean up the input.** At most 20 terms, each up to 4 words / 40 characters. They are converted to the model's alphabet (A–Z and apostrophe, upper case; hyphens become spaces).
2. **Decode twice from the same CTC log-probabilities:**
   - the **standard** beam search (beam 50, 3-gram LM, α 0.8, β 2.0), which is the evaluated system;
   - a **glossary-assisted** beam search, where:
     - only the *rare* words of each term get a bonus (weight 8), so in "Neeladri Road" only NEELADRI is boosted and ROAD is not pushed into the rest of the transcript;
     - glossary words the LM has never seen are scored like a rare word (log10 p = −5) instead of getting the unknown-word penalty.
3. **Find candidates.** Two kinds:
   - **diff:** wherever the two transcripts differ *and* the assisted side contains a complete glossary term. There is one candidate per term, covering only the standard words at the same time. For example, "GOSHEN COME ALIVE IN UTAPREDACE" gives GOSHEN COME → GULSHAN KUMAR and UTAPREDACE → UTTAR PRADESH, not one long suggestion.
   - **near (close spelling):** stretches of 1 to n+2 words, in either transcript, spelled close to a term. Closeness is the edit distance on the letters and on a rough sound key (CH/SH, PH/F, W/V, E/I, O/U, doubled letters…). This catches terms the boosted search spelled only partly or not at all, e.g. RAMMILASPASRAN → RAM VILAS PASWAN. At each place the closest few spans are scored against the audio and the best one is kept.
4. **Check against the audio.** CTC log-likelihood of the suggested words minus that of the standard words, on the same frames:
   - a diff candidate is kept if it is at least −4 nats; otherwise it is shown as *weak*;
   - a near candidate needs spelling distance ≤ 0.1 and at least −16 nats; otherwise it is dropped;
   - where candidates overlap, the best-supported one wins.

   This keeps unspoken terms such as "Aniruddha" out.
5. **Review.** The reviewer replays the region and accepts or rejects. Nothing changes without approval. Weak candidates are still shown in transcript B, so a reviewer who hears the term can accept them.

**How the settings were chosen.** The boost weight (8) was chosen on the dev set for the first version (`experiments/results/glossary/B960_rev_lm3_dev_grid.csv`). The diff and near limits were chosen on the dev set (`B960_rev_lm3_dev_grid_v2.csv`) with a rule fixed before that grid was run:
- the unspoken-term rate must stay ≤ 0.1 %, suggestion precision ≥ 95 %, and WER no worse than off;
- among those settings, take the highest spoken-term accuracy;
- within 1 point of it, take the highest precision.

## Results

Held-out LibriSpeech test (5,559 utterances per condition), glossary off vs on. v1 is the first version (diff suggestions only). v2 is the current one (one suggestion per term, plus near matches), scored on the same decodes (`B960_rev_lm3_test_v1_vs_v2.csv`).

| Condition | Spoken glossary terms correct: off | v1 | **v2** | Unspoken terms in output: off | v1 | v2 | WER: off | v2 |
|---|---|---|---|---|---|---|---|---|
| Unseen noise (ESC-50 + brown, 0–10 dB) | 58.6 % | 81.2 % | **82.5 %** | 0.001 % | 0.020 % | 0.021 % | 7.89 % | 6.90 % |
| ESC-50 0 dB (hardest) | 47.0 % | 71.5 % | **72.9 %** | 0.002 % | 0.074 % | 0.074 % | 15.06 % | 14.02 % |
| Clean | 63.9 % | 85.2 % | **86.7 %** | 0.001 % | 0.006 % | 0.009 % | 5.22 % | 4.26 % |
| Unseen noise, terms the LM does not know | 25.7 % | 70.9 % | **74.1 %** | | | | | |

"On" means every suggestion was accepted.

- **Glossary sizes.** Every clip had a 20-term glossary: up to 2 rare words actually spoken, plus unspoken rare words from other chapters and Indian names and places that never occur in LibriSpeech.
- **Gain over glossary off** (unseen noise, v2): +24.0 points (95 % paired-bootstrap CI 22.9 to 25.2).
- **Gain of v2 over v1:** +1.4 points (CI 0.9 to 1.8). For terms the LM does not know, +3.2 points (CI 1.2 to 5.3). 25 % of v2 suggestions are near matches.
- **Suggestion precision.** 96.8 % (v1: 96.9 %).
- **Reviewer load.** About 1.4 suggestions per minute of audio.
- **Unspoken terms.** With everything accepted, they appeared 135 times (v1: 126; off: 7) out of 645,642 clip-term pairs.
- **Live vs whole-file.** On 20 one-minute noisy recordings (ESC-50 / white at 5 dB, from test-other), live segmentation with 4 s left and 1 s right context gives 13.97 % WER, against 14.26 % when the whole file is decoded at once.
- **Speed.** With the GPU, each 1 s microphone chunk is processed in about 30 ms on average (at most about 0.25 s when a segment is finalised). The glossary adds about 35 ms of decoding per utterance (v1: about 25 ms).
