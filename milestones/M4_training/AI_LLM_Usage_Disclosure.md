# AI & LLM Usage Disclosure — Server Debugging and Configuration

**Group 9 · DS & AI Lab · Milestone 4 (November 2026)**

---

## 1. AI Tools Utilized

| Field | Details |
|---|---|
| Tool name | Claude Code (Anthropic command-line coding assistant), model Claude Opus 5.5 |
| Where it ran | In a terminal session on the group's GPU server (1 × NVIDIA L40S, Linux). |
| Purpose of use | Debugging and configuration: diagnosing failed jobs and library conflicts, setting up the Python environment, long-running job management, data-loading problems, Hugging Face model upload. |
| Shared chat link / conversation file | Claude Code sessions have no public share link. The session log is stored on the server. It can be provided on request. |
| Contribution and validation | The assistant diagnosed problems by reading error messages, logs and package metadata, and proposed and applied fixes. Every fix was checked by re-running the failing step and comparing results. Account log-ins (Hugging Face, GitHub) and the git push were done by the team member, not by the AI; tokens were never shared with the tool. |

---

## 2. Significant Prompts and Responses

Prompts are quoted as typed (spelling unchanged); responses are summarised.

### 2.1 Training jobs dying when the SSH session closed

- **Problem addressed:** the first training run stopped at step ≈ 100 when the SSH connection dropped, even when started with `nohup`.
- **Prompt (summary):** the training stopped after the connection closed; keep jobs running independently of the session.
- **Relevant response:** run every long job inside `tmux`, started in its own systemd user scope so it survives logout; make training resumable by saving `last/` (model, optimizer, scheduler, step, RNG state) every few thousand steps and adding a `--resume` option; make the run queues skip finished stages.
- **Action taken:** adopted. All later training runs (13 runs) ran in tmux and completed; the run queues were re-run safely, skipping finished stages. Standing rule from the team: never kill the tmux server.

### 2.2 Library conflicts in the audio stack

- **Problem addressed:** the installed `torchaudio` and `datasets` ≥ 4 (which requires `torchcodec`) did not match the PyTorch build, so audio could not be decoded.
- **Relevant response:** remove the dependency instead of downgrading PyTorch: decode the original FLAC bytes stored in the Hugging Face arrow shards with `soundfile`, and resample with `scipy.signal.resample_poly`.
- **Action taken:** adopted (`src/group9_ds_ai_lab/data/audio.py`). Verified with the full quality-control pass: 292,367 utterances decoded, 0 decode failures, all 16 kHz mono.

### 2.3 Memory blow-up in data-loader workers

- **Problem addressed:** with 28 data-loader workers on the 460 h / 960 h sets, each worker copied ≈ 0.5 GB per arrow shard into its own memory (out-of-memory risk).
- **Relevant response:** keep the memory-mapped Arrow `ChunkedArray` (zero-copy) instead of calling `combine_chunks()`.
- **Action taken:** adopted; the 460 h and 960 h trainings ran without memory errors.

### 2.4 Python 3.14 packaging problems

- **Prompts:**
  - "why there are two requirments file and not one can we merge them into one file or not or is there a point in them being separate will merging them break anything"
  - "before merging them check whether or not it will affect the code or current raining in any way"
- **Problems addressed:** installing `pyctcdecode` silently downgraded NumPy (it pins `numpy<2`); `kenlm` has no pre-built wheel for Python 3.14; two partly conflicting requirements files.
- **Relevant response:** install `pyctcdecode` with `pip install --no-deps`; build `kenlm` from source; merge both files into one fully pinned `requirements.txt` with the two-step install documented in its header; check the merged file against the running environment with a pip dry-run before changing anything.
- **Action taken:** adopted. The dry-run showed no version changes for the running jobs; NumPy stayed at 2.5.5. The virtual environment is kept outside the repository at the team's request, so it is never committed.

### 2.5 Offline model cache and GPU sharing

- **Problems addressed:** (a) runs failing or stalling when trying to reach the Hugging Face Hub; (b) whether to train two models at once on the single GPU.
- **Relevant response:**
  - (a) keep the Hugging Face cache inside the git-ignored `data/hf_home/` and run with `HF_HUB_OFFLINE=1` once the checkpoints are cached;
  - (b) measure throughput: two parallel runs gave ≈ 710 audio-seconds per second each vs ≈ 1,580 alone, so there was no total gain.
- **Action taken:** both adopted: offline mode in `scripts/env.sh`; models trained sequentially.

### 2.6 Publishing the models on Hugging Face

- **Prompt:** "i first want to upload my model to hugging face tell me the steps"
- **Relevant response:** explained how to upload a model to Hugging Face: log in (done by the team member via the device-code flow; token stored in the git-ignored cache); prepare staging folders with weights, tokenizer and pre-processing files and a model card; load test before upload; checksum and download test after upload.
- **Action taken:** six models published. Verified: SHA-256 of every uploaded weight file equals the local checkpoint; a fresh download transcribes test audio; WER of the uploaded files matches the evaluation pipeline (e.g. 3.92 % vs 3.99 % on the same 200 test-clean utterances).

### 2.7 Git authentication

- **Prompt:** "here si the thing i have the same repository in github where iam trying to push the content to . so how do i do that"
- **Relevant response:** explained the steps to push the content to the existing GitHub repository: create a fine-grained personal access token with Contents: read and write on this repository only, and push from the server copy.
- **Action taken:** the team member created the token and pushed.

---

## 3. Team Accountability Declaration

*To be confirmed and initialled by each member before submission.*

All AI-generated fixes, configurations and recommendations incorporated into the project have been reviewed and validated by the team (validation evidence for each item is given in Section 2). Project decisions, experimental results and conclusions are supported by the team's independent reasoning and evidence. Every team member understands their contributions and can explain and defend the relevant work independently.

| Name | Roll No. | Initials |
|---|---|---|
| Gulshan Kumar | 22f1000072 | G.K|
| Parashmani Datta | 21f2000987 | P.D|
| Subhajit Saha | 21f1003945 | S.S|
