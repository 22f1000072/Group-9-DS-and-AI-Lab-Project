"""Noise-robust transcription assistant (Group 9): transcribe, flag risky words, replay them, correct, export.

Run from the repository root:
    python deployment/app/app.py                 # http://127.0.0.1:7860
Environment variables (optional):
    APP_THREADS=4        CPU threads for the acoustic model
    APP_LM=3gram         3gram (default, fast to load) | 4gram (more accurate, ~4 GB, slow to load) | none
    APP_MODEL=<path or Hugging Face id>   default: local experiments/runs/B960_rev/best, else the published model
    APP_HOST=127.0.0.1   APP_PORT=7860   APP_SHARE=0 (1 = temporary public gradio.live link)
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from group9_ds_ai_lab.assistant.engine import Assistant, Transcript, Word, load_audio  # noqa: E402
from group9_ds_ai_lab.utils.paths import SAMPLE_RATE  # noqa: E402

LM = os.environ.get("APP_LM", "3gram")
ASSISTANT = Assistant(model=os.environ.get("APP_MODEL") or None, lm=None if LM == "none" else LM,
                      threads=int(os.environ.get("APP_THREADS", "4")))
FLAG_COLS = ["#", "word", "time (s)", "risk", "correction"]
LEGEND = {"likely wrong": "#f4a3a3", "check": "#f8d48a"}


# ------------------------------------------------------------------------------------------ helpers
def _fmt_t(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:05.2f}"


def _words(state) -> list[Word]:
    return [Word(**w) for w in state["words"]]


def _highlight(words: list[Word], thr: float):
    out = []
    for w in words:
        label = "likely wrong" if w.p_error >= max(thr, 0.7) else ("check" if w.p_error >= thr else None)
        out.append((w.word + " ", label))
    return out


def _flag_table(words: list[Word], thr: float, corrections: dict | None = None) -> pd.DataFrame:
    corrections = corrections or {}
    rows = [[i, w.word, _fmt_t(w.start), f"{100 * w.p_error:.0f} %", corrections.get(str(i), "")]
            for i, w in enumerate(words) if w.p_error >= thr]
    return pd.DataFrame(rows, columns=FLAG_COLS)


def _clip(state, i: int):
    w = state["words"][i]
    a, b = int(w["replay_start"] * SAMPLE_RATE), int(w["replay_end"] * SAMPLE_RATE)
    x = np.asarray(state["audio"], dtype=np.float32)[a:b]
    return (SAMPLE_RATE, (np.clip(x, -1, 1) * 32767).astype(np.int16))


def _stats(state, thr: float) -> str:
    n_flag = sum(w["p_error"] >= thr for w in state["words"])
    t = state["timings_s"]
    return (f"**Duration** {_fmt_t(state['duration_s'])} · **words** {len(state['words'])} · "
            f"**flagged** {n_flag} ({100 * n_flag / max(len(state['words']), 1):.1f} %) · "
            f"**processing** {sum(t.values()):.1f} s (real-time factor {state['rtf']:.2f}; "
            f"acoustic model {t['acoustic_model']:.1f} s, LM decoding {t['decoding']:.1f} s)")


# ------------------------------------------------------------------------------------------ callbacks
def transcribe(audio_path, thr):
    if not audio_path:
        raise gr.Error("Upload or record an audio file first.")
    x = load_audio(audio_path)
    if len(x) < SAMPLE_RATE * 0.3:
        raise gr.Error("The recording is too short.")
    tr: Transcript = ASSISTANT.transcribe(x, threshold=thr)
    state = tr.to_dict()
    state["audio"] = x  # gr.State stays on the server, so the array is not sent to the browser
    state["corrections"] = {}
    words = tr.words
    return (state, _highlight(words, thr), _flag_table(words, thr), _stats(state, thr),
            tr.text, None, "Click a word in the transcript or a row in the table to replay it.")


def rethreshold(state, thr):
    if not state:
        return gr.update(), gr.update(), gr.update()
    words = _words({"words": state["words"]})
    return _highlight(words, thr), _flag_table(words, thr, state["corrections"]), _stats(state, thr)


def replay_word(state, evt: gr.SelectData):
    if not state:
        return None, ""
    i = evt.index if isinstance(evt.index, int) else evt.index[0]
    w = state["words"][i]
    return _clip(state, i), (f"**{w['word']}** at {_fmt_t(w['start'])} · risk {100 * w['p_error']:.0f} % · "
                             f"playing {_fmt_t(w['replay_start'])}–{_fmt_t(w['replay_end'])}")


def replay_row(state, table, evt: gr.SelectData):
    if not state or table is None or len(table) == 0:
        return None, ""
    row = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    i = int(table.iloc[row]["#"])
    w = state["words"][i]
    return _clip(state, i), (f"**{w['word']}** at {_fmt_t(w['start'])} · risk {100 * w['p_error']:.0f} % · "
                             f"playing {_fmt_t(w['replay_start'])}–{_fmt_t(w['replay_end'])}")


def apply_corrections(state, table):
    if not state:
        return state, ""
    corr = dict(state.get("corrections", {}))
    if table is not None:
        for _, r in table.iterrows():
            c = str(r["correction"] or "").strip()
            if c:
                corr[str(int(r["#"]))] = c
            else:
                corr.pop(str(int(r["#"])), None)
    state["corrections"] = corr
    out = [corr.get(str(i), w["word"]) for i, w in enumerate(state["words"])]
    return state, " ".join(t for t in out if t != "-")  # '-' deletes a word


def export(state, final_text, thr):
    if not state:
        raise gr.Error("Nothing to export yet.")
    d = Path(tempfile.mkdtemp(prefix="g9_export_"))
    words = _words({"words": state["words"]})
    for w in words:
        w.flagged = w.p_error >= thr
    tr = Transcript(state["text"], words, state["duration_s"], state["timings_s"], {**state["settings"], "threshold": thr})
    (d / "transcript.txt").write_text((final_text or tr.text).strip() + "\n")
    (d / "transcript_flagged.srt").write_text(tr.to_srt())
    info = tr.to_dict()
    info["corrected_text"] = final_text
    info["corrections"] = {state["words"][int(k)]["word"] + f" (#{k})": v for k, v in state.get("corrections", {}).items()}
    (d / "transcript.json").write_text(json.dumps(info, indent=2))
    return [str(d / "transcript.txt"), str(d / "transcript_flagged.srt"), str(d / "transcript.json")]


# ------------------------------------------------------------------------------------------ layout
with gr.Blocks(title="Noise-robust transcription assistant") as demo:
    gr.Markdown("# Noise-robust transcription assistant\n"
                "Transcribes English speech recorded in noise, **highlights the words most likely to be wrong**, "
                "and lets you replay each one (±1 s) and correct it before exporting. "
                "Model: wav2vec 2.0 base fine-tuned on 960 h of LibriSpeech with noise and reverberation, "
                f"{'4-gram' if LM == '4gram' else '3-gram' if LM == '3gram' else 'no'} language model, word-error detector.")
    state = gr.State(None)
    with gr.Row():
        with gr.Column(scale=1):
            audio_in = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Audio (wav, flac, mp3, …)")
            thr = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Flag sensitivity: risk threshold",
                            info="Lower = more words flagged (more errors caught, more false alarms).")
            run = gr.Button("Transcribe", variant="primary")
            stats = gr.Markdown()
            ex = Path(__file__).parent / "examples"
            gr.Examples([[str(ex / "librispeech_white_noise_5dB.flac")], [str(ex / "librispeech_clean.flac")]],
                        inputs=[audio_in], label="Examples (LibriSpeech test-other, CC BY 4.0; white noise at 5 dB SNR / clean)")
        with gr.Column(scale=2):
            highlighted = gr.HighlightedText(label="Transcript (click a word to replay it)", color_map=LEGEND,
                                             show_legend=True, combine_adjacent=False)
            replay_info = gr.Markdown()
            replay = gr.Audio(label="Replay window", autoplay=True, interactive=False)
    with gr.Row():
        with gr.Column():
            flags = gr.Dataframe(headers=FLAG_COLS, label="Flagged words: click a row to replay, type a correction "
                                 "('-' deletes the word)", interactive=True, wrap=True)
            fix = gr.Button("Apply corrections")
        with gr.Column():
            final = gr.Textbox(label="Final transcript (editable)", lines=8)
            exp = gr.Button("Export TXT / SRT / JSON")
            files = gr.File(label="Downloads", file_count="multiple")

    run.click(transcribe, [audio_in, thr], [state, highlighted, flags, stats, final, replay, replay_info])
    thr.release(rethreshold, [state, thr], [highlighted, flags, stats])
    highlighted.select(replay_word, [state], [replay, replay_info])
    flags.select(replay_row, [state, flags], [replay, replay_info])
    fix.click(apply_corrections, [state, flags], [state, final])
    exp.click(export, [state, final, thr], [files])

if __name__ == "__main__":
    demo.queue().launch(server_name=os.environ.get("APP_HOST", "127.0.0.1"),
                        server_port=int(os.environ.get("APP_PORT", "7860")),
                        share=os.environ.get("APP_SHARE", "0") == "1")
