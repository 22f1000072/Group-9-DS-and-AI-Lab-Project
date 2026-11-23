"""Live noise-robust transcription assistant with an interview glossary (Group 9).

  * Live mode   : record from the microphone; the transcript is built while you speak (segments are cut at
                  pauses and transcribed right away; the unfinished tail is shown as a preview)
  * Upload mode : any audio file; the transcript appears segment by segment
  * Every word  : click to replay it (±1 s), then edit, delete or revert it
  * Highlights  : words the error detector thinks are likely wrong / worth checking
  * Glossary    : up to 20 names / places / terms you expect to hear; where the glossary-assisted decoding
                  differs on a term, a suggestion goes to the review queue; it changes the transcript only
                  when you accept it
Model: wav2vec 2.0 base fine-tuned on 960 h LibriSpeech with noise, reverberation, speed and telephone
augmentation (B960_rev), 3-gram LM, word-error detector.

Run from the repository root:
    python deployment/live_app/app.py                      # http://127.0.0.1:7861
Environment variables (optional):
    APP_DEVICE=cuda|cpu (default: cuda if available)   APP_THREADS=4   APP_LM=3gram|4gram|none
    APP_MODEL=<path or Hugging Face id>   APP_HOST=127.0.0.1   APP_PORT=7861   APP_SHARE=0
The browser only allows the microphone on https or on localhost: from another computer use an SSH tunnel
(ssh -L 7861:localhost:7861 user@server, then open http://localhost:7861) or APP_SHARE=1.
"""
import os
import sys
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import session as ss  # noqa: E402
from group9_ds_ai_lab.assistant.engine import Assistant, load_audio  # noqa: E402


def _device():
    d = os.environ.get("APP_DEVICE")
    if d:
        return d
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


LM = os.environ.get("APP_LM", "3gram")
A = Assistant(model=os.environ.get("APP_MODEL") or None, lm=None if LM == "none" else LM,
              threads=int(os.environ.get("APP_THREADS", "4")), device=_device())
COLORS = {"likely wrong": "#f4a3a3", "check": "#f8d48a", "glossary suggestion": "#9cc3f5",
          "glossary accepted": "#b6e3b6", "glossary term": "#d9c8f5", "edited": "#7fd1c7", "listening…": "#e6e6e6"}
EXAMPLES = ROOT / "deployment" / "app" / "examples"


# ------------------------------------------------------------------------------------------ helpers
def _terms(text):
    return A.parse_glossary(text)[0] if (text or "").strip() else []


def _views(S, thr):
    """transcript, word-id map, review queue, final text, stats."""
    if S is None:
        return [], [], pd.DataFrame(columns=ss.QUEUE_COLS), "", ""
    hl, ids = ss.highlighted(S, thr)
    q = pd.DataFrame(ss.queue_rows(S, thr), columns=ss.QUEUE_COLS)
    return hl, ids, q, ss.final_text(S), ss.stats(S, thr)


def _clip(S, a, b):
    x = ss.audio(S, a, b)
    return (ss.SR, (np.clip(x, -1, 1) * 32767).astype(np.int16)) if len(x) else None


def glossary_info(text):
    if not (text or "").strip():
        return "No glossary: standard transcription only."
    terms, warn = A.parse_glossary(text)
    known = A.unigram_logp
    parts = []
    for t in terms:
        new = [w for w in t.words if w not in known]
        parts.append(f"`{t.text}`" + (f" (new to the LM: {', '.join(new)})" if new else ""))
    msg = f"**{len(terms)}/20 terms:** " + ", ".join(parts)
    if warn:
        msg += "\n\n" + "\n".join(f"⚠ {w}" for w in warn)
    return msg


# ------------------------------------------------------------------------------------------ live mode
def live_start(gloss):
    S = ss.new_session()
    S["source"] = "microphone"
    S["glossary"] = [t.text for t in _terms(gloss)]
    return S, None, "Recording… the transcript appears as you speak."


def live_chunk(chunk, S, gloss, thr):
    if S is None:
        S = ss.new_session()
        S["source"] = "microphone"
    if chunk is not None:
        sr, y = chunk
        if y is not None and len(y):
            ss.append_audio(S, load_audio(y, sr))
    terms = _terms(gloss)
    S["glossary"] = [t.text for t in terms]
    ss.process(S, A, terms, thr, final=S["stopped"], max_segments=4 if S["stopped"] else 1)
    hl, ids, q, txt, st = _views(S, thr)
    return S, hl, ids, q, txt, st


def live_stop(S, gloss, thr):
    if S is None:
        return (S,) + tuple(_views(S, thr)) + ("",)
    S["stopped"] = True
    ss.process(S, A, _terms(gloss), thr, final=True, max_segments=1000)
    hl, ids, q, txt, st = _views(S, thr)
    return S, hl, ids, q, txt, st, "Recording finished. Click any word to replay and edit it."


# ------------------------------------------------------------------------------------------ upload mode
def upload_run(path, gloss, thr):
    if not path:
        raise gr.Error("Choose an audio file first.")
    x = load_audio(path)
    if len(x) < 0.3 * ss.SR:
        raise gr.Error("The recording is too short.")
    S = ss.new_session()
    S["source"] = Path(path).name
    terms = _terms(gloss)
    S["glossary"] = [t.text for t in terms]
    ss.append_audio(S, x)
    while ss.process(S, A, terms, thr, final=False, max_segments=1):
        hl, ids, q, txt, st = _views(S, thr)
        yield S, hl, ids, q, txt, st, f"Transcribing… {ss._fmt(S['committed'] / ss.SR)} of {ss._fmt(ss.duration(S))}"
    ss.process(S, A, terms, thr, final=True, max_segments=1000)
    hl, ids, q, txt, st = _views(S, thr)
    yield S, hl, ids, q, txt, st, "Done. Click any word to replay and edit it."


# ------------------------------------------------------------------------------------------ review
def select_word(S, ids, evt: gr.SelectData):
    if S is None or evt.index is None:
        return None, gr.update(), None, gr.update()
    k = evt.index if isinstance(evt.index, int) else evt.index[0]
    if k >= len(ids) or ids[k] is None:
        return None, gr.update(), None, "Still listening: this part is not transcribed yet."
    w = ss.word(S, ids[k])
    pend = ss.pending_of(S)
    if w["id"] in pend:
        return _select_sugg(S, pend[w["id"]])
    a, b = max(w["start"] - ss.REPLAY_S, 0), min(w["end"] + ss.REPLAY_S, ss.duration(S))
    info = (f"**{w['word']}** at {ss._fmt(w['start'])} · risk {100 * w['p_error']:.0f} %"
            + (f" · originally **{w['original']}**, corrected to **{ss.edited_text(S, w['id'])}**" if w["original"] else "")
            + (f" · added when you corrected **{ss.word(S, w['added_to'])['original']}**"
               if w.get("added_to") is not None and ss.word(S, w["added_to"]) else "")
            + f" · playing {ss._fmt(a)}–{ss._fmt(b)}. Type the correction below (empty = delete).")
    return {"kind": "word", "id": w["id"]}, (ss.edited_text(S, w["id"]) or w["word"]), _clip(S, a, b), info


def _select_sugg(S, sid):
    s = S["suggestions"][sid]
    a, b = max(s["start"] - ss.REPLAY_S, 0), min(s["end"] + ss.REPLAY_S, ss.duration(S))
    info = (f"**Glossary suggestion** ({', '.join(s['terms'])}) at {ss._fmt(s['start'])}: standard "
            f"**{s['standard'] or '(nothing)'}** → suggested **{s['suggested']}** · status {s['status']} · "
            f"playing {ss._fmt(a)}–{ss._fmt(b)}. Listen, then Accept or Reject.")
    return {"kind": "sugg", "id": sid}, s["suggested"], _clip(S, a, b), info


def select_row(S, q, evt: gr.SelectData):
    if S is None or q is None or len(q) == 0 or evt.index is None:
        return None, gr.update(), None, gr.update()
    r = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    ref = str(q.iloc[r]["ref"])
    if ref.startswith("s"):
        return _select_sugg(S, int(ref[1:]))
    w = ss.word(S, int(ref[1:]))
    if w is None:
        return None, gr.update(), None, "That word is no longer in the transcript."
    a, b = max(w["start"] - ss.REPLAY_S, 0), min(w["end"] + ss.REPLAY_S, ss.duration(S))
    info = (f"**{w['word']}** at {ss._fmt(w['start'])} · risk {100 * w['p_error']:.0f} % · playing "
            f"{ss._fmt(a)}–{ss._fmt(b)}. Type the correction below (empty = delete).")
    return {"kind": "word", "id": w["id"]}, (ss.edited_text(S, w["id"]) or w["word"]), _clip(S, a, b), info


def _act(S, sel, thr, fn, need):
    if S is None or not sel or sel["kind"] != need:
        raise gr.Error("Select a word first." if need == "word" else
                       "Select a glossary suggestion first (blue words or 'glossary' rows in the queue).")
    fn()
    hl, ids, q, txt, st = _views(S, thr)
    return S, hl, ids, q, txt, st


def do_replace(S, sel, text, thr):
    return _act(S, sel, thr, lambda: ss.edit_word(S, sel["id"], text), "word")


def do_delete(S, sel, thr):
    return _act(S, sel, thr, lambda: ss.edit_word(S, sel["id"], ""), "word")


def do_revert(S, sel, thr):
    if sel and sel.get("kind") == "sugg":
        return _act(S, sel, thr, lambda: ss.undo(S, sel["id"]), "sugg")
    return _act(S, sel, thr, lambda: ss.revert_word(S, sel["id"]), "word")


def do_accept(S, sel, thr):
    return _act(S, sel, thr, lambda: ss.accept(S, sel["id"]), "sugg")


def do_reject(S, sel, thr):
    return _act(S, sel, thr, lambda: ss.reject(S, sel["id"]), "sugg")


def rethreshold(S, thr):
    hl, ids, q, txt, st = _views(S, thr)
    return hl, ids, q, st


def export(S, thr):
    if S is None or not S["words"]:
        raise gr.Error("Nothing to export yet.")
    d = Path(tempfile.mkdtemp(prefix="g9_live_"))
    (d / "transcript.txt").write_text(ss.final_text(S) + "\n")
    (d / "transcript.srt").write_text(ss.to_srt(S))
    (d / "transcript.json").write_text(ss.to_json(S, thr, A.settings))
    files = [str(d / "transcript.txt"), str(d / "transcript.srt"), str(d / "transcript.json")]
    if S["source"] == "microphone":
        sf.write(d / "recording.wav", ss.audio(S), ss.SR, subtype="PCM_16")
        files.append(str(d / "recording.wav"))
    return files


# ------------------------------------------------------------------------------------------ layout
CSS = ".legend-note{font-size:13px;color:#555}"
with gr.Blocks(title="Live transcription assistant") as demo:
    gr.Markdown("# Live noise-robust transcription assistant\n"
                "Speak into the microphone or upload a recording. Words the model may have got wrong are "
                "highlighted; click any word to hear it (±1 s) and correct it. Add the names and places you expect "
                "in the **glossary**: the app suggests them where the audio supports it, and you decide.")
    S = gr.State(None)
    ids = gr.State([])
    sel = gr.State(None)
    with gr.Row():
        with gr.Column(scale=1, min_width=320):
            gloss = gr.Textbox(label="Glossary (optional): names, places, terms; one per line, max 20", lines=5,
                               placeholder="Neeladri Road\nAniruddha")
            gloss_md = gr.Markdown("No glossary: standard transcription only.", elem_classes="legend-note")
            thr = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Flag sensitivity (risk threshold)",
                            info="Lower = more words highlighted.")
            with gr.Tabs():
                with gr.Tab("🎙 Live"):
                    mic = gr.Audio(sources=["microphone"], streaming=True, type="numpy", label="Microphone")
                    gr.Markdown("Press **Record**: finished sentences appear after each pause; the grey text is a "
                                "preview of what is being said now. Press **Stop** to finish.",
                                elem_classes="legend-note")
                with gr.Tab("📁 Upload"):
                    up = gr.Audio(sources=["upload"], type="filepath", label="Audio file (wav, flac, mp3, …)")
                    run = gr.Button("Transcribe", variant="primary")
                    gr.Examples([[str(EXAMPLES / "librispeech_white_noise_5dB.flac")],
                                 [str(EXAMPLES / "librispeech_clean.flac")]], inputs=[up],
                                label="Examples (LibriSpeech test-other, white noise 5 dB / clean). "
                                      "Try the glossary: Talmash, Lady Vavasour, Aniruddha")
            status = gr.Markdown()
            stats = gr.Markdown()
        with gr.Column(scale=2):
            hl = gr.HighlightedText(label="Transcript: click a word to replay and edit it", color_map=COLORS,
                                    show_legend=True, combine_adjacent=False)
            info = gr.Markdown()
            replay = gr.Audio(label="Replay", autoplay=True, interactive=False)
            with gr.Row():
                edit = gr.Textbox(label="Correction for the selected word", scale=3)
                b_rep = gr.Button("Replace", scale=1)
                b_del = gr.Button("Delete", scale=1)
                b_rev = gr.Button("Revert / undo", scale=1)
    with gr.Row():
        with gr.Column(scale=3):
            queue = gr.Dataframe(headers=ss.QUEUE_COLS, label="Review queue: glossary suggestions and flagged words "
                                 "(click a row to replay it)", interactive=False, wrap=True)
            with gr.Row():
                b_acc = gr.Button("✔ Accept suggestion", variant="primary")
                b_rej = gr.Button("✘ Reject suggestion")
        with gr.Column(scale=2):
            final = gr.Textbox(label="Final transcript (after your edits and accepted suggestions)", lines=8,
                               interactive=False)
            b_exp = gr.Button("Export TXT / SRT / JSON (+ recording)")
            files = gr.File(label="Downloads", file_count="multiple")

    views = [S, hl, ids, queue, final, stats]
    gloss.change(glossary_info, [gloss], [gloss_md])
    mic.start_recording(live_start, [gloss], [S, sel, status])
    mic.stream(live_chunk, [mic, S, gloss, thr], views, stream_every=1.0, time_limit=3 * 3600,
               concurrency_limit=4)
    mic.stop_recording(live_stop, [S, gloss, thr], views + [status])
    run.click(upload_run, [up, gloss, thr], views + [status])
    hl.select(select_word, [S, ids], [sel, edit, replay, info])
    queue.select(select_row, [S, queue], [sel, edit, replay, info])
    b_rep.click(do_replace, [S, sel, edit, thr], views)
    edit.submit(do_replace, [S, sel, edit, thr], views)
    b_del.click(do_delete, [S, sel, thr], views)
    b_rev.click(do_revert, [S, sel, thr], views)
    b_acc.click(do_accept, [S, sel, thr], views)
    b_rej.click(do_reject, [S, sel, thr], views)
    thr.release(rethreshold, [S, thr], [hl, ids, queue, stats])
    b_exp.click(export, [S, thr], [files])

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=4).launch(
        server_name=os.environ.get("APP_HOST", "127.0.0.1"), server_port=int(os.environ.get("APP_PORT", "7861")),
        share=os.environ.get("APP_SHARE", "0") == "1", css=CSS)
