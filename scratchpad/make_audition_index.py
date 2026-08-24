#!/usr/bin/env python3
"""Build a browsable, markable page for every film's voice audition.

Beyond listening, this is the cast-decision surface: every voice can be marked
Approve / Change (with a reason + free note), decisions persist in localStorage
so a reload does not lose them, and "Save decisions" writes a JSON that the dub
pipeline can consume to regenerate per-movie tuning.

  python make_audition_index.py   ->  G:\\Transcode\\voice-auditions\\index.html
"""
from __future__ import annotations

import csv
import html
import json
import os

AUD = r"G:\Transcode\voice-auditions"
OUT = os.path.join(AUD, "index.html")

# what each flag means, in the reviewer's terms
GLOSSARY = [
    ("speech", "How much ORIGINAL dialogue this speaker has across the whole film. "
               "This is the size of the role - the top row is the lead."),
    ("spk NN", "A diarization cluster, not a named character. The pipeline groups similar "
               "voices; one spk is normally one actor, but a big spk with far more speech "
               "than the rest is usually several people merged together."),
    ("bucket / gender", "A pitch-based guess (adult_male, child_female, ...). COSMETIC ONLY - "
                        "cloning copies the real voice, so a wrong label does not affect the dub."),
    ("clips", "Reference clips found on disk / listed in the profile. The clone's timbre is "
              "taken from these, so they decide what the voice sounds like."),
    ("emo", "emo_alpha - how strongly the ORIGINAL line's emotion is pushed onto the clone. "
            "0.45 is the default. Lower = calmer and steadier; higher = more expressive but "
            "strains and can distort."),
    ("rms dB", "Loudness of the rendered sample in dBFS (0 = maximum). About -15 to -20 sits "
               "comfortably; -12 or louder is flagged HOT."),
]
FLAGS = [
    ("ok", "ok", "Nothing wrong detected."),
    ("HOT", "hot", "Very hot delivery (RMS >= -12 dBFS). Not distorted, but often reads as "
                   "shouty or harsh. Fix: lower emo_alpha to ~0.30-0.35."),
    ("CLIPPED", "clip", "Genuinely flat-topped - at least 0.1% of samples pinned at full scale, "
                        "which is audible distortion. Fix: lower emo_alpha, or pick a different "
                        "reference clip."),
    ("LOW_LEVEL", "none", "The clone came out too quiet. Fix: raise gain_db, or the reference "
                          "clip is weak."),
    ("SILENT_CLONE", "none", "Rendered essentially silent - the reference clip is unusable."),
    ("NO_TIMBRE", "none", "No reference clip on disk at all. This speaker would render SILENT "
                          "in the finished dub. Must be fixed before rendering."),
]
REASONS = ["too shouty / harsh", "distorted", "too quiet", "wrong age or gender feel",
           "sounds like several people", "flat / no emotion", "just wrong for the character",
           "use a different reference clip"]

CSS = """
:root{--bg:#12141a;--card:#1b1e26;--line:#2a2f3a;--tx:#e8eaf0;--dim:#98a0b3;
--ok:#3fb950;--hot:#d29922;--clip:#f85149;--none:#a371f7;--acc:#4493f8}
*{box-sizing:border-box}
body{margin:0;padding:24px 24px 80px;background:var(--bg);color:var(--tx);
font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif}
h1{font-size:20px;margin:0 0 4px}h2{font-size:15px;margin:0 0 8px}
.sub{color:var(--dim);margin-bottom:16px}
.panel{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:14px 16px;margin-bottom:14px}
.gl{display:grid;grid-template-columns:150px 1fr;gap:6px 14px;font-size:13px}
.gl b{color:var(--acc);font-weight:600}
.fl{display:grid;grid-template-columns:130px 1fr;gap:6px 14px;font-size:13px;margin-top:10px;
padding-top:10px;border-top:1px solid var(--line)}
details{background:var(--card);border:1px solid var(--line);border-radius:10px;
margin-bottom:12px;overflow:hidden}
summary{padding:12px 16px;cursor:pointer;display:flex;gap:10px;align-items:center;
flex-wrap:wrap;font-weight:600}
summary::-webkit-details-marker{display:none}
summary:hover{background:#20242e}
.count{color:var(--dim);font-weight:400}
.pill{font:600 11px/1 sans-serif;padding:4px 7px;border-radius:20px}
.p-ok{background:rgba(63,185,80,.15);color:var(--ok)}
.p-hot{background:rgba(210,153,34,.15);color:var(--hot)}
.p-clip{background:rgba(248,81,73,.15);color:var(--clip)}
.p-none{background:rgba(163,113,247,.15);color:var(--none)}
.body{padding:0 16px 16px}
.sheet{padding:10px 0 14px;border-bottom:1px solid var(--line);margin-bottom:10px}
.sheet b{display:block;margin-bottom:6px}
audio{height:32px;vertical-align:middle}
table{width:100%;border-collapse:collapse}
th{text-align:left;color:var(--dim);font-weight:500;font-size:12px;
padding:6px 8px;border-bottom:1px solid var(--line)}
td{padding:5px 8px;border-bottom:1px solid rgba(42,47,58,.5);white-space:nowrap}
td.n{text-align:right;font-variant-numeric:tabular-nums}
tr:hover{background:#20242e}
tr.appr{background:rgba(63,185,80,.07)}tr.chg{background:rgba(248,81,73,.07)}
.flag{font:600 11px/1 sans-serif}
.f-CLIPPED{color:var(--clip)}.f-HOT{color:var(--hot)}
.f-NO_TIMBRE,.f-SILENT_CLONE,.f-LOW_LEVEL{color:var(--none)}
.f-ok{color:var(--dim)}
button{font:inherit;padding:3px 9px;border-radius:6px;border:1px solid var(--line);
background:#232833;color:var(--tx);cursor:pointer}
button:hover{border-color:var(--acc)}
button.on-a{background:var(--ok);border-color:var(--ok);color:#07230f;font-weight:600}
button.on-c{background:var(--clip);border-color:var(--clip);color:#2a0605;font-weight:600}
select,input[type=text]{font:inherit;background:#232833;color:var(--tx);
border:1px solid var(--line);border-radius:6px;padding:3px 6px}
input[type=text]{width:190px}
.bar{position:fixed;left:0;right:0;bottom:0;background:#0d0f14;border-top:1px solid var(--line);
padding:10px 24px;display:flex;gap:14px;align-items:center;z-index:9}
.bar b{color:var(--ok)}.bar i{color:var(--clip);font-style:normal}
.grow{flex:1}
"""

JS = """
const KEY='dubAuditionDecisions_v1';
let D=JSON.parse(localStorage.getItem(KEY)||'{}');
function k(f,s){return f+'|'+s}
function save(){localStorage.setItem(KEY,JSON.stringify(D));tally()}
function tally(){
  let a=0,c=0;for(const v of Object.values(D)){if(v.d==='approve')a++;if(v.d==='change')c++}
  document.getElementById('na').textContent=a;
  document.getElementById('nc').textContent=c;
  document.getElementById('nt').textContent=document.querySelectorAll('tr[data-k]').length;
}
function paint(tr,v){
  tr.classList.toggle('appr',v&&v.d==='approve');
  tr.classList.toggle('chg',v&&v.d==='change');
  tr.querySelector('.bA').classList.toggle('on-a',!!v&&v.d==='approve');
  tr.querySelector('.bC').classList.toggle('on-c',!!v&&v.d==='change');
  const sel=tr.querySelector('.rsn'),note=tr.querySelector('.note');
  const show=!!v&&v.d==='change';
  sel.style.visibility=show?'visible':'hidden';
  note.style.visibility=show?'visible':'hidden';
  if(v){sel.value=v.reason||'';note.value=v.note||''}
}
function mark(tr,d){
  const key=tr.dataset.k;
  const cur=D[key];
  if(cur&&cur.d===d){delete D[key]}
  else{D[key]=Object.assign({},cur,{d:d,film:tr.dataset.film,spk:tr.dataset.spk,
        reason:(cur&&cur.reason)||'',note:(cur&&cur.note)||''})}
  paint(tr,D[key]);save();
}
function setField(tr,f,val){
  const key=tr.dataset.k;
  if(!D[key])D[key]={d:'change',film:tr.dataset.film,spk:tr.dataset.spk};
  D[key][f]=val;paint(tr,D[key]);save();
}
function dl(){
  const rows=Object.entries(D).map(([key,v])=>({film:v.film,speaker:v.spk,decision:v.d,
      reason:v.reason||'',note:v.note||''}));
  rows.sort((x,y)=>x.film.localeCompare(y.film)||x.speaker-y.speaker);
  const blob=new Blob([JSON.stringify({saved:new Date().toISOString(),decisions:rows},null,1)],
    {type:'application/json'});
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);
  a.download='audition_decisions.json';a.click();
}
function clearAll(){if(confirm('Clear every mark?')){D={};localStorage.removeItem(KEY);
  document.querySelectorAll('tr[data-k]').forEach(tr=>paint(tr,null));tally()}}
function onlyFlagged(cb){
  document.querySelectorAll('tr[data-k]').forEach(tr=>{
    tr.style.display=(cb.checked&&tr.dataset.flags==='ok')?'none':''})
}
window.addEventListener('DOMContentLoaded',()=>{
  document.querySelectorAll('tr[data-k]').forEach(tr=>paint(tr,D[tr.dataset.k]));
  tally();
});
"""


def mmss(x):
    x = int(float(x or 0))
    return f"{x//60}:{x%60:02d}"


def main():
    films = []
    for d in sorted(os.listdir(AUD)):
        p = os.path.join(AUD, d)
        csvp = os.path.join(p, "voice_samples.csv")
        if not os.path.isdir(p) or not os.path.exists(csvp):
            continue
        with open(csvp, encoding="utf-8", errors="replace") as f:
            rows = list(csv.DictReader(f))
        rows.sort(key=lambda r: -float(r.get("total_speech_sec") or 0))
        films.append((d, rows))

    P = [f"<!doctype html><meta charset='utf-8'><title>Voice auditions</title>"
         f"<style>{CSS}</style><script>{JS}</script>"]
    tot = sum(len(r) for _, r in films)
    P.append(f"<h1>Voice auditions</h1><div class='sub'>{len(films)} films, {tot} voices. "
             "Mark each voice <b>Approve</b> or <b>Change</b>; marks are saved in this browser "
             "as you go. When done, hit <b>Save decisions</b> and the pipeline will apply them.</div>")

    # glossary
    P.append("<div class='panel'><h2>What the columns mean</h2><div class='gl'>")
    for term, desc in GLOSSARY:
        P.append(f"<b>{html.escape(term)}</b><span>{html.escape(desc)}</span>")
    P.append("</div><div class='fl'>")
    for name, cls, desc in FLAGS:
        P.append(f"<b class='flag f-{name}'>{name}</b><span>{html.escape(desc)}</span>")
    P.append("</div></div>")

    # A/B comparison folders (name starts with "_") - alternative takes of one
    # voice, rendered to choose between them. Shown first, they are the decision.
    for d in sorted(os.listdir(AUD)):
        p = os.path.join(AUD, d)
        if not d.startswith("_") or not os.path.isdir(p):
            continue
        wavs = sorted(w for w in os.listdir(p) if w.endswith(".wav"))
        if not wavs:
            continue
        title = html.escape(d.lstrip("_").replace("_", " "))
        P.append(f"<div class='panel'><h2>Compare: {title}</h2>"
                 "<div class='sub' style='margin:0 0 10px'>Alternative takes of the same role. "
                 "Play them back to back and keep the one that sounds right.</div>")
        for w in wavs:
            P.append(f"<div style='margin-bottom:6px'><code>{html.escape(os.path.splitext(w)[0])}</code><br>"
                     f"<audio controls preload='none' src='{html.escape(d)}/{html.escape(w)}'></audio></div>")
        P.append("</div>")

    for d, rows in films:
        nclip = sum(1 for r in rows if "CLIPPED" in (r.get("flags") or ""))
        nhot = sum(1 for r in rows if "HOT" in (r.get("flags") or ""))
        nnone = sum(1 for r in rows if r.get("has_timbre", "True") == "False")
        pills = []
        if nnone:
            pills.append(f"<span class='pill p-none'>{nnone} NO TIMBRE</span>")
        if nclip:
            pills.append(f"<span class='pill p-clip'>{nclip} clipped</span>")
        if nhot:
            pills.append(f"<span class='pill p-hot'>{nhot} hot</span>")
        if not pills:
            pills.append("<span class='pill p-ok'>clean</span>")
        name = html.escape(d.replace("_", " ").strip())
        P.append(f"<details><summary>{name} <span class='count'>{len(rows)} voices</span> "
                 + " ".join(pills) + "</summary><div class='body'>")
        if os.path.exists(os.path.join(AUD, d, "_contact_sheet.wav")):
            P.append(f"<div class='sheet'><b>Contact sheet &mdash; every voice back to back</b>"
                     f"<audio controls preload='none' src='{html.escape(d)}/_contact_sheet.wav'></audio></div>")
        P.append("<table><tr><th>spk</th><th>bucket</th><th class='n'>speech</th>"
                 "<th class='n'>emo</th><th class='n'>rms dB</th><th>flags</th>"
                 "<th>listen</th><th>decision</th><th>if changing</th></tr>")
        for r in rows:
            fl = r.get("flags") or "ok"
            cls = "f-" + (fl.split(",")[0] if fl else "ok")
            wav = r.get("out_file") or ""
            spk = int(r["id"])
            key = f"{d}|{spk}"
            player = (f"<audio controls preload='none' src='{html.escape(d)}/{html.escape(wav)}'></audio>"
                      if wav else "<span class='f-ok'>&mdash;</span>")
            opts = "".join(f"<option value='{html.escape(x)}'>{html.escape(x)}</option>"
                           for x in [""] + REASONS)
            P.append(
                f"<tr data-k='{html.escape(key)}' data-film='{html.escape(d)}' data-spk='{spk}' "
                f"data-flags='{html.escape(fl)}'>"
                f"<td>spk{spk:02d}</td><td>{html.escape(r.get('bucket') or '?')}</td>"
                f"<td class='n'>{mmss(r.get('total_speech_sec'))}</td>"
                f"<td class='n'>{r.get('emo_alpha') or ''}</td>"
                f"<td class='n'>{r.get('raw_rms_dbfs') or ''}</td>"
                f"<td class='flag {cls}'>{html.escape(fl)}</td><td>{player}</td>"
                f"<td><button class='bA' onclick=\"mark(this.closest('tr'),'approve')\">Approve</button> "
                f"<button class='bC' onclick=\"mark(this.closest('tr'),'change')\">Change</button></td>"
                f"<td><select class='rsn' onchange=\"setField(this.closest('tr'),'reason',this.value)\">{opts}</select> "
                f"<input class='note' type='text' placeholder='note (optional)' "
                f"oninput=\"setField(this.closest('tr'),'note',this.value)\"></td></tr>")
        P.append("</table></div></details>")

    P.append("<div class='bar'><span><b id='na'>0</b> approved &nbsp; "
             "<i id='nc'>0</i> to change &nbsp; <span class='count'>of <span id='nt'>0</span></span></span>"
             "<label class='count'><input type='checkbox' onchange='onlyFlagged(this)'> "
             "only show flagged</label><span class='grow'></span>"
             "<button onclick='clearAll()'>Clear all</button>"
             "<button onclick='dl()' style='border-color:var(--acc)'>Save decisions</button></div>")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(P))
    print(f"{len(films)} films, {tot} voices -> {OUT}")


if __name__ == "__main__":
    main()
