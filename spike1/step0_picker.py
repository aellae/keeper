"""
STEP 0 — IL RITUALE DEL SABATO MATTINA

Genera un picker HTML per scegliere a mano le tue 15 foto.
Va fatto PRIMA di scrivere/eseguire qualunque algoritmo.

    python step0_picker.py day_A
    → apri day_A/picker.html, clicca le tue 15, scarica il JSON
    → salvalo come day_A/ground_truth.json
"""
import argparse
import base64
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from lib.common import list_images, load_image

THUMB = 240

HTML = """<!doctype html>
<meta charset="utf-8">
<title>Ground truth — scegli le tue {quota}</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; background:#111; color:#eee;
         margin:0; padding:24px 24px 120px; }}
  h1 {{ font-size:18px; font-weight:600; }}
  p  {{ color:#999; font-size:14px; max-width:60ch; line-height:1.5; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(180px,1fr)); gap:10px; margin-top:24px; }}
  .cell {{ position:relative; cursor:pointer; border-radius:6px; overflow:hidden;
           border:3px solid transparent; transition:border-color .1s, opacity .1s; opacity:.65; }}
  .cell img {{ width:100%; display:block; aspect-ratio:1; object-fit:cover; }}
  .cell.sel {{ border-color:#4ade80; opacity:1; }}
  .cell .n {{ position:absolute; top:6px; left:6px; background:#4ade80; color:#000;
              font-weight:700; font-size:12px; padding:2px 7px; border-radius:99px; display:none; }}
  .cell.sel .n {{ display:block; }}
  .cell .name {{ position:absolute; bottom:0; left:0; right:0; background:rgba(0,0,0,.75);
                 font-size:10px; padding:3px 5px; overflow:hidden; text-overflow:ellipsis;
                 white-space:nowrap; }}
  .bar {{ position:fixed; bottom:0; left:0; right:0; background:#1c1c1c; border-top:1px solid #333;
          padding:16px 24px; display:flex; align-items:center; gap:20px; }}
  .count {{ font-size:22px; font-weight:700; font-variant-numeric:tabular-nums; }}
  .count.ok {{ color:#4ade80; }}
  button {{ background:#4ade80; color:#000; border:0; padding:10px 20px; border-radius:6px;
            font-weight:600; font-size:14px; cursor:pointer; }}
  button:disabled {{ background:#444; color:#888; cursor:not-allowed; }}
  .hint {{ color:#888; font-size:13px; }}
</style>

<h1>Ground truth — {folder}</h1>
<p>Scegli le foto che <b>terresti davvero</b>. Non le più belle in astratto: quelle che
riguarderesti. Quota suggerita: <b>{quota}</b> (puoi sgarrare di poco).</p>
<p><b>Non aprire i risultati dell'algoritmo prima di aver finito qui.</b></p>

<div class="grid" id="g"></div>

<div class="bar">
  <span class="count" id="c">0</span>
  <span class="hint">selezionate su {n} foto · quota {quota}</span>
  <button id="b" disabled>Scarica ground_truth.json</button>
  <span class="hint">poi salvalo in <code>{folder}/ground_truth.json</code></span>
</div>

<script>
const FILES = {files};
const QUOTA = {quota};
const THUMBS = {thumbs};
const sel = [];
const g = document.getElementById('g');

FILES.forEach((f, i) => {{
  const d = document.createElement('div');
  d.className = 'cell';
  d.innerHTML = `<img src="${{THUMBS[i]}}"><span class="n"></span><span class="name">${{f}}</span>`;
  d.onclick = () => {{
    const k = sel.indexOf(f);
    if (k >= 0) sel.splice(k, 1); else sel.push(f);
    render();
  }};
  d.dataset.f = f;
  g.appendChild(d);
}});

function render() {{
  document.querySelectorAll('.cell').forEach(c => {{
    const k = sel.indexOf(c.dataset.f);
    c.classList.toggle('sel', k >= 0);
    if (k >= 0) c.querySelector('.n').textContent = k + 1;
  }});
  const c = document.getElementById('c');
  c.textContent = sel.length;
  c.classList.toggle('ok', sel.length === QUOTA);
  document.getElementById('b').disabled = sel.length === 0;
}}

document.getElementById('b').onclick = () => {{
  const blob = new Blob(
    [JSON.stringify({{folder: "{folder}", quota: QUOTA, selected: sel}}, null, 2)],
    {{type: 'application/json'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'ground_truth.json';
  a.click();
}};
</script>
"""


def thumb_data_uri(path):
    img = load_image(path, max_side=THUMB)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=72)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--quota", type=int, default=15)
    a = ap.parse_args()

    files = list_images(a.folder)
    if not files:
        raise SystemExit(f"[stop] nessuna immagine in {a.folder}/")

    print(f"[i] {len(files)} foto — genero le miniature…")
    thumbs = [thumb_data_uri(f) for f in files]

    import json
    html = HTML.format(
        folder=a.folder,
        quota=a.quota,
        n=len(files),
        files=json.dumps([f.name for f in files]),
        thumbs=json.dumps(thumbs),
    )
    out = Path(a.folder) / "picker.html"
    out.write_text(html)

    size_mb = out.stat().st_size / 1e6
    print(f"[ok] {out}  ({size_mb:.1f} MB)")
    print()
    print("    Aprilo nel browser, scegli le tue foto, scarica il JSON,")
    print(f"    e salvalo come {a.folder}/ground_truth.json")
    print()
    print("    ⚠️  Fallo PRIMA di eseguire gli altri step.")


if __name__ == "__main__":
    main()
