"""
STEP 5 — Giudizio vision-LLM (spike-2, Opzione A di ../spike2_vision_judge_design.md)

Sostituisce lo score numerico di step3 (nitidezza/esposizione/volti/unicità — dimostrato
non predittivo su due giornate, vedi memoria di sessione) con un giudizio diretto: MOSTRA
le foto di ogni burst a un modello multimodale (Claude) e chiedigli quale terrebbe questa
utente specifica, calibrato con foto che ha già tenuto in un'altra giornata.

Pipeline:
  STAGE 0  burst clustering    — già fatto da step2_cluster.py (clusters.json)
  STAGE 1  pre-filtro cheap    — taste-centroid mean-centered, top-K per burst (locale,
                                  gratis, riusa step3b/step3c). NON filtra per nitidezza:
                                  è stato misurato ~zero correlato ai keep reali, non ha
                                  senso reintrodurlo qui come filtro.
  STAGE 2  giudizio vision-LLM — QUESTO SCRIPT. Chiamate reali all'API Claude, a batch
                                  (più burst per chiamata, non 1 per foto) per tenere
                                  sotto controllo costo e numero di richieste.
  STAGE 3  MMR anti-ridondanza — riusa mmr() di step3_select, sui VINCITORI di burst,
                                  con score = similarità al centroide del vincitore
                                  (il giudizio vision-LLM sceglie IL RAPPRESENTANTE del
                                  gruppo, non decide se il gruppo entra in quota — quello
                                  resta al centroide + MMR, così i punteggi restano
                                  confrontabili tra chiamate API diverse).

Perché è un test pulito (a differenza del blind test fatto a mano in chat): ogni chiamata
API è stateless e parte senza alcun contesto — il modello che giudica qui non ha MAI visto
la ground truth del giorno di test, a differenza di questa conversazione.

Nota sul λ: come in step4/step3b/step3c, il recall riportato è al MIGLIOR λ trovato via
sweep contro la ground truth — è la convenzione di misura già usata in tutto il progetto
per restare confrontabile con i numeri precedenti (16.7% / 25% / 33%), non un meccanismo
che l'app userebbe in produzione (lì servirebbe un λ fisso, scelto una volta sola).

Serve ANTHROPIC_API_KEY nell'ambiente. Non viene mai letta, stampata o salvata da questo
script se non implicitamente dall'SDK ufficiale.

    # anteprima gratis: quanti burst, quante immagini, stima approssimativa dei token
    python step5_vision_judge.py --train ../PhotoOsaka --test ../PhotoKoyasan --dry-run

    # run reale, prima su pochi burst per farsi un'idea del costo
    python step5_vision_judge.py --train ../PhotoOsaka --test ../PhotoKoyasan \
        --quota 15 --max-bursts 5

    # run completo
    python step5_vision_judge.py --train ../PhotoOsaka --test ../PhotoKoyasan --quota 15
"""
import argparse
import base64
import json
import sys
import time
from io import BytesIO
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from lib.common import out_path, load_ground_truth, load_image, normalize
from step3_select import mmr
from step3b_taste_centroid import load_day, taste_centroid
from step3c_questionnaire_informed import center
from step4_evaluate import recall

DEFAULT_MODEL = "claude-sonnet-5"
MAX_TOKENS = 4096

DEFAULT_TASTE_NOTE = """\
Tieni ritratti candidi anche se mossi o sfocati, specialmente con espressioni vere.
Preferisci lo scatto tipico/rappresentativo di un momento, non quello raro o strano.
I volti contano di più nei giorni di città, quasi nulla nei giorni di natura/tempio.
Tieni anche foto puramente informative (cartelli, indicazioni) se documentano il
viaggio, anche se tecnicamente brutte — non scartarle solo perché "non sono arte"."""

SYSTEM_PROMPT = """Aiuti a scegliere, per ogni gruppo di foto quasi-duplicate (stesso \
momento, scatti multipli), qual è quella che questa specifica persona terrebbe nel suo \
archivio personale — non la "più bella" in astratto, la più significativa per LEI.

Ti mostro prima alcuni esempi di foto che ha già scelto di tenere in un'altra giornata, \
per calibrarti sul suo gusto. Poi una nota scritta da lei su cosa le importa. Poi i \
gruppi da giudicare.

Per ogni gruppo scegli UNA foto (il nome file esatto mostrato) e scrivi una riga sul \
perché — sarà mostrata a lei come spiegazione della scelta, quindi scrivi per lei, non \
per un log tecnico. Usa la funzione record_picks per rispondere, un elemento per gruppo."""

TOOL_SCHEMA = {
    "name": "record_picks",
    "description": "Registra la foto scelta per ciascun gruppo mostrato.",
    "input_schema": {
        "type": "object",
        "properties": {
            "picks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "burst_id": {"type": "string"},
                        "chosen_filename": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["burst_id", "chosen_filename", "reason"],
                },
            }
        },
        "required": ["picks"],
    },
}


def to_jpeg_b64(path, max_side):
    img = load_image(path, max_side=max_side)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def image_block(path, max_side, cache=False):
    block = {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/jpeg",
                   "data": to_jpeg_b64(path, max_side)},
    }
    if cache:
        block["cache_control"] = {"type": "ephemeral"}
    return block


def est_tokens(path, max_side):
    """Stima Anthropic: tokens ≈ (larghezza_px × altezza_px) / 750, sull'immagine ridotta."""
    img = load_image(path, max_side=max_side)
    return (img.width * img.height) / 750.0


def prefilter_bursts(test_folder, test_names, test_embs_c, bursts, centroid, k):
    """Per ogni burst, i top-k membri per similarità al centroide. Ritorna:
    { burst_id: [(filename, sim), ...] } ordinato per sim decrescente."""
    name_idx = {n: i for i, n in enumerate(test_names)}
    sims = normalize(test_embs_c @ centroid)
    out = {}
    for bid, members in bursts.items():
        ranked = sorted(members, key=lambda n: sims[name_idx[n]], reverse=True)
        out[bid] = [(n, float(sims[name_idx[n]])) for n in ranked[:max(k, 1)]]
    return out


def build_calibration_blocks(train_folder, train_truth, n_calib, max_side, seed):
    rng = np.random.RandomState(seed)
    pool = sorted(train_truth)
    chosen = [str(x) for x in rng.choice(pool, size=min(n_calib, len(pool)), replace=False)]
    blocks = [{"type": "text", "text":
               "ESEMPI — foto che questa persona ha GIÀ scelto di tenere in un'altra giornata:"}]
    for name in chosen:
        p = Path(train_folder) / name
        if not p.exists():
            continue
        blocks.append({"type": "text", "text": f"(esempio tenuto: {name})"})
        blocks.append(image_block(p, max_side))
    return blocks, chosen


def build_batch_blocks(test_folder, batch, max_side):
    """batch: lista di (burst_id, [(filename, sim), ...]). Ritorna content blocks."""
    blocks = [{"type": "text", "text":
               f"\nGRUPPI DA GIUDICARE ({len(batch)} in questa richiesta):"}]
    for bid, cands in batch:
        blocks.append({"type": "text", "text":
                        f"\n--- GRUPPO {bid} — scegli 1 tra {len(cands)} foto ---"})
        for name, _ in cands:
            p = Path(test_folder) / name
            blocks.append({"type": "text", "text": f"Foto: {name}"})
            blocks.append(image_block(p, max_side))
    return blocks


def call_batch(client, model, calib_blocks, batch, test_folder, max_side, max_retries=2):
    content = list(calib_blocks)
    content.append({"type": "text", "text": f"\nNOTA DI GUSTO SCRITTA DALL'UTENTE:\n{DEFAULT_TASTE_NOTE}",
                     "cache_control": {"type": "ephemeral"}})
    content += build_batch_blocks(test_folder, batch, max_side)

    last_err = None
    for attempt in range(max_retries + 1):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=MAX_TOKENS,
                system=[{"type": "text", "text": SYSTEM_PROMPT,
                         "cache_control": {"type": "ephemeral"}}],
                tools=[TOOL_SCHEMA],
                tool_choice={"type": "tool", "name": "record_picks"},
                messages=[{"role": "user", "content": content}],
            )
            for block in resp.content:
                if block.type == "tool_use" and block.name == "record_picks":
                    return block.input.get("picks", []), resp.usage
            return [], resp.usage
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(2 ** attempt)
    print(f"[!!] batch fallito dopo {max_retries + 1} tentativi: {last_err}")
    return [], None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="giorno da cui prendere calibrazione + centroide")
    ap.add_argument("--train-truth", help="JSON alternativo per il centroide (default: ground_truth.json del train)")
    ap.add_argument("--test", required=True, help="giorno da giudicare, MAI visto dal modello prima")
    ap.add_argument("--quota", type=int, default=15)
    ap.add_argument("--prefilter-k", type=int, default=2, help="candidati per burst mostrati al modello")
    ap.add_argument("--batch-size", type=int, default=6, help="burst per chiamata API")
    ap.add_argument("--n-calibration", type=int, default=4)
    ap.add_argument("--max-side", type=int, default=1024, help="ridimensionamento immagini prima dell'invio")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-bursts", type=int, default=None, help="limita a N burst multi-foto, per un run economico di prova")
    ap.add_argument("--dry-run", action="store_true", help="non chiama l'API: mostra piano e stima token")
    a = ap.parse_args()

    train_names, train_embs, _ = load_day(a.train)
    train_truth = (set(json.loads(Path(a.train_truth).read_text())["selected"])
                   if a.train_truth else load_ground_truth(a.train))
    train_embs_c = center(train_embs)
    centroid = taste_centroid(train_names, train_embs_c, train_truth)

    test_names, test_embs, test_bursts = load_day(a.test)
    test_truth = load_ground_truth(a.test)
    test_embs_c = center(test_embs)
    name_idx = {n: i for i, n in enumerate(test_names)}

    prefiltered = prefilter_bursts(a.test, test_names, test_embs_c, test_bursts, centroid, a.prefilter_k)

    # burst singoli: vince l'unica foto, nessun giudizio necessario
    solo_winners = {bid: cands[0] for bid, cands in prefiltered.items() if len(cands) == 1}
    to_judge = {bid: cands for bid, cands in prefiltered.items() if len(cands) > 1}
    if a.max_bursts is not None:
        to_judge = dict(list(to_judge.items())[:a.max_bursts])

    print(f"[i] {len(test_names)} foto → {len(test_bursts)} burst "
          f"({len(solo_winners)} da 1 foto, {len(to_judge)} da giudicare)")

    batches = list(to_judge.items())
    batches = [batches[i:i + a.batch_size] for i in range(0, len(batches), a.batch_size)]

    if a.dry_run:
        total_candidate_imgs = sum(len(c) for c in to_judge.values())
        sample_paths = [Path(a.test) / c[0][0] for c in list(to_judge.values())[:1]]
        est_per_img = est_tokens(sample_paths[0], a.max_side) if sample_paths else 0
        total_imgs = a.n_calibration + total_candidate_imgs
        print(f"[dry-run] {len(batches)} chiamate API previste")
        print(f"[dry-run] {total_candidate_imgs} immagini candidate + "
              f"{a.n_calibration} di calibrazione (ripetute per chiamata, ma cache_control "
              f"le fa pagare piene solo alla prima chiamata) = ~{total_imgs} immagini totali")
        print(f"[dry-run] stima ~{est_per_img:.0f} token/immagine a max-side={a.max_side} "
              f"→ ordine di grandezza ~{total_imgs * est_per_img:.0f} token immagine, "
              f"più testo. Controlla il prezzo per-token corrente di {a.model} prima di lanciare.")
        return

    import anthropic
    client = anthropic.Anthropic()  # legge ANTHROPIC_API_KEY dall'ambiente

    calib_blocks, calib_names = build_calibration_blocks(a.train, train_truth, a.n_calibration, a.max_side, a.seed)
    print(f"[i] calibrazione: {calib_names}")

    winners = dict(solo_winners)  # bid -> (filename, sim)
    reasons = {}
    total_usage = {"input_tokens": 0, "output_tokens": 0,
                   "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}

    for batch in tqdm(batches, desc="giudizio vision-LLM"):
        picks, usage = call_batch(client, a.model, calib_blocks, batch, a.test, a.max_side)
        if usage:
            for k in total_usage:
                total_usage[k] += getattr(usage, k, 0) or 0

        cand_by_bid = dict(batch)
        seen = set()
        for p in picks:
            bid, chosen, reason = p.get("burst_id"), p.get("chosen_filename"), p.get("reason", "")
            cands = cand_by_bid.get(bid)
            if cands is None:
                continue
            valid = {n for n, _ in cands}
            if chosen not in valid:
                print(f"[!!] burst {bid}: il modello ha risposto '{chosen}', non tra i candidati {sorted(valid)} — uso il top pre-filtro")
                chosen = cands[0][0]
                reason = "(fallback: risposta non valida, tenuto il migliore per taste-centroid)"
            sim = dict(cands)[chosen]
            winners[bid] = (chosen, sim)
            reasons[bid] = reason
            seen.add(bid)

        for bid, cands in batch:
            if bid not in seen:
                winners[bid] = cands[0]
                reasons[bid] = "(fallback: chiamata API fallita o burst non risposto, tenuto il top pre-filtro)"

    print(f"[i] token usati: {total_usage}")

    # Stage 3 — MMR sui soli vincitori di burst
    win_names = [w[0] for w in winners.values()]
    win_scores = np.array([w[1] for w in winners.values()])
    win_idx = [name_idx[n] for n in win_names]
    win_embs = test_embs_c[win_idx]

    lams = np.linspace(0.0, 1.0, 21)
    curve = []
    for lam in lams:
        sel = [win_names[i] for i in mmr(win_scores, win_embs, a.quota, float(lam))]
        r, h = recall(sel, test_truth)
        curve.append((float(lam), r, h, sel))
    best_lam, best_r, best_h, best_sel = max(curve, key=lambda x: x[1])

    bar = "─" * 62
    print(f"\n{bar}")
    print(f"  VISION-LLM JUDGE   train={a.train}   test={a.test}   model={a.model}")
    print(f"  ground truth test: {len(test_truth)} foto · quota: {a.quota}")
    print(bar)
    print(f"  ▶ MMR + giudizio vision-LLM (λ={best_lam:.2f})   recall  {best_r:5.1%}   ({best_h}/{len(test_truth)})")
    print(bar)

    out_path(a.test, "result_vision_judge.json").write_text(json.dumps({
        "train": a.train, "test": a.test, "model": a.model, "quota": a.quota,
        "prefilter_k": a.prefilter_k, "calibration": calib_names,
        "best_lambda": best_lam, "recall": best_r, "hits": best_h,
        "selected": best_sel, "truth": sorted(test_truth),
        "burst_winners": {bid: {"chosen": w[0], "reason": reasons.get(bid, "(burst da 1 foto)")}
                           for bid, w in winners.items()},
        "token_usage": total_usage,
    }, indent=1))
    print(f"[ok] {a.test}/result_vision_judge.json\n")


if __name__ == "__main__":
    main()
