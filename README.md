# CleanPhone

App per liberare spazio sul telefono e ritrovare le foto belle senza scorrerne 2.000 — un selettore con vincolo di diversità (non un ranker) sceglie un album, non i 60 tramonti migliori.

Il progetto è guidato da cancelli (gate), non da date: ogni fase ha un criterio di kill scritto **prima** di vederne il risultato. Se un cancello non si apre, la fase dopo non parte.

## Struttura

| File / cartella | Cosa contiene |
|---|---|
| [`roadmap-lavoro.md`](roadmap-lavoro.md) | Piano operativo fase per fase (G0 → G4 → store), con i criteri di ogni cancello |
| [`photo-curation-spec.pdf`](photo-curation-spec.pdf) | Spec di prodotto — algoritmo di selezione, audit dello spazio, guardrail di cancellazione, modello di gusto |
| [`spike1/`](spike1/) | Toolkit Python usa-e-getta per gli spike 1 e 2: valida "il selettore sceglie come te?" e "il gusto generalizza?" prima di scrivere una riga di Swift |

Le cartelle `Photo*/` (foto dei viaggi, output generati dalla pipeline: `features.npz`, `picker.html`, ecc.) sono **gitignored** — dati personali e binari pesanti, mai destinati al version control.

## Setup

```bash
cd spike1
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Nota: `opencv-python` è pinnato a `<5` in `requirements.txt` — la 5.x ha rimosso `cv2.CascadeClassifier` usato per la face detection in `step1_features.py`. Usare Python 3.11 (non 3.13+): alcune dipendenze (torch, opencv) non hanno ancora wheel stabili per le versioni più recenti.

## Stato — spike-1 e spike-2 completati, entrambi i cancelli chiusi

Eseguita la pipeline completa su due giornate reali indipendenti (Koyasan, Osaka):

- **G1** (selettore, `spike1/step0-4`): recall@15 identico su entrambe le giornate, **2/12 (16.7%)**. La diversità (MMR) non ha mai aiutato. 0% degli errori erano "gruppo giusto, foto sbagliata" — quasi tutti erano gruppi mai considerati.
- **G2** (modello di gusto, `spike1/step5-7`): testato in entrambe le direzioni (train Koyasan→test Osaka e viceversa), **chiuso, sotto il caso** (accuracy 28–44% contro 50% atteso dal random). I pesi imparati cambiano segno da un giorno all'altro (es. nitidezza: −0.93 su Koyasan, +0.48 su Osaka) — non è un problema di quantità di dati, le feature attuali (nitidezza, esposizione, contrasto, volti, embedding CLIP) semplicemente non predicono cosa questa utente tiene.

**Conclusione**: non continuare a cercare pesi/feature migliori sullo stesso vocabolario — testato a sufficienza, converge sempre allo stesso esito nullo. Il clustering (burst/evento) invece funziona bene una volta tarata la soglia (`--sim 0.85`, non 0.92). La direzione più promettente discussa: spostare la Fase 4 da "selezione automatica" a "review assistita per gruppi" (Fase 5.3 della roadmap) — il clustering fa il lavoro pesante, l'utente sceglie con un tap per gruppo, senza bisogno che l'algoritmo indovini il gusto.

## Regola d'oro

Nessuna fase produce codice destinato a sopravvivere finché il cancello precedente non si è aperto.
