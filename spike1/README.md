# Spike-1 — Il selettore sceglie come te?

**Time-box: 1 weekend. Il codice si butta. L'output è una risposta, non un'app.**

Domanda: *dato un giorno di foto, il selettore con vincolo di diversità sceglie le stesse foto che sceglieresti tu?*

Criterio deciso **prima** di vedere il numero:

| recall@15 | Verdetto |
|---|---|
| **≥ 12/15** | ✅ il selettore vive |
| **8–11/15** | ⚠️ serve il modello di gusto (→ spike-2) |
| **≤ 7/15** | ❌ approccio sbagliato. Fermati. |

---

## ⚠️ IL RITUALE DEL SABATO MATTINA

**Non saltarlo, o l'intero spike è inutile.**

Devi scegliere le tue 15 foto **prima** che l'algoritmo esista. Se guardi prima il suo output, riconoscerai come "giuste" le foto che ha scelto lui, e ti autoingannerai senza accorgertene.

```
1. Esporta una giornata del Giappone in ./day_A/     (~150 foto)
2. python step0_picker.py day_A
3. Apri picker.html nel browser. Scegli le tue 15. Scarica il JSON.
4. Salvalo come ./day_A/ground_truth.json
5. NON RIAPRIRLO fino alla fine.
```

Solo dopo, procedi.

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Le foto iPhone sono HEIC: `pillow-heif` è già nei requirements e viene registrato in automatico.
Se preferisci, esporta da Foto come JPEG ("Esporta" → "Esporta N foto" → JPEG, con metadati).

**Esporta con i metadati**: servono `DateTimeOriginal` e GPS per il clustering.

---

## Pipeline

```bash
python step0_picker.py    day_A                # → picker.html  (SABATO MATTINA)
python step1_features.py  day_A                # → features.npz (embedding + qualità + volti)
python step2_cluster.py   day_A                # → clusters.json
python step3_select.py    day_A --quota 15     # → selection.json
python step4_evaluate.py  day_A --quota 15     # → IL NUMERO + lambda_sweep.png
```

---

## Cosa ti dice `step4`

Non solo il recall. Anche **la diagnosi dei due modi di sbagliare**, che è la parte più utile:

| Errore | Significato | Cosa fare |
|---|---|---|
| **Gruppo mancato** — l'algoritmo non ha scelto nulla dal gruppo dove tu avevi scelto | le **priorità sui gruppi** sono sbagliate | rivedi le feature: volti? unicità? |
| **Gruppo giusto, foto sbagliata** — ha scelto dal gruppo giusto, ma un altro scatto | i gruppi sono giusti, manca il **gusto** | ✅ **spike-2 è esattamente questo.** Buona notizia. |

Se i tuoi errori sono quasi tutti del secondo tipo, il selettore funziona e il modello di gusto (§5.3 della spec) è la cura corretta. Se sono del primo, il problema è più a monte e spike-2 non ti salverà.

---

## `favorites.txt` (opzionale ma consigliato)

Il segnale comportamentale è il più onesto che esista, ma non sopravvive all'export. Se hai dei preferiti in quella giornata, mettili qui, uno per riga:

```
IMG_4471.HEIC
IMG_4502.HEIC
```

Lo score li peserà. Se il file non c'è, viene ignorato.

---

## Baseline (le trovi in step4)

Il recall da solo non significa niente senza confronto:

- **random** — il pavimento
- **solo qualità (λ=0)** — il ranker ingenuo. **Se il selettore non lo batte, il vincolo di diversità è inutile e la §4.2 della spec è sbagliata.**
- **uno per gruppo** — baseline stupida ma sorprendentemente forte
- **MMR (λ ottimale)** — la tesi

**Il confronto che conta è contro "solo qualità".** È lì che si gioca l'intera idea.

---

# Spike-2 — Il gusto generalizza?

Domanda: *le coppie intra-cluster del giorno A prevedono le tue scelte sul giorno B (mai visto, di natura diversa)?*

## Prerequisito

Il **giorno B** deve passare per la stessa pipeline dello spike-1, **compresa la scelta a mano**:

```bash
python step0_picker.py   day_B        # scegli anche qui, PRIMA di tutto
python step1_features.py day_B
python step2_cluster.py  day_B
```

Scegli un giorno B di **natura diversa** dal giorno A: una cena al chiuso, non un altro tempio. È il punto dell'esperimento (§5.3.2: il modello deve aver imparato te, non Kyoto).

## Pipeline

```bash
python step5_pairs.py       day_A     # coppie intra-cluster dal giorno A
python step5_pairs.py       day_B     # coppie del giorno B (per la valutazione)
python step6_train_taste.py day_A     # Bradley-Terry + controllo binario
python step7_eval_taste.py  day_A day_B    # IL NUMERO + cancello G2
```

## Il modello

```
u(x)      = w · f(x)                                (utilità di una foto)
P(a > b)  = sigmoid( w · (f(a) − f(b)) )            (Bradley-Terry)
```

Logistic regression **sulla differenza** dei feature vector, **senza intercetta**
(l'antisimmetria è strutturale). Feature = [nitidezza, esposizione, contrasto,
volti] standardizzate + embedding CLIP ridotto a 12 dim via PCA (poche coppie,
512 dim = overfitting garantito). Ogni coppia entra due volte, simmetrizzata.

**Perché le coppie intra-cluster**: dentro un cluster il contenuto è costante,
quindi la differenza f(a) − f(b) **cancella le componenti di contenuto** e isola
lo stile — che è l'unica cosa che generalizza a un giorno diverso (§5.3.3).

## Il controllo (task 2.5 — quello che può darmi torto)

`step6` addestra anche un classificatore **binario** tengo/butto per foto, stesse
identiche feature. `step7` li confronta sul giorno B. **Se le coppie non battono
le binarie, la tesi §5.3.3 è sbagliata** e conviene il modello più semplice.

## Cancello G2 (doppia condizione, fissata prima di guardare)

| Condizione | Soglia |
|---|---|
| A — accuracy coppie sul giorno B | ≥ 75% |
| B — coppie **>** binarie | strettamente |

✅ entrambe → modello a due livelli · ⚠️ solo A → semplifica, usa le binarie · ❌ ~50% → hai imparato Kyoto, non te

## Validazione della macchina (fatta su dati sintetici)

| Scenario | Binarie | Coppie | Gate |
|---|---|---|---|
| Stile fortissimo, zero confondenti | 100% | 100% | ⚠️ parziale (pari: coppie inutili) — onesto |
| Rumore che domina lo stile | 60% | 56% | ❌ chiuso — onesto |
| **Realistico** (stile netto, rumore moderato) | 88% | **92%** | ✅ **aperto: le coppie battono le binarie** |

Il gate discrimina correttamente in tutti e tre i regimi. Il numero vero, come sempre, esiste solo sulle tue foto.
