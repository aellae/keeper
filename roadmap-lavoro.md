# Photo Curation — Roadmap di lavoro
### Piano operativo · da lunedì 13 luglio 2026

---

## Come è costruita questa roadmap

**È guidata dai cancelli, non dalle date.** Ogni fase termina con una domanda a risposta binaria e un **criterio di kill scritto prima di iniziarla**. Se il cancello non si apre, la fase successiva **non parte** — si torna indietro o si chiude il progetto. Le date sono indicative; i cancelli no.

**Capacità assunta: ~8 ore/settimana** (due serate + mezza giornata di weekend). È una stima onesta, non ottimistica. Se in qualche settimana è zero, la roadmap slitta e non succede niente: **nessuna fase ha una scadenza esterna.** Questo progetto non compete con nient'altro nella tua vita, ed è importante che resti così — un side project con una deadline è solo un secondo lavoro.

**Regola d'oro**: nessuna fase produce codice destinato a sopravvivere finché il cancello precedente non si è aperto.

---

## Struttura a colpo d'occhio

```
G0 ──► G1 ──► G2 ──► G3 ──► G4 ──► [PAUSA 1 MESE] ──► decisione
 │      │      │      │      │
 │      │      │      │      └─ hai cancellato 1.500 foto senza panico
 │      │      │      └──────── l'app gira e la scansione regge
 │      │      └─────────────── il gusto generalizza
 │      └────────────────────── il selettore sceglie come te
 └───────────────────────────── lo vuoi ancora costruire
```

---

## FASE 0 — La prova del nove
### 🗓️ Stasera · ⏱️ 20 minuti · 💻 zero codice

| # | Task | Output |
|---|---|---|
| 0.1 | OldRoll → esporta nel rullino ciò che vuoi tenere | — |
| 0.2 | Cerca in impostazioni un *clear cache* / *elimina foto sviluppate*. Se non c'è: esporta tutto e reinstalla l'app | **~6 GB liberati** |
| 0.3 | Installa **Slidebox** o **Swipewipe**. Usalo 20 minuti. **Annota cosa ti fa incazzare del flusso** | `competitor-notes.md` |
| 0.4 | Screenshot di Impostazioni → Spazio iPhone, prima e dopo | baseline reale |

### 🚪 CANCELLO G0
> **Con 6 GB liberi e il problema di spazio risolto, vuoi ancora costruirla?**

- ✅ **Sì, perché voglio ritrovare le 60 foto belle del Giappone senza guardarne 2.000** → prosegui
- ❌ **No, mi bastava lo spazio** → **chiudi qui.** Hai risparmiato sei mesi con venti minuti. È il miglior esito possibile di questa serata.

---

## FASE 1 — Spike-1: il selettore
### 🗓️ Weekend 18-19 luglio · ⏱️ 1 weekend · 💻 **Python, non Swift**

> **Perché Python**: lo spike deve rispondere a *"l'algoritmo sceglie come me?"*, non a *"so usare PhotoKit?"* — quello lo sai già. In Swift passeresti il sabato a combattere con `PHImageManager` e la domanda vera resterebbe senza risposta. **Il codice di questa fase si butta. È previsto.**

### Sabato mattina — PRIMA di scrivere codice ⚠️

| # | Task | Output |
|---|---|---|
| 1.1 | Esporta su disco **una giornata del Giappone** (~150 foto). Scegline una densa e difficile: molte foto, luoghi ripetuti, burst | `/day_A/` |
| 1.2 | **Scegli a mano le tue 15.** Guardale, decidi, salva i nomi file, **e non riaprire il file** | `ground_truth_A.json` |

> **L'ordine non è negoziabile.** Se scrivi prima l'algoritmo, riconoscerai come "giuste" le foto che ha scelto lui. Ti autoinganneresti senza accorgertene: è il modo standard in cui gli spike falliscono.

### Sabato pomeriggio + domenica

| # | Task | Output |
|---|---|---|
| 1.3 | Embedding di tutte le foto (CLIP o simili, via `open_clip` / `transformers`) | `embeddings.npy` |
| 1.4 | Feature di qualità: laplacian variance, esposizione, volti (face detection base) | `features.csv` |
| 1.5 | Clustering: near-duplicate (soglia + tempo) → gruppi | `clusters.json` |
| 1.6 | **Selettore con vincolo di diversità** (greedy MMR, §4.2 della spec) | `select.py` |
| 1.7 | **Calcolo del recall@15** contro `ground_truth_A.json` | **un numero** |
| 1.8 | Sweep di **λ** (0 → 1) e curva del recall | grafico |

### 🚪 CANCELLO G1 — *criterio fissato ORA, prima di vedere il numero*

| recall@15 | Verdetto |
|---|---|
| **≥ 12/15** | ✅ il selettore vive → **G2** |
| **8–11/15** | ⚠️ serve il modello di gusto per colmare il divario → **G2**, ma con aspettative ricalibrate |
| **≤ 7/15** | ❌ **l'approccio è sbagliato.** Non passare a G2. Torna alle feature, o chiudi. |

**Bonus gratis**: la curva di λ ti dice se la varietà conta davvero, e quale valore usare come default.

---

## FASE 2 — Spike-2: il gusto generalizza?
### 🗓️ Weekend 25-26 luglio · ⏱️ 1 weekend · 💻 Python

| # | Task | Output |
|---|---|---|
| 2.1 | Esporta un **giorno B di natura diversa** (una cena al chiuso, non un tempio). Scegli a mano le tue preferite anche lì | `/day_B/`, `ground_truth_B.json` |
| 2.2 | Estrai le **coppie intra-cluster** dal giorno A (`foto_3 > foto_1`, ecc.) | `pairs_A.json` |
| 2.3 | Addestra un **Bradley-Terry / logistic regression sulla differenza dei feature vector** | `taste_model.pkl` |
| 2.4 | **Test sulle coppie del giorno B**: data una coppia dello stesso cluster, indovina quale hai tenuto? | **accuratezza su B** |
| 2.5 | ⚠️ **CONTROLLO**: riaddestra usando **etichette binarie** *tengo/butto* invece delle coppie. Confronta su B | **accuratezza binarie su B** |

### 🚪 CANCELLO G2 — *doppia condizione*

| Condizione | Soglia |
|---|---|
| **A** — accuratezza coppie sul giorno B | **≥ 75%** |
| **B** — **le coppie devono BATTERE le binarie** | `acc_coppie > acc_binarie` |

- ✅ **Entrambe** → il modello a due livelli funziona. Vai a G3.
- ⚠️ **A sì, B no** → funziona, ma la tesi delle coppie intra-cluster è inutile: **semplifica**, usa le binarie e risparmia complessità.
- ❌ **A no (~50%)** → **stai tirando a caso.** Hai imparato Kyoto, non l'utente. Ripensa le feature prima di costruirci sopra.

> Questo spike è progettato **per falsificare**, non per confermare. Il task 2.5 esiste apposta per potermi dare torto. Un esperimento che non può fallire non è un esperimento.

---

## FASE 3 — Fondamenta
### 🗓️ ~4 settimane (agosto-inizio settembre) · 💻 **Swift, e da qui il codice resta**

Prima riga di codice destinata a sopravvivere. Non prima.

| Settimana | Contenuto | Done quando |
|---|---|---|
| **S1** | Scheletro app · permessi PhotoKit · SQLite (GRDB) · schema `Asset`/`Cluster`/`Proposal`/`Card` | l'app enumera la libreria e la scrive su DB |
| **S2** | **Ingestion a batch**: resumable, stato persistito, niente in RAM · BackgroundTasks | 40k asset scansionati **senza crash**, riprende dopo kill |
| **S3** | **Console + job queue**: catalogo operazioni, esecuzione background, progresso, interruzione | avvii un job, chiudi l'app, riapri: riprende |
| **S4** | ⭐ **AUDIT ONESTO** (§3 spec): misura, rileva iCloud ottimizzato, **dice la verità**, sceglie la modalità | la schermata ti dice, con numeri veri, che libererai 0,4 GB sul telefono |

### 🚪 CANCELLO G3
> **La scansione completa della tua libreria gira in background senza crash, in un tempo accettabile, e l'audit dice la verità.**

Se qui si rompe (memoria, tempi, batteria), **si ferma tutto**: nessuna feature sopra fondamenta che non reggono. È la fase meno divertente e la più importante.

---

## FASE 4 — Il selettore in app
### 🗓️ ~3 settimane · 💻 Swift

| # | Contenuto |
|---|---|
| 4.1 | Porting del selettore validato in Fase 1 · embedding via `VNGenerateImageFeaturePrint` |
| 4.2 | Clustering per evento (DBSCAN su tempo+GPS) → *"Kyoto, 14 giugno"* |
| 4.3 | Quota per evento + slider λ |
| 4.4 | **Output: un album vero** in Foto, esportabile |
| 4.5 | Guardrail §5.4 + spiegabilità §5.5 (una riga di motivo per ogni scelta) |

### 🚪 CANCELLO G4a — *soggettivo ma reale*
> **Fai girare l'app sul Giappone. Guardi l'album delle 60. È un album che rivedresti davvero?**

Non è una metrica, è un giudizio. Ma è **il giudizio per cui esiste il prodotto**, quindi vale più di qualunque numero. Se l'album è tecnicamente corretto e emotivamente vuoto, qualcosa nelle feature è sbagliato (§4.3: volti e unicità sopra l'estetica).

---

## FASE 5 — La cancellazione
### 🗓️ ~3 settimane · 💻 Swift · ⚠️ **prima fase distruttiva**

**L'ordine interno è vincolante** (§4.1): niente cancellazione prima che il salvataggio sia consolidato e verificato.

| # | Contenuto | Nota |
|---|---|---|
| 5.1 | **Set "protette"** — screenshot con card attive, documenti, scontrini, persone rare | **protette ≠ non selezionate.** Il bug più costoso possibile |
| 5.2 | 👻 **Archivio fantasma** — miniatura 200px + metadati + OCR per ogni foto condannata (2.000 ≈ 35 MB) | **è ciò che rende la cancellazione psicologicamente possibile** |
| 5.3 | **Review per gruppi** — la foto tenuta accanto alle scartate, un tap per gruppo | 1.940 foto ≈ 150 gruppi ≈ 20 minuti |
| 5.4 | Export originali su Mac/disco — **la domanda si fa una volta, prima della prima cancellazione di massa** | |
| 5.5 | **Doppia conferma** → `PHPhotoLibrary` batch delete | una conferma di sistema per batch |
| 5.6 | Cestino interno 30 giorni | seconda rete sotto quella di sistema |

### 🚪 CANCELLO G4
> **Hai cancellato 1.500 foto del Giappone. Il giorno dopo, hai un rimpianto?**

- ✅ **No** → il prodotto funziona. È il momento in cui esiste davvero.
- ❌ **Sì, e mi manca X** → **fermati e capisci perché il guardrail non l'ha protetta.** Non aggiungere feature finché non lo sai.

---

## FASE 6 — Il modello di gusto
### 🗓️ ~3 settimane · 💻 Swift + Accelerate

Arriva **dopo** la review, non prima — perché è la review che produce le etichette, gratis.

| # | Contenuto |
|---|---|
| 6.1 | Raccolta **coppie intra-cluster** da ogni sessione di review (nessuna domanda in più all'utente) |
| 6.2 | **Modello globale** (persiste) + **adattamento di sessione** (effimero, muore col batch) |
| 6.3 | **Active learning**: mostra i gruppi su cui il modello è più incerto, non 5 a caso |
| 6.4 | 🔴 **Esplorazione forzata** (ε = 5-10%) contro la bolla |
| 6.5 | 🔴 **Soglie asimmetriche permanenti** — confidenza per *cancellare* ≫ confidenza per *tenere*. **Non cala mai col tempo.** |
| 6.6 | **Accordo misurato** e mostrato all'utente → autonomia progressiva, revocabile |

> **Le righe 6.4 e 6.5 sono quelle che tra sei mesi sarai tentata di ammorbidire perché "ormai è bravo".** Non farlo. Il fallimento peggiore del prodotto arriva *dopo* il successo.

---

## FASE 7 — Screenshot
### 🗓️ ~4-5 settimane

| # | Contenuto |
|---|---|
| 7.1 | OCR (`VNRecognizeTextRequest`, IT/EN/JA) + `NSDataDetector` + barcode |
| 7.2 | Classificazione categoria + **Card tipizzate** |
| 7.3 | ⏳ **Scadenze** (`expiresAt`) → "Scaduti (47)" con cancella-tutto |
| 7.4 | 🔍 **Ricerca full-text** su `rawText` ← *forse la feature che userai di più* |
| 7.5 | Azioni: Calendario · Mappe · Note · CSV |
| 7.6 | Collezioni: *Viaggio Giappone 2026*, *Da provare a Roma*, *Ricette* |

---

## ⏸️ PAUSA OBBLIGATORIA — 1 mese
### Non è una fase. È il punto in cui **smetti di scrivere codice e usi l'app.**

Un mese di uso reale, tutti i giorni, senza toccare l'IDE. È l'unico modo per sapere se ciò che resta serve davvero, ed è **la fase che salterai se non te la scrivi in roadmap**. Per questo è scritta in roadmap.

Alla fine del mese, tre domande:

1. **La uso ancora?** Se no → il prodotto è finito, e va bene così. Ha risolto il tuo problema.
2. **Cosa mi manca davvero?** La lista che scrivi *dopo* un mese d'uso vale dieci volte quella scritta prima.
3. **Voglio darla ad altri?** → apre la Fase 8. Altrimenti **hai finito**, e hai un'app che usi tu. Che era l'obiettivo.

---

## FASE 8 — Solo se la pausa dice sì
### 🗓️ 4-6 settimane + raddoppio per lo store

| # | Contenuto | Vincolo |
|---|---|---|
| 8.1 | **Modalità Spazio**: purge tecnica, video (trim/scene/audio), transcodifica HEVC | ⚠️ **richiede un tester esterno** — non puoi validarla su di te (§11 spec) |
| 8.2 | Best-of / racconto dell'evento | |
| 8.3 | Store: onboarding, edge case, privacy policy, empty state, localizzazione | **× 2 su tutto** |

> **Prerequisito bloccante di 8.1: aver trovato il tester del segmento Spazio.** Senza di lui, la costruiresti al buio — ed è la parte distruttiva. **Un solo tester in quel segmento vale sei settimane di supposizioni.**

---

## Riepilogo tempi

| | Cumulativo |
|---|---|
| G0 (vuoi ancora?) | **stasera** |
| G1 + G2 (gli spike) | **2 weekend** |
| G3 (fondamenta) | + 4 settimane |
| G4a (l'album funziona) | + 3 settimane |
| **G4 (cancelli senza rimpianti)** | **≈ 3 mesi** ← *qui l'app risolve il tuo problema* |
| + gusto + screenshot | ≈ 5 mesi |
| Pausa di un mese | ≈ 6 mesi |
| Store, se mai | 8-10 mesi |

**Il traguardo vero è G4, a ~3 mesi.** Tutto il resto è opzionale, e la roadmap è costruita perché tu possa fermarti lì senza che sia una sconfitta.

---

## Le tre cose da non fare

1. **Non saltare la ground truth del sabato mattina.** È l'unico punto in cui puoi ancora essere onesta con te stessa.
2. **Non riusare il codice degli spike.** È scritto per morire; portarselo dietro significa portarsi dietro tutte le scorciatoie che avevi il permesso di prendere.
3. **Non ammorbidire le soglie asimmetriche** quando il modello diventa bravo. Sarà proprio allora che sembrerà ragionevole farlo.

---

## Le tre cose da tracciare

| Metrica | Dove | Perché |
|---|---|---|
| **recall@15** | spike-1, poi in-app | dice se il selettore sceglie come te |
| **accordo modello/utente** | ogni sessione di review | è la moneta con cui l'app compra autonomia |
| **rimpianti** (n. di foto che ti sono mancate) | a mano, onestamente | **l'unica metrica che può uccidere il prodotto** |

La terza non ha una dashboard. Va tenuta a mano, ed è la più importante.
