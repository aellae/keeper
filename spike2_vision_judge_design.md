# Spike-2 — selettore basato su giudizio vision-LLM

**Stato: proposta, non ancora implementata.** Scritto dopo aver validato il meccanismo su
due giornate indipendenti (vedi `spike1/` e la memoria di sessione). Non sostituisce
`roadmap-lavoro.md` — è l'analisi che la Fase 2 (G2) chiedeva di fare quando l'approccio
a feature numeriche non generava abbastanza segnale, integrata con quanto scoperto dopo.

---

## 1. Cosa abbiamo imparato (riassunto delle evidenze)

Quattro meccanismi testati, in ordine di forza misurata:

| Meccanismo | Risultato | Verdetto |
|---|---|---|
| Feature tecniche (nitidezza/esposizione/contrasto/volti/unicità CLIP), pesate a mano o imparate (Bradley-Terry) | 16.7% recall, G2 sotto il caso | **non funziona**, in nessuna combinazione — i pesi imparati cambiano perfino segno da un giorno all'altro |
| Modello estetico oggettivo (LAION, consenso umano aggregato) | 8-16.7% recall | **non funziona** — non è "bella secondo tutti", è "significativa per lei" |
| Taste-centroid + volti adattivi + tipicità (informato dal questionario) | 25-33% recall | **funziona un po'** — progresso reale ma insufficiente per G1 (80%) |
| **Giudizio diretto vision-LLM** (guardare le foto e scegliere, con solo comprensione qualitativa accumulata del suo gusto) | **58-82% recall**, due giornate indipendenti | **funziona chiaramente meglio di tutto il resto** |

L'unico meccanismo dell'algoritmo originale che ha retto in ogni test è **MMR come
anti-ridondanza** (penalizzare la somiglianza alle foto GIÀ scelte nella selezione finale) —
diverso dal "premiare la rarità globale", che è la parte che falliva.

**Perché il gradino tra 33% e 58-82% è così grande**: le feature tecniche/CLIP catturano
"quanto è nitida/esposta/rara/simile alla media questa foto". Non catturano niente che
somigli a "questa è la foto del treno con l'amica che sorride, la sfocatura di movimento fa
parte del perché è bella" — un giudizio narrativo/compositivo/emotivo che nessuna feature
hand-crafted o embedding grezzo rappresenta, ma che un modello multimodale con contesto
sufficiente riesce ad approssimare guardando direttamente l'immagine.

**Limite onesto della misura**: il blind test non era pulito al 100% — la lista della
ground truth era già comparsa altrove nella stessa conversazione prima del test. Il numero
esatto (58% vs 82%) non va preso come un tasso di accuratezza validato; va preso come
"il giudizio vision-LLM batte nettamente la pipeline numerica". Prima di fidarsi del numero
per decidere se costruire su questo, serve un test pulito da sessione fresh (§5).

---

## 2. La tensione con la roadmap originale — da decidere esplicitamente

`roadmap-lavoro.md` Fase 4 assume **on-device**: `VNGenerateImageFeaturePrint`, zero
chiamate di rete, zero costo marginale, zero foto che lasciano il telefono. Il meccanismo
che ha appena vinto (giudizio vision-LLM) è strutturalmente il contrario: richiede una
chiamata a un modello multimodale in cloud (Claude o simile), con conseguenze reali:

- **Costo**: non più zero-marginale. Ogni sessione di declutter costa qualcosa (centesimi-
  pochi euro a seconda del volume, vedi stima §4).
- **Rete**: la scansione non può più girare offline in background come previsto in S2/S3.
- **Privacy**: le foto (o almeno le miniature dei candidati finali, dopo il pre-filtro)
  lascerebbero il telefono verso un'API di terzi. Per un'app *personale* che usi solo tu,
  è la stessa cosa che stai facendo in questa conversazione — ma è una scelta diversa da
  "mai nessuna foto lascia il device", che era l'assunzione implicita della spec originale.
- **Latenza**: non istantaneo — una chiamata multimodale su un batch di foto richiede
  secondi, non millisecondi come un embedding on-device.

Questo non è un dettaglio implementativo, è un cambio di natura del prodotto. Tre strade,
non a esclusione reciproca:

**A — Ibrido cloud per il giudizio, on-device per tutto il resto (proposta principale, §3).**
Dedup/burst/pre-filtro restano on-device (già validati, già gratis). Solo il passo finale di
giudizio (poche decine di chiamate per sessione, non una per foto) va in cloud. Costo e
privacy diventano un compromesso esplicito e limitato, non un cambio totale di architettura.

**B — Restare on-device, accettare il tetto del 33%.** La pipeline numerica (taste-centroid
+ tipicità) resta l'unica cosa che gira in Fase 4 così come pianificata. Più debole, ma
zero costo/rete/privacy-tradeoff. Da rivalutare quando/se Apple offre modelli multimodali
on-device abbastanza capaci per questo tipo di giudizio qualitativo (Apple Intelligence
è nella direzione giusta ma non è chiaro se già oggi regge un compito così sfumato — da
verificare con un test dedicato, non da assumere).

**C — Abbassare l'ambizione di automazione, non la qualità del prodotto.** Usare il
burst-clustering (già validato, gratis, on-device) solo per **ridurre il volume di
revisione** 3-5x, poi lasciare la scelta finale a lei con la UI di revisione veloce già
prevista in Fase 5.3 (foto tenuta accanto alle scartate, un tap per gruppo). Rinuncia
all'automazione del *selettore*, non al problema che l'app risolve — coerente con quello
che il G1 stesso prevede come esito "⚠️": *"serve il modello di gusto per colmare il
divario... ma con aspettative ricalibrate"*.

**Raccomandazione**: A come meccanismo principale del prossimo spike (è quello con
l'evidenza più forte e il costo più controllabile), con C sempre disponibile come rete di
sicurezza per la Fase 5 indipendentemente da come va A. B resta un'opzione da rivalutare,
non da scartare.

---

## 3. Pipeline proposta (Opzione A)

```
foto grezze del giorno
        │
        ▼
┌─────────────────────────────────────────┐
│ STAGE 0 — burst clustering (ON-DEVICE)   │  già validato in spike1/step2_cluster.py
│ CLIP/VNFeaturePrint + soglia tempo       │  sim=0.85, riduce ~130 foto → ~50-60 gruppi
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│ STAGE 1 — pre-filtro cheap (ON-DEVICE)   │  taste-centroid mean-centered (step3b/c)
│ top 2-3 per burst per similarità al      │  taglia il volume PRIMA della chiamata cloud
│ centroide + nitidezza minima             │  senza bisogno di essere accurato da solo —
│                                           │  deve solo non scartare il candidato giusto
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│ STAGE 2 — giudizio vision-LLM (CLOUD)    │  ★ meccanismo primario, nuovo
│ chiamata batch multi-immagine per        │  input: 2-3 candidati per burst, RAGGRUPPATI
│ gruppo di burst affini, non 1 per foto   │  per burst (giudizio comparativo, non score
│                                           │  assoluto isolato — coerente con come già
│                                           │  funziona MMR)
│ contesto nel prompt:                     │
│  • 3-5 foto di calibrazione ("queste le  │
│    hai tenute in un'altra giornata")     │
│  • nota di gusto persistente e modifi-   │
│    cabile dall'utente (§3.2, il pezzo    │
│    "AI chat" della richiesta originale)  │
│ output: vincitore per burst + 1 riga di  │
│ motivazione (esplicabilità, spec §5.5)   │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│ STAGE 3 — MMR anti-ridondanza (ON-DEVICE)│  invariato da spike1/step3_select.py —
│ tra vincitori di burst diversi, evita    │  l'UNICO pezzo dell'algoritmo originale
│ selezionare troppe foto simili tra loro  │  sopravvissuto a ogni test
└─────────────────────────────────────────┘
        │
        ▼
   proposta finale (quota N foto)
        │
        ▼
┌─────────────────────────────────────────┐
│ STAGE 4 — refinement conversazionale     │  il pezzo "chat" della richiesta originale,
│ (CLOUD, on-demand, non ogni sessione)    │  ma come loop persistente, non questionario
│ "perché hai tenuto/scartato questa?"     │  one-shot — coerente con quanto emerso dal
│  → risponde con la motivazione salvata   │  test stated-vs-revealed: le preferenze
│    allo Stage 2, di solito senza nuova   │  condizionate al contesto ("dipende dalla
│    chiamata                              │  scena") sono quelle che l'utente sa
│ "tieni sempre i cani anche mossi" ecc.   │  descrivere bene a parole
│  → si aggiunge alla nota di gusto        │
│    persistente usata allo Stage 2 nella  │
│    prossima sessione                     │
└─────────────────────────────────────────┘
```

### 3.1 — perché il giudizio va fatto per gruppi di burst, non foto per foto

Il test blind ha funzionato viste in sequenza con comprensione accumulata del gusto
dell'utente, non foto isolate valutate una a una senza contesto. Praticamente: una singola
chiamata multi-immagine con "questi sono 6 gruppi di foto simili di [giornata], per ognuno
dì qual è il migliore e perché" riproduce meglio quella condizione di una chiamata per foto
— ed è anche l'unico modo per tenere il numero di chiamate (quindi il costo) sotto
controllo su un batch reale.

### 3.2 — la nota di gusto persistente (il pezzo "questionario")

Non un questionario one-shot come step3c, ma un testo libero, modificabile dall'utente
dalla chat, che entra nel prompt dello Stage 2 ad ogni sessione. Esempio concreto basato su
quanto già misurato:

> "Tieni ritratti candidi anche se mossi o sfocati, specialmente con espressioni vere.
> Preferisci lo scatto tipico/rappresentativo di un gruppo, non quello raro o strano.
> I volti contano di più nei giorni di città, quasi nulla nei giorni di natura/tempio.
> Tieni anche foto puramente informative (cartelli, indicazioni) se documentano il
> viaggio, anche se tecnicamente brutte."

L'ultima riga viene direttamente dall'unico miss reale del test Koyasan (la foto del
cartello stradale sfocato, scartata per errore come "non artistica").

---

## 4. Stima di costo (ordine di grandezza, da verificare)

Per una giornata tipo (~130 foto grezze, come Koyasan):
- Stage 0-1 (on-device): gratis, già cronometrato in spike1 (~1-2 min per 130 foto su CPU).
- Stage 2 (cloud): burst tipici ~50-60, con 2-3 candidati ciascuno → si può comprimere in
  poche chiamate multi-immagine batch (es. 5-10 gruppi di burst per chiamata) invece di
  50-60 chiamate singole. Ordine di grandezza: **singole cifre di chiamate per giornata**,
  non centinaia. Il costo esatto dipende dal modello scelto e va misurato con l'API reale,
  non stimato qui — è il primo numero concreto da raccogliere nel prototipo (§5).

---

## 5. Prossimo passo concreto (prima di scrivere Swift)

1. Script Python che chiama l'**API di Claude reale** (non il tool Read interattivo di
   questa conversazione) sui burst pre-filtrati di Koyasan e Osaka, con il prompt di
   Stage 2 sopra descritto.
2. Eseguirlo da una **sessione senza contaminazione** — questa conversazione ha già visto
   la ground truth di entrambi i giorni, quindi non può produrre un numero pulito. Serve
   una chiamata API isolata (script standalone, non questa chat) per un test davvero cieco.
3. Misurare recall@quota, tempo per chiamata, costo reale in dollari.
4. Se il recall pulito resta comodamente sopra il 33% del meccanismo numerico (anche se
   sotto l'82% contaminato) → l'Opzione A è validata, si passa a G3 con questa come
   pipeline di selezione.
5. Se crolla vicino al 33% una volta tolta la contaminazione → il vantaggio era in parte
   un artefatto della fuga di informazione, non del meccanismo in sé. Si ricade
   sull'Opzione C (riduzione volume + revisione umana) senza aver perso il lavoro fatto,
   perché Stage 0-1-3 restano comunque utili come pre-filtro per quella UI.
