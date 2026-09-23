# Prima demo: sketch → SDXL nell'Unity Editor

Questa fase produce un'immagine 2D a partire da uno sketch. La generazione 3D
verrà collegata dopo che il servizio SDXL sarà stato verificato.

## Server

1. Preparare ComfyUI e `SDXL/app_sdxl.py` seguendo
   [la guida del progetto di Suemi e Sofia](https://github.com/Sobnt/sketch-to-3d-sdxl-trellis2/blob/main/REPRODUCIBILITY.md).
   Servono SDXL Base, ControlNet Scribble e i nodi indicati nella guida.
2. Provare lo sketch dalla pagina web del loro server (`http://127.0.0.1:5001`)
   prima di aprire Unity. Il server può trovarsi su un'altra macchina, ma in
   quel caso occorre usare il suo IP di rete nel componente Unity.

## Scena `Model3D.unity`

1. Creare un materiale **Unlit**, bianco, senza trasparenza, da assegnare alle
   linee. `LineRenderer.startColor/endColor` sono neri, quindi usare un materiale
   che rispetti il vertex color; in alternativa usare direttamente un materiale
   Unlit nero. Evitare shader Lit, la luce della scena altera il controllo.
2. Creare un GameObject vuoto `Editor Sketch` e aggiungere
   `EditorSketchToImage`. Assegnare `Main Camera` a **View Camera**, il materiale
   a **Stroke Material**, il campo prompt esistente a **Prompt Input** e una
   `RawImage` a **Generated Image**. Si può usare la RawImage già nella scena,
   sapendo che lo sketch sostituirà la texture attualmente mostrata.
3. Aggiungere due Button alla Canvas, **Genera da sketch** e **Cancella sketch**,
   e assegnarli rispettivamente a **Generate Button** e **Clear Button** del
   componente. Lo script registra da solo i listener all'avvio; non aggiungere
   ulteriori OnClick nell'Inspector.
4. Lasciare `Sketch Layer = 30` solo se il layer 30 è libero. Le linee sono
   isolate per acquisire un PNG con sfondo bianco. La camera di acquisizione
   temporanea vede soltanto quel layer.
5. In **SDXL Url** indicare `http://127.0.0.1:5001/generate` quando SDXL gira
   sullo stesso computer di Unity; altrimenti usare l'IP del server.

Entrare in Play Mode. Disegnare nell'area davanti alla camera tenendo premuto
il pulsante sinistro del mouse; ogni rilascio conclude un tratto. Scrivere il
prompt e premere **Genera da sketch**. L'immagine restituita compare nella
RawImage. **Cancella sketch** elimina i tratti. Lo script espone anche
`BeginStroke`, `AddPoint` ed `EndStroke` per un futuro adattatore ai controller.

Se il click del mouse viene catturato dallo XR Device Simulator, disattivare il
simulatore durante questa prova o assegnargli controlli diversi. Il disegno
ignora i click sopra gli elementi UI. Per disegnare sulla stessa prospettiva
del visore e rispettare le dimensioni fisiche servirà una fase successiva:
questa prima acquisizione è una proiezione 2D dei tratti su un piano fisso.
