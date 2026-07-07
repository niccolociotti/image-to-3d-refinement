# Progetto-CG

API Flask per generare immagini, fare editing guidato da maschera, segmentare immagini con SAM2 e produrre una mesh 3D da una singola immagine. Il progetto è organizzato come un backend che espone endpoint REST e delega il lavoro pesante a servizi separati in `services/`.

## Cosa fa il progetto

Il backend espone quattro funzionalità principali:

- generazione immagine da prompt
- editing/inpainting su immagine con maschera
- segmentazione immagine con SAM2
- generazione 3D da immagine con TRELLIS.2

L’app salva tutti i risultati nella cartella `runs/`, organizzata per sessione.

## Struttura del progetto

- `app.py`: entrypoint Flask e definizione delle rotte
- `services/`: logica dei singoli moduli AI
- `utils/storage.py`: gestione delle directory di output e salvataggio file base64
- `runs/`: output generati durante l’esecuzione
- `models/`: cache e pesi locali
- `checkpoints/`: checkpoint locali, inclusi i pesi SAM2
- `ComfyUI/`: copia locale di ComfyUI usata dai servizi basati su nodi
- `trellis_hf/`: supporto e risorse per TRELLIS.2

## Flusso dell’app

All’avvio, `app.py`:

1. carica una `.env` se presente nella root del progetto
2. configura alcune variabili d’ambiente di default per TRELLIS.2 e Hugging Face
3. crea l’app Flask
4. inizializza i servizi
5. prova a caricare i modelli principali all’avvio

Le risposte includono header CORS aperti, quindi il backend è pensato per essere chiamato anche da una UI separata.

## Rotte disponibili

### `GET /health`

Endpoint di controllo stato.

Risposta tipica:

```json
{"status": "ok"}
```

### `GET /files/<path>`

Espone i file salvati sotto `runs/` tramite URL pubblico.

### `POST /generate-image`

Genera una nuova immagine a partire da un prompt.

Campi principali:

- `prompt` obbligatorio
- `negative_prompt` opzionale
- `width`, `height` opzionali
- `steps`, `cfg`, `seed` opzionali

Output:

- `image_path`
- `image_url`

### `POST /edit-image`

Fa inpainting partendo da un’immagine e da una maschera.

Campi principali:

- `image_path` obbligatorio
- `prompt` obbligatorio
- `mask_path` oppure `mask_base64`
- `negative_prompt` opzionale
- `steps`, `cfg`, `denoise` opzionali
- `grow_mask_by`, `mask_blur`, `mask_threshold` opzionali
- `invert_mask` opzionale

Output:

- `edited_image_path`
- `edited_image_url`

### `POST /segment-image`

Segmenta una zona dell’immagine con SAM2 partendo da un punto.

Campi principali:

- `image_path` oppure `image_base64`
- `points` oppure `x` e `y`
- `box` opzionale
- `multimask_output` opzionale
- `mask_index` opzionale
- `grow_mask_by`, `mask_blur`opzionali
- `invert_mask` opzionale
- `coordinates_normalized` opzionale

Output:

- `mask_path`
- `mask_url`
- `score`
- `selected_mask_index`

### `POST /generate-3d`

Genera un modello 3D da una singola immagine con TRELLIS.2.

Campi principali:

- `image_path` obbligatorio
- `prompt` opzionale
- `session_id` opzionale

Output:

- `model3d_path`
- `model3d_url`

## Flusso di funzionamento

Il flusso tipico è questo:

1. generazione di un’immagine
2. Creazione una maschera con `segment-image`
3. Il risultato viene passato come `mask_path` o `mask_base64` a `edit-image`
4. il servizio di inpaint modifica solo l’area mascherata

Nel progetto attuale l’editing usa `services/edit_service2.py`, che combina ComfyUI e un pipeline di inpainting con maschera precisa e blending finale.

## Come funziona la generazione 3D

`services/model3d_service.py` usa TRELLIS.2.

Il servizio prova a trovare una copia locale di TRELLIS.2 e, se presente, preferisce quella rispetto al download da Hugging Face. Usa anche una cache locale sotto `models/trellis_cache`.

## Requisiti

Le dipendenze principali sono elencate in `requirements.txt`.

## Installazione

Il progetto si avvia dentro un container Docker già predisposto.

Da host:

```bash
docker restart cg2026-gr4-GPU1
docker exec -it cg2026-gr4-GPU1 bash
```

Dentro il container, dalla root del progetto:

```bash
cd /app/Progetto-CG
pip install --upgrade pip
pip install -r requirements.txt
```

Se il container è già configurato con CUDA, non serve creare un virtualenv locale: basta lavorare dentro il container stesso.

## File e cartelle necessari

Devono esistere o essere configurati correttamente:

- `/app/Progetto-CG/checkpoints/sam2.1_hiera_base_plus.pt`
- la copia locale di ComfyUI montata nel container
- la copia locale di TRELLIS.2, per la generazione 3D
- una cartella `/app/Progetto-CG/models/trellis_cache/` per la cache Hugging Face/TRELLIS

## Avvio

Si avvia il server dentro il container, dalla root del progetto:

```bash
cd /app/Progetto-CG
python3 app.py
```

Il server ascolta su `0.0.0.0`

## Esempi di uso con chiamate

### Generazione immagine

```bash
curl -X POST http://192.168.80.138:8081/generate-image \
  -H "Content-Type: application/json" \
  -d '{
  "session_id":"test",
  "prompt":"a blue cup, with blue handle"
  }'
```

### Segmentazione

```bash
curl -X POST http://localhost:8081/segment-image \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "/app/Progetto-CG/runs/test/image.png",
    "points": [{"x": 500, "y": 500, "label": 1}]
  }'
```

### Editing

```bash
curl -X POST http://localhost:8081/edit-image \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "/app/Progetto-CG/runs/test/image.png",
    "mask_path": "/app/Progetto-CG/runs/test/mask.png",
    "prompt": "black handle, mantain the same siza and shape as in the original image"
  }'
```

### Generazione 3D

```bash
curl -X POST http://localhost:8081/generate-3d \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "/app/Progetto-CG/runs/test/image.png",
  }'
```
