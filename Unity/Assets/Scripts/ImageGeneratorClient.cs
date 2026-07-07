using System.Collections;
using System.Text;
using System.Threading.Tasks; // Necessario per async/await di glTFast
using TMPro;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.UI;
using GLTFast; // La libreria per caricare il modello 3D

public class ImageGeneratorClient : MonoBehaviour
{
    [Header("UI Elements - Immagine")]
    public TMP_InputField promptInput;
    public Button generateImageButton; 

    public Button resetMaskButton;
    public RawImage resultImage;

    public ImageMaskPainter maskPainter;

    public RawImage maskOverlayImage;

    [Header("UI Elements - 3D")]
    public Button generate3DButton; 

    [Header("Visualizzazione Modello 3D")]
    [Min(0.01f)] public float modelSize = 0.35f;
    [Min(0f)] public float modelGapFromImage = 0.12f;
    public float modelDepthOffset = 0.15f;

    [Header("Impostazioni Server")]
    public string serverUrlImage = "http://192.168.80.138:8081/generate-image";
    public string serverUrlEditImage = "http://192.168.80.138:8081/edit-image";
    public string serverUrlSegmentImage = "http://192.168.80.138:8081/segment-image";
    public string serverUrl3D = "http://192.168.80.138:8081/generate-3d"; 
    public string sessionId = "test";

    [Header("Maschera SAM2")]
    public bool useSam2Masking = true;
    public string sam2SelectionMode = "part";
    [Min(0)] public int sam2GrowMaskBy = 0;
    [Min(0f)] public float sam2MaskBlur = 0f;

    // Variabile per memorizzare il path dell'immagine generata
    private string lastImagePath = ""; 
    private GameObject currentModelContainer;
    private bool isSegmentingMask;

    #region Classi Dati JSON
    [System.Serializable]
    private class GenerateRequest
    {
        public string session_id;
        public string prompt;
    }

    [System.Serializable]
    private class GenerateResponse
    {
        public string image_path;
        public string image_url;
        public string status;
    }

    [System.Serializable]
    private class Generate3DRequest
    {
        public string session_id;
        public string image_path;
    }

    [System.Serializable]
    private class Generate3DResponse
    {
        public string model3d_url; 
        public string model3d_path; 
        public string status;
    }
    #endregion

    void Start()
    {
        // Imposta i listener per i bottoni
        generateImageButton.onClick.AddListener(OnGenerateImageClicked);
        
        if (generate3DButton != null)
        {
            generate3DButton.onClick.AddListener(OnGenerate3DClicked);
            // Disabilita il bottone 3D finché non abbiamo generato un'immagine
            generate3DButton.interactable = false; 
        }

        if (resetMaskButton != null)
        {
            resetMaskButton.onClick.AddListener(OnResetMaskClicked);
            resetMaskButton.gameObject.SetActive(false);
        }

        if (maskPainter != null)
        {
            maskPainter.PointSelected += OnMaskPointSelected;
            maskPainter.brushEnabled = !useSam2Masking;
        }
    }

    void OnResetMaskClicked()
    {
        if (maskPainter != null)
            maskPainter.ClearMask();
    }

    #region Fase 1: Generazione Immagine (Z-Image Turbo)
    void OnGenerateImageClicked()
    {
        string prompt = promptInput.text.Trim();

        if (string.IsNullOrEmpty(prompt))
        {
            Debug.LogWarning("Prompt vuoto. Inserisci un testo.");
            return;
        }

        // Evita click multipli
        generateImageButton.interactable = false; 
        if (generate3DButton != null) generate3DButton.interactable = false;

        StartCoroutine(GenerateImage(prompt));
    }

    IEnumerator GenerateImage(string prompt)
    {
        GenerateRequest body = new GenerateRequest
        {
            session_id = sessionId,
            prompt = prompt
        };

        string json = JsonUtility.ToJson(body);
        byte[] jsonBytes = Encoding.UTF8.GetBytes(json);

        using (UnityWebRequest request = new UnityWebRequest(serverUrlImage, "POST"))
        {
            request.uploadHandler = new UploadHandlerRaw(jsonBytes);
            request.downloadHandler = new DownloadHandlerBuffer();
            request.SetRequestHeader("Content-Type", "application/json");

            yield return request.SendWebRequest();

            if (request.result != UnityWebRequest.Result.Success)
            {
                Debug.LogError("Errore POST Immagine: " + request.error);
                generateImageButton.interactable = true;
                RestoreGenerate3DButton();
                yield break;
            }

            GenerateResponse response = JsonUtility.FromJson<GenerateResponse>(request.downloadHandler.text);

            if (response == null || response.status != "ok" || string.IsNullOrEmpty(response.image_url))
            {
                Debug.LogError("Risposta Immagine non valida: " + request.downloadHandler.text);
                generateImageButton.interactable = true;
                RestoreGenerate3DButton();
                yield break;
            }

            // Salviamo il path dell'immagine per passarlo a Trellis successivamente
            lastImagePath = response.image_path;

            yield return StartCoroutine(DownloadAndShowImage(response.image_url));

            if (resetMaskButton != null)
            {
                resetMaskButton.gameObject.SetActive(true);
            }

            // L'immagine è pronta, possiamo sbloccare il bottone per il 3D
            if (generate3DButton != null) generate3DButton.interactable = true;
        }

        generateImageButton.interactable = true;
    }

    IEnumerator DownloadAndShowImage(string imageUrl)
    {
        using (UnityWebRequest request = UnityWebRequestTexture.GetTexture(imageUrl))
        {
            yield return request.SendWebRequest();

            if (request.result != UnityWebRequest.Result.Success)
            {
                Debug.LogError("Errore download immagine: " + request.error);
                yield break;
            }

            Texture2D texture = DownloadHandlerTexture.GetContent(request);
            resultImage.texture = texture;

            maskPainter.Initialize(maskOverlayImage, texture.width, texture.height);
            maskPainter.brushEnabled = !useSam2Masking;
        }
    }
    #endregion

    #region Maschera SAM2
    void OnMaskPointSelected(Vector2Int unityPixel)
    {
        if (!useSam2Masking)
        {
            return;
        }

        if (isSegmentingMask)
        {
            Debug.Log("SAM2 è già in corso, attendo il risultato della maschera precedente.");
            return;
        }

        if (string.IsNullOrEmpty(lastImagePath) || resultImage == null || resultImage.texture == null)
        {
            Debug.LogWarning("Nessuna immagine disponibile per generare la maschera SAM2.");
            return;
        }

        float sam2Y = resultImage.texture.height - 1 - unityPixel.y;
        Vector2 sam2Point = new Vector2(unityPixel.x, sam2Y);
        StartCoroutine(GenerateSam2Mask(sam2Point));
    }

    IEnumerator GenerateSam2Mask(Vector2 sam2Point)
    {
        isSegmentingMask = true;
        if (generate3DButton != null)
        {
            generate3DButton.interactable = false;
        }

        SAM2SegmentationClient.SegmentResult segmentResult = null;
        yield return StartCoroutine(SAM2SegmentationClient.SegmentImage(
            serverUrlSegmentImage,
            sessionId,
            lastImagePath,
            sam2Point,
            result => segmentResult = result,
            sam2SelectionMode,
            sam2GrowMaskBy,
            sam2MaskBlur));

        isSegmentingMask = false;

        if (segmentResult == null || !segmentResult.success)
        {
            Debug.LogError(segmentResult != null ? segmentResult.error : "La richiesta segment-image non ha restituito un risultato.");
            RestoreGenerate3DButton();
            yield break;
        }

        maskPainter.SetMaskFromPng(segmentResult.maskPng);
        Debug.Log($"Maschera SAM2 pronta. Score={segmentResult.score:0.000}, index={segmentResult.selectedMaskIndex}");
        RestoreGenerate3DButton();
    }
    #endregion

    #region Fase 2: Generazione Modello 3D (Trellis)
    void OnGenerate3DClicked()
    {
        if (string.IsNullOrEmpty(lastImagePath))
        {
            Debug.LogWarning("Nessuna immagine disponibile per generare il modello 3D.");
            return;
        }

        if (maskPainter != null && maskPainter.HasPaintedPixels())
        {
            string editPrompt = promptInput != null ? promptInput.text.Trim() : "";
            if (string.IsNullOrEmpty(editPrompt))
            {
                Debug.LogWarning("Prompt di inpainting vuoto. Descrivi come modificare l'area dipinta.");
                return;
            }

            StartCoroutine(RunGenerate3DFlow(true, editPrompt));
            return;
        }

        StartCoroutine(RunGenerate3DFlow(false, ""));
    }

    IEnumerator RunGenerate3DFlow(bool useInpainting, string editPrompt)
    {
        generate3DButton.interactable = false;

        if (useInpainting)
        {
            yield return StartCoroutine(EditImageThenGenerate3D(editPrompt));
        }
        else
        {
            yield return StartCoroutine(Generate3DModel(lastImagePath));
        }

        RestoreGenerate3DButton();
    }

    IEnumerator EditImageThenGenerate3D(string editPrompt)
    {
        byte[] maskPng = maskPainter.GetMaskPngBytes();
        ImageInpaintingClient.EditResult editResult = null;

        yield return StartCoroutine(ImageInpaintingClient.EditImage(
            serverUrlEditImage,
            sessionId,
            editPrompt,
            lastImagePath,
            maskPng,
            result => editResult = result));

        if (editResult == null || !editResult.success)
        {
            Debug.LogError(editResult != null ? editResult.error : "La richiesta edit-image non ha restituito un risultato.");
            yield break;
        }

        lastImagePath = editResult.imagePath;

        if (!string.IsNullOrEmpty(editResult.imageUrl))
        {
            yield return StartCoroutine(DownloadAndShowImage(editResult.imageUrl));
        }

        Debug.Log("Inpainting completato. Avvio automatico della generazione 3D.");

        yield return StartCoroutine(Generate3DModel(lastImagePath));
    }

    IEnumerator Generate3DModel(string imagePath)
    {
        Generate3DRequest body = new Generate3DRequest
        {
            session_id = sessionId,
            image_path = imagePath
        };

        string json = JsonUtility.ToJson(body);
        byte[] jsonBytes = Encoding.UTF8.GetBytes(json);

        Debug.Log("Richiesta a Trellis avviata. La generazione 3D richiede tempo, attendere...");

        using (UnityWebRequest request = new UnityWebRequest(serverUrl3D, "POST"))
        {
            request.uploadHandler = new UploadHandlerRaw(jsonBytes);
            request.downloadHandler = new DownloadHandlerBuffer();
            request.SetRequestHeader("Content-Type", "application/json");
            
            // Timeout aumentato a 5 minuti (300 secondi) dato che i modelli 3D sono lenti da generare
            request.timeout = 300; 

            yield return request.SendWebRequest();

            if (request.result != UnityWebRequest.Result.Success)
            {
                Debug.LogError("Errore POST 3D: " + request.error);
                yield break;
            }

            Generate3DResponse response = JsonUtility.FromJson<Generate3DResponse>(request.downloadHandler.text);

            if (response == null || response.status != "ok" || string.IsNullOrEmpty(response.model3d_url))
            {
                Debug.LogError("Risposta 3D non valida: " + request.downloadHandler.text);
                yield break;
            }

            Debug.Log("Generazione 3D completata! Inizio il download del modello.");
            yield return StartCoroutine(DownloadAndSave3DModel(response.model3d_url));
        }

    }

    void RestoreGenerate3DButton()
    {
        if (generate3DButton != null)
        {
            generate3DButton.interactable = !string.IsNullOrEmpty(lastImagePath) && !isSegmentingMask;
        }
    }

    IEnumerator DownloadAndSave3DModel(string modelUrl)
    {
        using (UnityWebRequest request = UnityWebRequest.Get(modelUrl))
        {
            yield return request.SendWebRequest();

            if (request.result != UnityWebRequest.Result.Success)
            {
                Debug.LogError("Errore download modello 3D: " + request.error);
                yield break;
            }

            // Assicuriamoci che l'estensione sia corretta. Trellis dovrebbe restituire un .glb
            string extension = modelUrl.EndsWith(".obj") ? ".obj" : ".glb";
            
            // Salva il file localmente sul dispositivo
            string savePath = Application.persistentDataPath + "/generated_model" + extension;
            System.IO.File.WriteAllBytes(savePath, request.downloadHandler.data);
            
            Debug.Log($"Modello salvato in: {savePath}");

            // Avvia il caricamento nella scena solo se è un .glb
            if (extension == ".glb")
            {
                LoadModelIntoScene(savePath);
            }
            else
            {
                Debug.LogWarning("Il file generato non è un .glb. glTFast supporta solo .glb o .gltf.");
            }
        }
    }
    #endregion

    #region Fase 3: Visualizzazione Modello (glTFast)
    // async void perché viene chiamata da una Coroutine standard di Unity
    async void LoadModelIntoScene(string filePath)
    {
        Debug.Log("Caricamento del modello nella scena in corso...");

        if (currentModelContainer != null)
        {
            Destroy(currentModelContainer);
        }

        // Crea un contenitore indipendente dal Canvas.
        GameObject modelContainer = new GameObject("Trellis_Generated_Model");
        currentModelContainer = modelContainer;
        
        // 2. Inizializza l'importer di glTFast
        var gltfImport = new GltfImport();
        
        // 3. Carica il file locale in modo asincrono (serve il prefisso file://)
        bool success = await gltfImport.Load($"file://{filePath}");
        
        if (success)
        {
            // 4. (MODIFICATO) Usa il nuovo metodo asincrono per istanziare la scena
            bool instantiateSuccess = await gltfImport.InstantiateMainSceneAsync(modelContainer.transform);
            
            if (instantiateSuccess)
            {
                if (!NormalizeAndPositionModel(modelContainer))
                {
                    Debug.LogWarning("Modello caricato, ma non sono stati trovati Renderer per calcolarne scala e posizione.");
                }

                Debug.Log("Successo! Il modello 3D è ora visibile.");
            }
            else
            {
                Debug.LogError("Il file .glb è stato caricato, ma c'è stato un errore durante la sua generazione nella scena.");
                Destroy(modelContainer);
            }
        }
        else
        {
            Debug.LogError("Impossibile caricare il file .glb. Il file potrebbe essere corrotto o invalido.");
            Destroy(modelContainer); // Pulisce la scena eliminando il contenitore vuoto
        }
    }

    bool NormalizeAndPositionModel(GameObject modelContainer)
    {
        Renderer[] renderers = modelContainer.GetComponentsInChildren<Renderer>(true);
        if (renderers.Length == 0)
        {
            return false;
        }

        Bounds bounds = renderers[0].bounds;
        for (int i = 1; i < renderers.Length; i++)
        {
            bounds.Encapsulate(renderers[i].bounds);
        }

        float largestDimension = Mathf.Max(bounds.size.x, bounds.size.y, bounds.size.z);
        if (largestDimension <= Mathf.Epsilon)
        {
            return false;
        }

        float scaleFactor = modelSize / largestDimension;
        modelContainer.transform.localScale *= scaleFactor;

        bounds = renderers[0].bounds;
        for (int i = 1; i < renderers.Length; i++)
        {
            bounds.Encapsulate(renderers[i].bounds);
        }

        RectTransform imageRect = resultImage.rectTransform;
        Vector3[] corners = new Vector3[4];
        imageRect.GetWorldCorners(corners);
        Vector3 imageRightCenter = (corners[2] + corners[3]) * 0.5f;

        Vector3 targetCenter =
            imageRightCenter
            + imageRect.right * (modelGapFromImage + modelSize * 0.5f)
            + imageRect.forward * modelDepthOffset;

        modelContainer.transform.position += targetCenter - bounds.center;
        return true;
    }
    #endregion

    void OnDestroy()
    {
        if (maskPainter != null)
        {
            maskPainter.PointSelected -= OnMaskPointSelected;
        }
    }
}
