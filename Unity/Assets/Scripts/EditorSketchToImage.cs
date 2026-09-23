using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using TMPro;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.InputSystem;
using UnityEngine.Networking;
using UnityEngine.UI;

/// <summary>
/// Editor prototype: draws world-space LineRenderers with the mouse, projects them
/// onto a white sketch image, then sends that image to SDXL ControlNet Scribble.
/// The stroke methods are public so a Quest/controller input adapter can reuse them.
/// </summary>
public class EditorSketchToImage : MonoBehaviour
{
    [Header("Scene")]
    public Camera viewCamera;
    public Material strokeMaterial;
    public TMP_InputField promptInput;
    public RawImage generatedImage;
    public Button generateButton;
    public Button clearButton;

    [Header("Sketch plane in front of the camera")]
    [Min(0.1f)] public float distance = 2f;
    [Min(0.1f)] public float width = 1.5f;
    [Min(0.1f)] public float height = 1.5f;
    [Min(0.001f)] public float strokeWidth = 0.008f;
    [Min(0.001f)] public float minPointDistance = 0.005f;
    [Range(0, 31)] public int sketchLayer = 30;

    [Header("SDXL server")]
    public string sdxlUrl = "http://127.0.0.1:5001/generate";
    public bool magicEnrich = false;
    [Range(0f, 1f)] public float strength = 0.85f;
    [Range(0f, 1f)] public float endPercent = 0.9f;
    [Range(1f, 15f)] public float cfg = 7f;
    [Range(10, 50)] public int steps = 20;
    [Min(128)] public int imageSize = 1024;

    public Texture2D LastGeneratedImage { get; private set; }

    private readonly List<LineRenderer> strokes = new List<LineRenderer>();
    private LineRenderer activeStroke;
    private Vector3 lastPoint;
    private Vector3 planeCenter;
    private Quaternion planeRotation;
    private bool busy;

    [Serializable]
    private class GenerateRequest
    {
        public string image;
        public string prompt;
        public bool magic_enrich;
        public float strength;
        public float start_percent;
        public float end_percent;
        public float cfg;
        public int steps;
    }

    [Serializable]
    private class GenerateResponse
    {
        public string status;
        public string image;
        public string message;
    }

    private void Start()
    {
        if (viewCamera == null) viewCamera = Camera.main;
        if (viewCamera == null || strokeMaterial == null)
        {
            Debug.LogError("EditorSketchToImage: assegna View Camera e Stroke Material.");
            enabled = false;
            return;
        }

        // Freeze the drawing plane so existing strokes do not follow camera motion.
        planeRotation = viewCamera.transform.rotation;
        planeCenter = viewCamera.transform.position + viewCamera.transform.forward * distance;
        if (generateButton != null) generateButton.onClick.AddListener(GenerateFromSketch);
        if (clearButton != null) clearButton.onClick.AddListener(ClearSketch);
    }

    private void OnDestroy()
    {
        if (generateButton != null) generateButton.onClick.RemoveListener(GenerateFromSketch);
        if (clearButton != null) clearButton.onClick.RemoveListener(ClearSketch);
    }

    private void Update()
    {
        Mouse mouse = Mouse.current;
        if (mouse == null || busy) return;
        if (mouse.leftButton.wasReleasedThisFrame) EndStroke();
        if (!mouse.leftButton.isPressed) return;
        if (mouse.leftButton.wasPressedThisFrame && EventSystem.current != null &&
            EventSystem.current.IsPointerOverGameObject()) return;
        if (activeStroke == null && !mouse.leftButton.wasPressedThisFrame) return;

        Ray ray = viewCamera.ScreenPointToRay(mouse.position.ReadValue());
        Plane plane = new Plane(planeRotation * Vector3.forward, planeCenter);
        if (!plane.Raycast(ray, out float t)) return;
        Vector3 point = ray.GetPoint(t);
        Vector3 local = Quaternion.Inverse(planeRotation) * (point - planeCenter);
        if (Mathf.Abs(local.x) > width * 0.5f || Mathf.Abs(local.y) > height * 0.5f)
        {
            EndStroke();
            return;
        }
        if (mouse.leftButton.wasPressedThisFrame) BeginStroke(point);
        else AddPoint(point);
    }

    public void BeginStroke(Vector3 worldPoint)
    {
        EndStroke();
        GameObject obj = new GameObject("Sketch stroke");
        obj.layer = sketchLayer;
        activeStroke = obj.AddComponent<LineRenderer>();
        activeStroke.useWorldSpace = true;
        activeStroke.material = strokeMaterial;
        activeStroke.startColor = Color.black;
        activeStroke.endColor = Color.black;
        activeStroke.startWidth = strokeWidth;
        activeStroke.endWidth = strokeWidth;
        activeStroke.numCapVertices = 8;
        activeStroke.numCornerVertices = 4;
        // Two points make a click visible too.
        activeStroke.positionCount = 2;
        activeStroke.SetPosition(0, worldPoint);
        activeStroke.SetPosition(1, worldPoint);
        lastPoint = worldPoint;
        strokes.Add(activeStroke);
    }

    public void AddPoint(Vector3 worldPoint)
    {
        if (activeStroke == null || Vector3.Distance(lastPoint, worldPoint) < minPointDistance) return;
        activeStroke.positionCount++;
        activeStroke.SetPosition(activeStroke.positionCount - 1, worldPoint);
        lastPoint = worldPoint;
    }

    public void EndStroke() { activeStroke = null; }

    public void ClearSketch()
    {
        EndStroke();
        foreach (LineRenderer stroke in strokes)
            if (stroke != null) Destroy(stroke.gameObject);
        strokes.Clear();
    }

    public void GenerateFromSketch()
    {
        if (busy || strokes.Count == 0) return;
        string prompt = promptInput != null ? promptInput.text.Trim() : "";
        if (prompt.Length == 0)
        {
            Debug.LogWarning("Scrivi un prompt prima di generare.");
            return;
        }
        StartCoroutine(SendSketch(prompt));
    }

    private Texture2D RenderSketch()
    {
        GameObject cameraObject = new GameObject("Temporary sketch camera");
        Camera capture = cameraObject.AddComponent<Camera>();
        RenderTexture target = null;
        Texture2D texture = null;
        RenderTexture previous = RenderTexture.active;
        try
        {
            capture.enabled = false;
            capture.orthographic = true;
            capture.orthographicSize = height * 0.5f;
            capture.aspect = width / height;
            capture.clearFlags = CameraClearFlags.SolidColor;
            capture.backgroundColor = Color.white;
            capture.cullingMask = 1 << sketchLayer;
            capture.nearClipPlane = 0.01f;
            capture.farClipPlane = 2f;
            cameraObject.transform.SetPositionAndRotation(planeCenter - planeRotation * Vector3.forward,
                planeRotation);
            target = RenderTexture.GetTemporary(imageSize, imageSize, 24, RenderTextureFormat.ARGB32);
            capture.targetTexture = target;
            capture.Render();
            RenderTexture.active = target;
            texture = new Texture2D(imageSize, imageSize, TextureFormat.RGB24, false);
            texture.ReadPixels(new Rect(0, 0, imageSize, imageSize), 0, 0);
            texture.Apply();
            return texture;
        }
        finally
        {
            RenderTexture.active = previous;
            capture.targetTexture = null;
            if (target != null) RenderTexture.ReleaseTemporary(target);
            Destroy(cameraObject);
        }
    }

    private IEnumerator SendSketch(string prompt)
    {
        busy = true;
        if (generateButton != null) generateButton.interactable = false;
        Texture2D sketch = null;
        try
        {
            sketch = RenderSketch();
        }
        catch (Exception error)
        {
            Debug.LogError("Impossibile acquisire lo sketch: " + error);
            busy = false;
            if (generateButton != null) generateButton.interactable = true;
            yield break;
        }

        GenerateRequest body = new GenerateRequest {
            image = Convert.ToBase64String(sketch.EncodeToPNG()), prompt = prompt,
            magic_enrich = magicEnrich, strength = strength,
            start_percent = 0f, end_percent = endPercent, cfg = cfg, steps = steps
        };
        Destroy(sketch);
        using (UnityWebRequest request = new UnityWebRequest(sdxlUrl, "POST"))
        {
            request.uploadHandler = new UploadHandlerRaw(Encoding.UTF8.GetBytes(JsonUtility.ToJson(body)));
            request.downloadHandler = new DownloadHandlerBuffer();
            request.SetRequestHeader("Content-Type", "application/json");
            request.timeout = 300;
            yield return request.SendWebRequest();
            if (request.result != UnityWebRequest.Result.Success)
                Debug.LogError("SDXL: " + request.error + " " + request.downloadHandler.text);
            else
            {
                try
                {
                    GenerateResponse response = JsonUtility.FromJson<GenerateResponse>(request.downloadHandler.text);
                    if (response == null || response.status != "success" || string.IsNullOrEmpty(response.image))
                        Debug.LogError("SDXL: " + (response == null ? request.downloadHandler.text : response.message));
                    else
                    {
                        Texture2D result = new Texture2D(2, 2);
                        if (result.LoadImage(Convert.FromBase64String(response.image)))
                        {
                            if (LastGeneratedImage != null) Destroy(LastGeneratedImage);
                            LastGeneratedImage = result;
                            if (generatedImage != null) generatedImage.texture = result;
                            Debug.Log("Immagine generata dallo sketch.");
                        }
                        else Destroy(result);
                    }
                }
                catch (Exception error)
                {
                    Debug.LogError("Risposta SDXL non valida: " + error.Message);
                }
            }
        }
        busy = false;
        if (generateButton != null) generateButton.interactable = true;
    }
}
