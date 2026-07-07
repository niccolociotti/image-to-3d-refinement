using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;

public static class SAM2SegmentationClient
{
    public class SegmentResult
    {
        public bool success;
        public byte[] maskPng;
        public string maskPath;
        public string maskUrl;
        public float score;
        public int selectedMaskIndex;
        public string error;
    }

    [Serializable]
    private class SegmentPoint
    {
        public float x;
        public float y;
        public int label;
    }

    [Serializable]
    private class SegmentImageRequest
    {
        public string session_id;
        public string image_path;
        public List<SegmentPoint> points;
        public bool return_base64;
        public bool coordinates_normalized;
        public bool multimask_output;
        public string selection_mode;
        public int grow_mask_by;
        public float mask_blur;
        public bool invert_mask;
    }

    [Serializable]
    private class SegmentImageResponse
    {
        public string status;
        public string mask_path;
        public string mask_url;
        public string mask_base64;
        public float score;
        public int selected_mask_index;
    }

    public static IEnumerator SegmentImage(
        string serverUrl,
        string sessionId,
        string imagePath,
        Vector2 imagePixelPoint,
        Action<SegmentResult> onCompleted,
        string selectionMode = "part",
        int growMaskBy = 0,
        float maskBlur = 0f)
    {
        SegmentResult result = new SegmentResult();

        if (string.IsNullOrEmpty(serverUrl))
        {
            result.error = "URL /segment-image non configurato.";
            onCompleted?.Invoke(result);
            yield break;
        }

        if (string.IsNullOrEmpty(imagePath))
        {
            result.error = "Percorso dell'immagine originale non disponibile.";
            onCompleted?.Invoke(result);
            yield break;
        }

        SegmentImageRequest body = new SegmentImageRequest
        {
            session_id = sessionId,
            image_path = imagePath,
            points = new List<SegmentPoint>
            {
                new SegmentPoint
                {
                    x = imagePixelPoint.x,
                    y = imagePixelPoint.y,
                    label = 1
                }
            },
            return_base64 = true,
            coordinates_normalized = false,
            multimask_output = true,
            selection_mode = selectionMode,
            grow_mask_by = growMaskBy,
            mask_blur = maskBlur,
            invert_mask = false
        };

        byte[] jsonBytes = Encoding.UTF8.GetBytes(JsonUtility.ToJson(body));
        Debug.Log($"Richiesta SAM2 avviata nel punto ({imagePixelPoint.x:0}, {imagePixelPoint.y:0}).");

        using (UnityWebRequest request = new UnityWebRequest(serverUrl, "POST"))
        {
            request.uploadHandler = new UploadHandlerRaw(jsonBytes);
            request.downloadHandler = new DownloadHandlerBuffer();
            request.SetRequestHeader("Content-Type", "application/json");
            request.timeout = 300;

            yield return request.SendWebRequest();

            if (request.result != UnityWebRequest.Result.Success)
            {
                result.error = "Errore POST segment-image: " + request.error + "\n" + request.downloadHandler.text;
                onCompleted?.Invoke(result);
                yield break;
            }

            SegmentImageResponse response = JsonUtility.FromJson<SegmentImageResponse>(request.downloadHandler.text);

            if (response == null || response.status != "ok" || string.IsNullOrEmpty(response.mask_base64))
            {
                result.error = "Risposta segment-image non valida: " + request.downloadHandler.text;
                onCompleted?.Invoke(result);
                yield break;
            }

            try
            {
                result.maskPng = Convert.FromBase64String(response.mask_base64);
            }
            catch (Exception exception)
            {
                result.error = "mask_base64 non valida: " + exception.Message;
                onCompleted?.Invoke(result);
                yield break;
            }

            result.success = true;
            result.maskPath = response.mask_path;
            result.maskUrl = response.mask_url;
            result.score = response.score;
            result.selectedMaskIndex = response.selected_mask_index;
            onCompleted?.Invoke(result);
        }
    }
}
