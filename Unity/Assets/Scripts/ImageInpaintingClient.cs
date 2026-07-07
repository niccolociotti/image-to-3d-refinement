using System;
using System.Collections;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;

public static class ImageInpaintingClient
{
    public class EditResult
    {
        public bool success;
        public string imagePath;
        public string imageUrl;
        public string error;
    }

    [Serializable]
    private class EditImageRequest
    {
        public string session_id;
        public string image_path;
        public string mask_base64;
        public string prompt;
    }

    [Serializable]
    private class EditImageResponse
    {
        public string edited_image_path;
        public string edited_image_url;
        public string status;
    }

    public static IEnumerator EditImage(
        string serverUrl,
        string sessionId,
        string prompt,
        string originalImagePath,
        byte[] maskPng,
        Action<EditResult> onCompleted)
    {
        EditResult result = new EditResult();

        if (string.IsNullOrEmpty(originalImagePath))
        {
            result.error = "Percorso dell'immagine originale non disponibile.";
            onCompleted?.Invoke(result);
            yield break;
        }

        if (maskPng == null || maskPng.Length == 0)
        {
            result.error = "Impossibile generare mask.png.";
            onCompleted?.Invoke(result);
            yield break;
        }

        EditImageRequest body = new EditImageRequest
        {
            session_id = sessionId,
            image_path = originalImagePath,
            mask_base64 = Convert.ToBase64String(maskPng),
            prompt = prompt
        };

        byte[] jsonBytes = Encoding.UTF8.GetBytes(JsonUtility.ToJson(body));
        Debug.Log("Richiesta di inpainting avviata: la maschera sarà salvata solo sul server.");

        using (UnityWebRequest request = new UnityWebRequest(serverUrl, "POST"))
        {
            request.uploadHandler = new UploadHandlerRaw(jsonBytes);
            request.downloadHandler = new DownloadHandlerBuffer();
            request.SetRequestHeader("Content-Type", "application/json");
            request.timeout = 300;

            yield return request.SendWebRequest();

            if (request.result != UnityWebRequest.Result.Success)
            {
                result.error = "Errore POST edit-image: " + request.error + "\n" + request.downloadHandler.text;
                onCompleted?.Invoke(result);
                yield break;
            }

            EditImageResponse response = JsonUtility.FromJson<EditImageResponse>(request.downloadHandler.text);

            if (response == null || response.status != "ok" || string.IsNullOrEmpty(response.edited_image_path))
            {
                result.error = "Risposta edit-image non valida: " + request.downloadHandler.text;
                onCompleted?.Invoke(result);
                yield break;
            }

            result.success = true;
            result.imagePath = response.edited_image_path;
            result.imageUrl = response.edited_image_url;
            onCompleted?.Invoke(result);
        }
    }
}
