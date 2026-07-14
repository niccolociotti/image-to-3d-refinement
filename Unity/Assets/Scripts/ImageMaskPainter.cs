using System;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.InputSystem;
using UnityEngine.UI;

public class ImageMaskPainter : MonoBehaviour, IPointerDownHandler, IDragHandler, IPointerUpHandler, IScrollHandler
{
    public int brushRadius = 28;
    public int minBrushRadius = 4;
    public int maxBrushRadius = 128;
    public Color32 overlayPaintColor = new Color32(255, 70, 70, 120);
    public bool brushEnabled = true;

    public event Action<Vector2Int> PointSelected;

    RawImage overlayImage;
    Texture2D maskTexture;
    Texture2D overlayTexture;
    Vector2Int previousPixel;
    bool isDrawing;
    bool hasPaintedPixels;

    public void Initialize(RawImage image, int width, int height)
    {
        overlayImage = image;
        ResetMask(width, height);
    }

    public void ResetMask(int width, int height)
    {
        if (width <= 0 || height <= 0)
        {
            return;
        }

        DestroyTextures();

        maskTexture = new Texture2D(width, height, TextureFormat.RGBA32, false);
        overlayTexture = new Texture2D(width, height, TextureFormat.RGBA32, false);
        ClearMask();

        if (overlayImage != null)
        {
            overlayImage.texture = overlayTexture;
            overlayImage.color = Color.white;
            overlayImage.raycastTarget = true;
        }
    }

    public string GetMaskBase64()
    {
        byte[] maskPng = GetMaskPngBytes();
        if (maskPng == null)
        {
            return "";
        }

        return Convert.ToBase64String(maskPng);
    }

    public byte[] GetMaskPngBytes()
    {
        if (maskTexture == null)
        {
            return null;
        }

        return maskTexture.EncodeToPNG();
    }

    public bool HasPaintedPixels()
    {
        return hasPaintedPixels;
    }

    public void SetMaskFromPng(byte[] maskPng)
    {
        ApplyMaskFromPng(maskPng, false);
    }

    public void AddMaskFromPng(byte[] maskPng)
    {
        ApplyMaskFromPng(maskPng, true);
    }

    void ApplyMaskFromPng(byte[] maskPng, bool addToExistingMask)
    {
        if (maskPng == null || maskPng.Length == 0)
        {
            return;
        }

        Texture2D sourceTexture = new Texture2D(2, 2, TextureFormat.RGBA32, false);
        if (!sourceTexture.LoadImage(maskPng))
        {
            Destroy(sourceTexture);
            return;
        }

        bool canReuseTextures = addToExistingMask
            && maskTexture != null
            && overlayTexture != null
            && maskTexture.width == sourceTexture.width
            && maskTexture.height == sourceTexture.height;

        if (!canReuseTextures)
        {
            DestroyTextures();

            maskTexture = new Texture2D(sourceTexture.width, sourceTexture.height, TextureFormat.RGBA32, false);
            overlayTexture = new Texture2D(sourceTexture.width, sourceTexture.height, TextureFormat.RGBA32, false);
        }

        Color32[] sourcePixels = sourceTexture.GetPixels32();
        Color32[] maskPixels = canReuseTextures ? maskTexture.GetPixels32() : new Color32[sourcePixels.Length];
        Color32[] overlayPixels = canReuseTextures ? overlayTexture.GetPixels32() : new Color32[sourcePixels.Length];
        Color32 black = new Color32(0, 0, 0, 255);
        Color32 white = new Color32(255, 255, 255, 255);
        Color32 transparent = new Color32(0, 0, 0, 0);
        bool anySelected = false;

        for (int i = 0; i < sourcePixels.Length; i++)
        {
            Color32 pixel = sourcePixels[i];
            bool selected = pixel.r > 127 || pixel.g > 127 || pixel.b > 127;

            if (addToExistingMask && !selected)
            {
                selected = maskPixels[i].r > 127 || maskPixels[i].g > 127 || maskPixels[i].b > 127;
            }

            maskPixels[i] = selected ? white : black;
            overlayPixels[i] = selected ? overlayPaintColor : transparent;
            anySelected |= selected;
        }

        maskTexture.SetPixels32(maskPixels);
        overlayTexture.SetPixels32(overlayPixels);
        ApplyTextures();

        if (overlayImage != null)
        {
            overlayImage.texture = overlayTexture;
            overlayImage.color = Color.white;
            overlayImage.raycastTarget = true;
        }

        hasPaintedPixels = anySelected;
        Destroy(sourceTexture);
    }

    public void ClearMask()
    {
        if (maskTexture == null || overlayTexture == null)
        {
            return;
        }

        Color32[] maskPixels = new Color32[maskTexture.width * maskTexture.height];
        Color32[] overlayPixels = new Color32[overlayTexture.width * overlayTexture.height];
        Color32 black = new Color32(0, 0, 0, 255);
        Color32 transparent = new Color32(0, 0, 0, 0);

        for (int i = 0; i < maskPixels.Length; i++)
        {
            maskPixels[i] = black;
            overlayPixels[i] = transparent;
        }

        maskTexture.SetPixels32(maskPixels);
        overlayTexture.SetPixels32(overlayPixels);
        ApplyTextures();
        hasPaintedPixels = false;
    }

    public void OnPointerDown(PointerEventData eventData)
    {
        if (eventData.button != PointerEventData.InputButton.Left)
        {
            return;
        }

        if (!TryGetTexturePixel(eventData, out Vector2Int pixel))
        {
            return;
        }

        PointSelected?.Invoke(pixel);

        if (!brushEnabled)
        {
            return;
        }

        isDrawing = true;
        previousPixel = pixel;
        PaintLine(pixel, pixel, IsErasePressed());
    }

    public void OnDrag(PointerEventData eventData)
    {
        if (!brushEnabled || !isDrawing || !TryGetTexturePixel(eventData, out Vector2Int pixel))
        {
            return;
        }

        PaintLine(previousPixel, pixel, IsErasePressed());
        previousPixel = pixel;
    }

    public void OnPointerUp(PointerEventData eventData)
    {
        if (eventData.button == PointerEventData.InputButton.Left)
        {
            isDrawing = false;
        }
    }

    public void OnScroll(PointerEventData eventData)
    {
        int direction = eventData.scrollDelta.y >= 0f ? 1 : -1;
        brushRadius = Mathf.Clamp(brushRadius + direction * 4, minBrushRadius, maxBrushRadius);
        Debug.Log($"ImageMaskPainter: raggio pennello {brushRadius}px");
    }

    void PaintLine(Vector2Int start, Vector2Int end, bool erase)
    {
        if (maskTexture == null || overlayTexture == null)
        {
            return;
        }

        float distance = Vector2.Distance(start, end);
        int steps = Mathf.Max(1, Mathf.CeilToInt(distance / Mathf.Max(1f, brushRadius * 0.35f)));

        for (int i = 0; i <= steps; i++)
        {
            float t = (float)i / steps;
            int x = Mathf.RoundToInt(Mathf.Lerp(start.x, end.x, t));
            int y = Mathf.RoundToInt(Mathf.Lerp(start.y, end.y, t));
            PaintCircle(x, y, erase);
        }

        ApplyTextures();
    }

    void PaintCircle(int centerX, int centerY, bool erase)
    {
        int radiusSquared = brushRadius * brushRadius;
        Color32 maskColor = erase ? new Color32(0, 0, 0, 255) : new Color32(255, 255, 255, 255);
        Color32 displayColor = erase ? new Color32(0, 0, 0, 0) : overlayPaintColor;

        for (int y = -brushRadius; y <= brushRadius; y++)
        {
            for (int x = -brushRadius; x <= brushRadius; x++)
            {
                if (x * x + y * y > radiusSquared)
                {
                    continue;
                }

                int pixelX = centerX + x;
                int pixelY = centerY + y;
                if (pixelX < 0 || pixelY < 0 || pixelX >= maskTexture.width || pixelY >= maskTexture.height)
                {
                    continue;
                }

                if (!erase)
                {
                    hasPaintedPixels = true;
                }

                maskTexture.SetPixel(pixelX, pixelY, maskColor);
                overlayTexture.SetPixel(pixelX, pixelY, displayColor);
            }
        }
    }

    bool TryGetTexturePixel(PointerEventData eventData, out Vector2Int pixel)
    {
        pixel = default;
        RectTransform rectTransform = transform as RectTransform;
        if (rectTransform == null || maskTexture == null)
        {
            return false;
        }

        if (!RectTransformUtility.ScreenPointToLocalPointInRectangle(
                rectTransform,
                eventData.position,
                eventData.pressEventCamera,
                out Vector2 localPoint))
        {
            return false;
        }

        Rect rect = rectTransform.rect;
        float normalizedX = Mathf.InverseLerp(rect.xMin, rect.xMax, localPoint.x);
        float normalizedY = Mathf.InverseLerp(rect.yMin, rect.yMax, localPoint.y);

        pixel = new Vector2Int(
            Mathf.Clamp(Mathf.RoundToInt(normalizedX * (maskTexture.width - 1)), 0, maskTexture.width - 1),
            Mathf.Clamp(Mathf.RoundToInt(normalizedY * (maskTexture.height - 1)), 0, maskTexture.height - 1));

        return true;
    }

    bool IsErasePressed()
    {
        return Keyboard.current != null
            && (Keyboard.current.leftShiftKey.isPressed || Keyboard.current.rightShiftKey.isPressed);
    }

    void ApplyTextures()
    {
        maskTexture.Apply(false);
        overlayTexture.Apply(false);
    }

    void OnDestroy()
    {
        DestroyTextures();
    }

    void DestroyTextures()
    {
        if (maskTexture != null)
        {
            Destroy(maskTexture);
            maskTexture = null;
        }

        if (overlayTexture != null)
        {
            Destroy(overlayTexture);
            overlayTexture = null;
        }

        hasPaintedPixels = false;
    }
}
