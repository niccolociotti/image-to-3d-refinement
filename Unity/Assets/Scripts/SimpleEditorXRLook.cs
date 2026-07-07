using UnityEngine;
using UnityEngine.InputSystem;

public class SimpleEditorXRLook : MonoBehaviour
{
    public Transform cameraOffset;
    public float sensitivity = 0.15f;

    private float pitch = 0f;

    void Update()
    {
        if (Mouse.current == null || cameraOffset == null)
            return;

        bool rightPressed = Mouse.current.rightButton.isPressed;
        Cursor.visible = !rightPressed;

        if (!rightPressed)
            return;

        Vector2 delta = Mouse.current.delta.ReadValue();

        // Orizzontale
        transform.Rotate(Vector3.up, delta.x * sensitivity);

        // Verticale
        pitch -= delta.y * sensitivity;
        pitch = Mathf.Clamp(pitch, -80f, 80f);

        cameraOffset.localRotation = Quaternion.Euler(pitch, 0f, 0f);
    }
}