using UnityEngine;
using TMPro;

public class DisableXRSimulatorWhileTyping : MonoBehaviour
{
    public GameObject xrDeviceSimulator;

    private TMP_InputField inputField;

    private void Awake()
    {
        inputField = GetComponent<TMP_InputField>();

        inputField.onSelect.AddListener(OnSelected);
        inputField.onDeselect.AddListener(OnDeselected);
    }

    private void OnSelected(string value)
    {
        if (xrDeviceSimulator != null)
            xrDeviceSimulator.SetActive(false);
    }

    private void OnDeselected(string value)
    {
        if (xrDeviceSimulator != null)
            xrDeviceSimulator.SetActive(true);
    }
} 