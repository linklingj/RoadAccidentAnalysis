using UnityEngine;

[RequireComponent(typeof(Camera))]
public class CameraSetup : MonoBehaviour
{
    public DataManager dataManager;

    public void Setup()
    {
        if (dataManager == null)
            dataManager = FindObjectOfType<DataManager>();

        if (dataManager?.Data == null)
        {
            Debug.LogError("[CameraSetup] DataManager has no loaded data.");
            return;
        }

        Apply(dataManager.Data);
    }

    void Apply(SceneData data)
    {
        var cam = GetComponent<Camera>();
        var cp = data.camera;

        // Vertical field of view
        cam.fieldOfView = cp.vfov_deg;

        // Position the camera above the scene origin at the recorded height
        transform.position = new Vector3(0f, data.camera_height_m, 0f);

        // PerspectiveFields convention: y = down, z = forward.
        // Optical axis y-component = -sin(pitch_deg), so pitch_deg < 0 means looking downward.
        // In Unity (y = up): looking downward = positive X rotation → negate pitch_deg.
        // roll_deg > 0   → clockwise roll when viewed from behind = positive Z rotation.
        transform.rotation = Quaternion.Euler(-cp.pitch_deg, 0f, cp.roll_deg);
    }
}
