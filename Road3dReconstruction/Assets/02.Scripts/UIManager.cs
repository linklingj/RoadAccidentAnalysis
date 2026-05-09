using UnityEngine;
using UnityEngine.UI;

public class UIManager : MonoBehaviour
{
    public DataManager dataManager;
    public CameraSetup cameraSetup;
    public RoadBuilder roadBuilder;
    public CarSpawner carSpawner;

    void Start()
    {
        CreateUI();
    }

    void CreateUI()
    {
        // Canvas
        var canvasGo = new GameObject("HUD_Canvas");
        var canvas = canvasGo.AddComponent<Canvas>();
        canvas.renderMode = RenderMode.ScreenSpaceOverlay;
        canvasGo.AddComponent<CanvasScaler>();
        canvasGo.AddComponent<GraphicRaycaster>();

        // Button
        var btnGo = new GameObject("GenerateButton");
        btnGo.transform.SetParent(canvasGo.transform, false);

        var rt = btnGo.AddComponent<RectTransform>();
        rt.anchorMin = new Vector2(0f, 0f);
        rt.anchorMax = new Vector2(0f, 0f);
        rt.pivot = new Vector2(0f, 0f);
        rt.anchoredPosition = new Vector2(20f, 20f);
        rt.sizeDelta = new Vector2(200f, 50f);

        var img = btnGo.AddComponent<Image>();
        img.color = new Color(0.15f, 0.15f, 0.15f, 0.9f);

        var btn = btnGo.AddComponent<Button>();
        btn.onClick.AddListener(OnGenerateClicked);

        // Label
        var textGo = new GameObject("Label");
        textGo.transform.SetParent(btnGo.transform, false);

        var textRt = textGo.AddComponent<RectTransform>();
        textRt.anchorMin = Vector2.zero;
        textRt.anchorMax = Vector2.one;
        textRt.offsetMin = Vector2.zero;
        textRt.offsetMax = Vector2.zero;

        var text = textGo.AddComponent<Text>();
        text.text = "씬 생성";
        text.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        text.fontSize = 20;
        text.alignment = TextAnchor.MiddleCenter;
        text.color = Color.white;
    }

    void OnGenerateClicked()
    {
        if (dataManager == null)
            dataManager = FindObjectOfType<DataManager>();

        // Load JSON if not yet loaded
        if (dataManager != null && dataManager.Data == null)
            dataManager.Load(dataManager.jsonPath);

        cameraSetup?.Setup();
        roadBuilder?.Build();
        carSpawner?.Generate();
    }
}
