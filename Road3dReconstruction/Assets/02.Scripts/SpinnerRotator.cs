using UnityEngine;

namespace RoadReconstruction
{
    // Rotates the attached RectTransform around Z. Used by the LoadingPanel spinner —
    // a tiny script so we don't have to ship an animator clip in this asset folder.
    [DisallowMultipleComponent]
    [RequireComponent(typeof(RectTransform))]
    public class SpinnerRotator : MonoBehaviour
    {
        [Tooltip("Degrees per second; negative spins counterclockwise.")]
        public float degreesPerSecond = -240f;

        private RectTransform _rt;

        private void Awake()
        {
            _rt = (RectTransform)transform;
        }

        private void Update()
        {
            _rt.Rotate(0f, 0f, degreesPerSecond * Time.deltaTime);
        }
    }
}
