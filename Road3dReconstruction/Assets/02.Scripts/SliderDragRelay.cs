using UnityEngine;
using UnityEngine.EventSystems;

namespace RoadReconstruction
{
    // Forwards pointer-down / pointer-up events from the slider handle to a controller
    // so we can distinguish user-driven scrubbing from programmatic value changes.
    public class SliderDragRelay : MonoBehaviour, IPointerDownHandler, IPointerUpHandler
    {
        [HideInInspector] public VideoPlaybackController controller;

        public void OnPointerDown(PointerEventData eventData)
        {
            if (controller != null) controller.OnSliderBeginDrag();
        }

        public void OnPointerUp(PointerEventData eventData)
        {
            if (controller != null) controller.OnSliderEndDrag();
        }
    }
}
