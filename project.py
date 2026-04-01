"""
Project-level usage example.

Run:
    python project.py
"""

from infer import PipelineConfig, run_pipeline


def example() -> None:
    config = PipelineConfig(
        road_model_path="runs/segment/0401-road/weights/best.pt",
        object_model_path="runs/segment/0401-object/weights/best.pt",
        perspective_version="Paramnet-360Cities-edina-centered",
        road_conf=0.25,
        object_conf=0.15,
        camera_height_m=2.5,
        pixels_per_meter=28.0,
        bev_width=960,
        bev_height=960,
    )

    outputs = run_pipeline(
        image_path="input/image1.png",
        save_dir="output",
        config=config,
    )

    print("camera:", outputs["camera"])
    print("detected objects:", len(outputs["detections_2d"]))
    print("projected objects:", len(outputs["detections_bev"]))
    print("overlay:", outputs.get("saved_overlay_path"))
    print("bev:", outputs.get("saved_bev_path"))


if __name__ == "__main__":
    example()
