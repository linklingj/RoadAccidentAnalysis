import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Road/Object inference + PerspectiveFields + BEV projection")
    parser.add_argument("--image", type=str, default="input/image2.png", help="Input image path")
    parser.add_argument("--video", type=str, default=None, help="Input video path (if set, video mode runs)")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory to save overlay and BEV images")
    parser.add_argument("--road-model", type=str, default="runs/segment/0401-road/weights/best.pt", help="Road YOLO model path")
    parser.add_argument("--object-model", type=str, default="runs/segment/0401-object/weights/best.pt", help="Object YOLO model path")
    parser.add_argument("--perspective-version", type=str, default="Paramnet-360Cities-edina-centered", help="PerspectiveFields model version")
    parser.add_argument("--road-conf", type=float, default=0.25, help="Road model confidence threshold")
    parser.add_argument("--object-conf", type=float, default=0.15, help="Object model confidence threshold")
    parser.add_argument("--camera-height", type=float, default=2.5, help="Assumed camera height in meters")
    parser.add_argument("--ppm", type=float, default=28.0, help="Pixels per meter for BEV rendering")
    parser.add_argument("--bev-width", type=int, default=960, help="BEV image width")
    parser.add_argument("--bev-height", type=int, default=960, help="BEV image height")
    parser.add_argument("--device", type=str, default=None, help="Inference device. e.g., 'cpu', '0', 'cuda:0'")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        from infer import PipelineConfig, run_pipeline, run_video_pipeline
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Missing dependency while importing pipeline: {exc}. "
            "Install required packages (e.g., opencv-python, torch, ultralytics)."
        ) from exc

    parsed_device = None
    if args.device is not None:
        parsed_device = int(args.device) if args.device.isdigit() else args.device

    config = PipelineConfig(
        road_model_path=args.road_model,
        object_model_path=args.object_model,
        perspective_version=args.perspective_version,
        road_conf=args.road_conf,
        object_conf=args.object_conf,
        camera_height_m=args.camera_height,
        pixels_per_meter=args.ppm,
        bev_width=args.bev_width,
        bev_height=args.bev_height,
        device=parsed_device,
    )

    if args.video:
        outputs = run_video_pipeline(
            video_path=args.video,
            save_dir=args.output_dir,
            config=config,
        )
        cam = outputs["camera"]
        print(f"[Mode] video")
        print(f"[Camera] roll={cam['roll_deg']:.3f}, pitch={cam['pitch_deg']:.3f}, vfov={cam['vfov_deg']:.3f}")
        print(f"[Frames] processed={outputs['frames_processed']}")
        print(f"[Tracking] avg_2d={outputs['avg_detections_2d_per_frame']:.2f}, avg_tracked={outputs['avg_tracked_per_frame']:.2f}")
        print(f"[Saved] video={outputs.get('saved_video_path', 'N/A')}")
    else:
        outputs = run_pipeline(
            image_path=args.image,
            save_dir=args.output_dir,
            config=config,
        )
        cam = outputs["camera"]
        print(f"[Mode] image")
        print(f"[Camera] roll={cam['roll_deg']:.3f}, pitch={cam['pitch_deg']:.3f}, vfov={cam['vfov_deg']:.3f}")
        print(f"[Objects] 2D detections={len(outputs['detections_2d'])}, BEV projected={len(outputs['detections_bev'])}")
        print(f"[Saved] overlay={outputs.get('saved_overlay_path', 'N/A')}")
        print(f"[Saved] bev={outputs.get('saved_bev_path', 'N/A')}")


if __name__ == "__main__":
    main()
