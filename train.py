import torch
from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='road', help='Model type to train: road | crosswalk | object | rfdetr-object')
parser.add_argument('--dataset', type=str, default='cctv-object-dataset-coco', help='COCO format dataset dir (rfdetr-object only)')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch', type=int, default=4, help='Batch size per step (rfdetr-object only)')
parser.add_argument('--grad-accum', type=int, default=4, help='Gradient accumulation steps (rfdetr-object only)')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (rfdetr-object only)')
parser.add_argument('--output-dir', type=str, default='runs/detect/rfdetr-object', help='Checkpoint output dir (rfdetr-object only)')
args = parser.parse_args()

def train_road_model():

    DATA_PATH = 'cctv-roadseg-dataset/data.yaml'

    model = YOLO('yolo26l-seg.pt')

    model.train(
        data=DATA_PATH,
        epochs=50,
        imgsz=1280,
        batch=-1,
        device=0,
        # CCTV 특화 augmentation
        hsv_v=0.4,       # 야간·역광 조도 변화 대응
        degrees=5.0,     # 고정 카메라이므로 소폭 회전만 허용
        shear=2.0,       # 미세 원근 변화
        mosaic=1.0,      # 다양한 장면 조합
        copy_paste=0.1,  # 도로 마스크 붙여넣기 보강
    )

def train_crosswalk_model():

    DATA_PATH = 'cctv-crosswalk-dataset/data.yaml'

    model = YOLO('yolo26l-seg.pt')

    model.train(
        data=DATA_PATH,
        epochs=50,
        imgsz=1280,
        batch=-1,
        device=0,
        patience=20,
        # CCTV 특화 augmentation
        hsv_v=0.4,       # 야간·역광 조도 변화 대응
        degrees=5.0,
        shear=2.0,
        mosaic=1.0,
        copy_paste=0.3,  # 횡단보도 희소 클래스 보강
    )

def train_object_model():

    DATA_PATH = 'cctv-object-dataset/data.yaml'

    model = YOLO('yolo26s-seg.pt')

    model.train(
        data=DATA_PATH,
        epochs=50,
        imgsz=1280,
        batch=-1,
        device=0,
        # CCTV 특화 augmentation
        hsv_v=0.4,       # 야간·역광 조도 변화 대응
        degrees=5.0,
        shear=2.0,
        mosaic=1.0,
        copy_paste=0.3,  # 보행자·오토바이 희소 클래스 보강
        mixup=0.1,       # 클래스 간 혼동 방지
    )


def train_rfdetr_object_model():
    """RF-DETR 기반 객체 탐지 모델 학습.

    사전 준비:
        pip install rfdetr supervision
        python util/convert_yolo_to_coco.py --src cctv-object-dataset --dst cctv-object-dataset-coco

    학습 결과는 args.output_dir 아래에 checkpoint.pth / best_checkpoint.pth 로 저장된다.
    """
    try:
        from rfdetr import RFDETRLarge
    except ImportError:
        raise SystemExit(
            "rfdetr 패키지가 없습니다. `pip install rfdetr supervision` 후 재시도하세요."
        )

    import os
    dataset_dir = args.dataset
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(
            f"COCO 포맷 데이터셋 디렉토리를 찾을 수 없습니다: {dataset_dir}\n"
            "먼저 변환 스크립트를 실행하세요:\n"
            "  python util/convert_yolo_to_coco.py --src cctv-object-dataset --dst cctv-object-dataset-coco"
        )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[RF-DETR] device={device}, dataset={dataset_dir}, epochs={args.epochs}")
    print(f"[RF-DETR] batch={args.batch}, grad_accum={args.grad_accum}, lr={args.lr}")
    print(f"[RF-DETR] effective batch size = {args.batch * args.grad_accum}")

    model = RFDETRLarge()
    model.train(
        dataset_dir=dataset_dir,
        epochs=args.epochs,
        batch_size=args.batch,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        output_dir=args.output_dir,
        device=device,
    )


if __name__ == "__main__":
    if args.model == "road":
        train_road_model()
    elif args.model == "crosswalk":
        train_crosswalk_model()
    elif args.model == "object":
        train_object_model()
    elif args.model == "rfdetr-object":
        train_rfdetr_object_model()
