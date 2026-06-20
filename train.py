import os
# RF-DETR(torch) + conda 환경에서 libiomp5md.dll 중복 로드(OMP Error #15) 회피
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='road', help='Model type to train: road | crosswalk | object | rfdetr-object')
parser.add_argument('--dataset', type=str, default='cctv-object-dataset-coco', help='COCO format dataset dir (rfdetr-object only)')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch', type=int, default=4, help='Batch size per step (rfdetr-object only)')
parser.add_argument('--grad-accum', type=int, default=4, help='Gradient accumulation steps (rfdetr-object only)')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (rfdetr-object only)')
parser.add_argument('--output-dir', type=str, default='runs/detect/rfdetr-object', help='Checkpoint output dir (rfdetr-object only)')
parser.add_argument('--rfdetr-size', type=str, default='nano',
                    choices=['nano', 'small', 'medium', 'base', 'large'],
                    help='RF-DETR 모델 크기 (rfdetr-object only)')
parser.add_argument('--resolution', type=int, default=None, help='입력 해상도 override (rfdetr-object only)')
parser.add_argument('--early-stopping', action='store_true', help='검증 mAP 정체 시 조기 종료 (rfdetr-object only)')
parser.add_argument('--no-run-test', action='store_true', help='학습 종료 후 test 스플릿 평가 생략 (rfdetr-object only)')
args = parser.parse_args()

def train_road_model():
    from ultralytics import YOLO

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
    from ultralytics import YOLO

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
    from ultralytics import YOLO

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
        pip install rfdetr supervision pycocotools
        # AI-Hub + Roboflow 통합 COCO 데이터셋 빌드
        python util/build_aihub_coco.py --ratio 2.0 --seed 42

    학습 결과는 args.output_dir 아래에 checkpoint_best_total.pth / checkpoint_best_ema.pth 로 저장된다.
    """
    SIZE_TO_CLASS = {
        'nano': 'RFDETRNano', 'small': 'RFDETRSmall', 'medium': 'RFDETRMedium',
        'base': 'RFDETRBase', 'large': 'RFDETRLarge',
    }
    try:
        import rfdetr
        ModelClass = getattr(rfdetr, SIZE_TO_CLASS[args.rfdetr_size])
    except ImportError:
        raise SystemExit(
            "rfdetr 패키지가 없습니다. `pip install rfdetr supervision pycocotools` 후 재시도하세요."
        )

    dataset_dir = args.dataset
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(
            f"COCO 포맷 데이터셋 디렉토리를 찾을 수 없습니다: {dataset_dir}\n"
            "먼저 데이터셋 빌드 스크립트를 실행하세요:\n"
            "  python util/build_aihub_coco.py --ratio 2.0 --seed 42"
        )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[RF-DETR] size={args.rfdetr_size} ({SIZE_TO_CLASS[args.rfdetr_size]}), device={device}")
    print(f"[RF-DETR] dataset={dataset_dir}, epochs={args.epochs}, output={args.output_dir}")
    print(f"[RF-DETR] batch={args.batch}, grad_accum={args.grad_accum} (effective {args.batch * args.grad_accum}), lr={args.lr}")

    model_kwargs = {}
    if args.resolution is not None:
        model_kwargs['resolution'] = args.resolution
    model = ModelClass(**model_kwargs)

    model.train(
        dataset_dir=dataset_dir,
        epochs=args.epochs,
        batch_size=args.batch,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        output_dir=args.output_dir,
        device=device,
        early_stopping=args.early_stopping,
        run_test=not args.no_run_test,
        tensorboard=True,
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
