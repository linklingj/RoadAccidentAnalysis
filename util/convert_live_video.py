import requests
import numpy as np
import argparse
import shutil
import subprocess
import os

def get_cctv_url(lat, lng):
    # CCTV 탐색 범위 지정을 위해 임의로 ±1 만큼 가감
    minX = str(lng-1)
    maxX = str(lng+1)
    minY = str(lat-1)
    maxY = str(lat+1)

    # 개인key 입력
    api_call = 'https://openapi.its.go.kr:9443/cctvInfo?' \
               'apiKey=' + os.getenv("cctv_api_key") + \
               '&type=ex&cctvType=2' \
               '&minX=' + minX + \
               '&maxX=' + maxX + \
               '&minY=' + minY + \
               '&maxY=' + maxY + \
               '&getType=json'

    w_dataset = requests.get(api_call).json()
    cctv_data = w_dataset['response']['data']

    coordx_list = []
    for index, data in enumerate(cctv_data):
        xy_couple = (float(cctv_data[index]['coordy']),float(cctv_data[index]['coordx']))
        coordx_list.append(xy_couple)

    # 입력한 위경도 좌표에서 가장 가까운 위치에 있는 CCTV를 찾는 과정
    coordx_list = np.array(coordx_list)
    leftbottom = np.array((lat, lng))
    distances = np.linalg.norm(coordx_list - leftbottom, axis=1)
    min_index = np.argmin(distances)

    return cctv_data[min_index]


def download_video_as_mp4(video_url, output_path="cctv_video.mp4", timeout=60):
    """CCTV 영상 URL을 MP4 파일로 저장한다.

    우선 ffmpeg를 이용해 스트리밍 URL(m3u8 등)까지 처리하고,
    ffmpeg가 없거나 실패하면 직접 MP4 URL일 때 requests로 저장한다.
    """
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        cmd = [
            ffmpeg_path,
            "-y",
            "-i",
            video_url,
            "-t",
            str(timeout),
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            output_path,
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return output_path
        except subprocess.CalledProcessError:
            pass

    with requests.get(video_url, stream=True, timeout=10) as response:
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")
        if "mp4" not in content_type.lower() and not video_url.lower().endswith(".mp4"):
            raise RuntimeError(
                "직접 MP4 다운로드가 불가능한 URL입니다. ffmpeg 설치 후 다시 시도하세요."
            )

        with open(output_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)

    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="가까운 CCTV 영상 URL 조회 및 MP4 다운로드")
    parser.add_argument("--lat", type=float, default=36.58629, help="위도")
    parser.add_argument("--lng", type=float, default=128.186793, help="경도")
    parser.add_argument("--download", action="store_true", help="MP4 다운로드 실행")
    parser.add_argument("--output", default="input/video.mp4", help="저장할 MP4 파일 경로")
    parser.add_argument("--seconds", type=int, default=60, help="저장할 영상 길이(초)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cctv_data = get_cctv_url(args.lat, args.lng)
    print('CCTV명:', cctv_data['cctvname'])  # 가장 가까운 CCTV명
    print('CCTV 영상 URL:', cctv_data['cctvurl'])  # 가장 가까운 CCTV 영상 URL

    if args.download:
        saved_file = download_video_as_mp4(
            cctv_data['cctvurl'],
            output_path=args.output,
            timeout=args.seconds,
        )
        print('저장 완료:', saved_file)