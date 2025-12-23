import torch
import numpy as np
import cv2
import pyautogui
import os
import time
from PIL import Image # 리사이즈를 위해 필요

# 우리가 만든 모델
from models.encoder import SplitSender
from models.decoder import SplitReceiver

# 설정
CAPTURE_SIZE = 1024  # 학습한 크기와 똑같이 설정
FPS_LIMIT = 30      # 최대 프레임 제한

def run_live_test():
    # 1. 장치 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Testing on device: {device}")

    # 2. 모델 준비
    sender = SplitSender(n_channels=3).to(device)
    receiver = SplitReceiver(n_classes=3).to(device)

    # 학습된 파일 로드
    if os.path.exists("sender_final.pth") and os.path.exists("receiver_final.pth"):
        try:
            sender.load_state_dict(torch.load("sender_final.pth", map_location=device))
            receiver.load_state_dict(torch.load("receiver_final.pth", map_location=device))
            print("✅ 학습된 모델을 로드했습니다!")
        except Exception as e:
            print(f"⚠️ 모델 로드 실패: {e}")
            print("랜덤 가중치로 시작합니다.")
    else:
        print("⚠️ 학습된 파일(.pth)이 없습니다. 회색 노이즈가 출력됩니다.")

    sender.eval()
    receiver.eval()

    print("\n=== [실시간 화면 송출 테스트] ===")
    print(f"화면 좌상단 {CAPTURE_SIZE}x{CAPTURE_SIZE} 영역을 캡처합니다.")
    print("종료하려면 화면 창을 클릭하고 'q' 키를 누르세요.\n")

    try:
        while True:
            start_time = time.time()

            # --- [1] 화면 캡처 (Source) ---
            # region=(left, top, width, height)
            screenshot = pyautogui.screenshot(region=(0, 0, CAPTURE_SIZE, CAPTURE_SIZE))
            
            # [핵심 수정] Retina 디스플레이 대응 및 모델 입력 크기 고정
            # 캡처된 이미지가 512여도 강제로 256으로 줄여버림
            if screenshot.size != (CAPTURE_SIZE, CAPTURE_SIZE):
                screenshot = screenshot.resize((CAPTURE_SIZE, CAPTURE_SIZE), Image.BILINEAR)

            screenshot = screenshot.convert('RGB')
            frame_np = np.array(screenshot)

            # Numpy array (H, W, C) -> 혹시 4채널이면 3채널로 자르기
            if frame_np.shape[2] == 4:
                frame_np = frame_np[:, :, :3]

            # Tensor 변환: (H, W, C) -> (C, H, W) -> 0~1 정규화
            frame_tensor = torch.from_numpy(frame_np).permute(2, 0, 1).float() / 255.0
            frame_tensor = frame_tensor.unsqueeze(0).to(device)

            # --- [2] 모델 파이프라인 ---
            with torch.no_grad():
                packet_bot, list_skips = sender(frame_tensor)
                restored_tensor = receiver(packet_bot, list_skips)

            # --- [3] 결과 시각화 ---
            restored_np = restored_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
            restored_np = np.clip(restored_np * 255, 0, 255).astype(np.uint8)

            input_show = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            output_show = cv2.cvtColor(restored_np, cv2.COLOR_RGB2BGR)

            # [안전장치] 만약 모델 출력이 입력과 크기가 다르면, 강제로 입력 크기에 맞춤
            # (hstack 에러 방지용)
            if input_show.shape != output_show.shape:
                output_show = cv2.resize(output_show, (input_show.shape[1], input_show.shape[0]))

            # 두 이미지를 가로로 붙여서 보여줌
            combined_view = np.hstack((input_show, output_show))
            
            cv2.imshow('Live Test', combined_view)

            # --- [4] 종료 조건 ---
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # FPS 조절
            elapsed = time.time() - start_time
            if elapsed < 1.0 / FPS_LIMIT:
                time.sleep((1.0 / FPS_LIMIT) - elapsed)

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        print("테스트 종료.")

if __name__ == "__main__":
    run_live_test()