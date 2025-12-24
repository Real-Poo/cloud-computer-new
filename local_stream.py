import torch
import numpy as np
import cv2
import mss
import os
import time

from models.encoder import SplitSender
from models.decoder import SplitReceiver

# 설정
CAPTURE_SIZE = 384
FPS_LIMIT = 120

monitor_area = {"top": 0, "left": 0, "width": CAPTURE_SIZE, "height": CAPTURE_SIZE}

def run_live_test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Testing on device: {device}")

    sender = SplitSender(n_channels=3).to(device)
    receiver = SplitReceiver(n_classes=3).to(device)

    SENDER_PATH = "sender_final.pth"
    RECEIVER_PATH = "receiver_final.pth"

    if os.path.exists(SENDER_PATH) and os.path.exists(RECEIVER_PATH):
        try:
            sender.load_state_dict(torch.load(SENDER_PATH, map_location=device))
            receiver.load_state_dict(torch.load(RECEIVER_PATH, map_location=device))
            print(f"✅ 학습된 모델 로드 완료")
        except:
            print("⚠️ 모델 로드 실패, 랜덤 가중치 사용")
    else:
        print("⚠️ 학습 파일 없음")

    sender.eval()
    receiver.eval()

    print("\n=== [MSS 초고속 실시간 송출 테스트 (FPS 표시)] ===")
    sct = mss.mss()

    # [FPS 계산을 위한 변수]
    prev_time = 0
    fps = 0

    try:
        while True:
            # 시간 측정 시작 (FPS 제한용)
            loop_start = time.time()
            t0 = time.time()

            # 1. 캡처
            sct_img = sct.grab(monitor_area)
            frame_np = np.array(sct_img)

            if frame_np.shape[0] != CAPTURE_SIZE:
                frame_np = cv2.resize(frame_np, (CAPTURE_SIZE, CAPTURE_SIZE), interpolation=cv2.INTER_LINEAR)

            frame_rgb = cv2.cvtColor(frame_np, cv2.COLOR_BGRA2RGB)

            frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
            frame_tensor = frame_tensor.unsqueeze(0).to(device)

            t1 = time.time() # 캡처 완료 시간

            # 2. 모델 추론
            with torch.no_grad():
                packet_bot, list_skips = sender(frame_tensor)
                restored_tensor = receiver(packet_bot, list_skips)

            if device.type == 'mps': torch.mps.synchronize()
            elif device.type == 'cuda': torch.cuda.synchronize()

            t2 = time.time() # 추론 완료 시간

            # 3. 시각화
            restored_np = restored_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
            restored_np = np.clip(restored_np * 255, 0, 255).astype(np.uint8)

            input_show = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            output_show = cv2.cvtColor(restored_np, cv2.COLOR_RGB2BGR)

            if input_show.shape != output_show.shape:
                output_show = cv2.resize(output_show, (input_show.shape[1], input_show.shape[0]))

            combined_view = np.hstack((input_show, output_show))

            # ==========================================
            # [추가됨] FPS 계산 및 화면 표시
            # ==========================================
            cur_time = time.time()
            if prev_time != 0:
                # 1초 / (현재시간 - 이전시간) = FPS
                fps = 1 / (cur_time - prev_time)
            prev_time = cur_time

            # 화면 왼쪽 상단에 초록색으로 FPS 글자 쓰기
            cv2.putText(combined_view, f"FPS: {fps:.1f}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # ==========================================
            
            cv2.imshow('MSS Live Test', combined_view)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            t3 = time.time() # 시각화 완료 시간

            capture_ms = (t1 - t0) * 1000
            ai_ms = (t2 - t1) * 1000
            display_ms = (t3 - t2) * 1000
            total_ms = (t3 - t0) * 1000

            print(f"FPS: {1000/total_ms:.1f} | 캡처: {capture_ms:.1f}ms | AI: {ai_ms:.1f}ms | 화면: {display_ms:.1f}ms", end='\r')

            # FPS 제한 (너무 빠르면 CPU 점유율 폭발 방지)
            # elapsed = time.time() - loop_start
            # if elapsed < 1.0 / FPS_LIMIT:
            #     time.sleep((1.0 / FPS_LIMIT) - elapsed)

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        print("종료")

if __name__ == "__main__":
    run_live_test()