import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from models.encoder import SplitSender
from models.decoder import SplitReceiver

# 설정
IMAGE_PATH = 'test_image.jpg' # 테스트할 이미지 파일명
ITERATIONS = 1000              # 학습 반복 횟수
LR = 0.001                    # 학습률

def main():
    # 1. 장치 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    print(f"Using device: {device}")

    # 2. 모델 초기화
    sender = SplitSender(n_channels=3).to(device)
    receiver = SplitReceiver(n_classes=3).to(device) # RGB 복원

    # 3. 이미지 로드 및 전처리
    try:
        raw_img = Image.open(IMAGE_PATH).convert('RGB')
    except FileNotFoundError:
        print(f"Error: '{IMAGE_PATH}' 파일을 찾을 수 없습니다.")
        return

    # 이미지를 256x256으로 줄이고 텐서로 변환
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(), # 0~1 사이 값으로 변환
    ])
    
    input_tensor = transform(raw_img).unsqueeze(0).to(device) # Batch 차원 추가 (1, 3, 256, 256)

    # 4. 최적화 도구 설정 (Sender와 Receiver를 동시에 학습)
    # 두 모델의 파라미터를 합쳐서 Optimizer에 전달
    params = list(sender.parameters()) + list(receiver.parameters())
    optimizer = optim.Adam(params, lr=LR)
    criterion = nn.MSELoss() # 픽셀 간 차이를 줄이는 손실 함수

    print(f"Start training on a single image for {ITERATIONS} steps...")
    
    # 5. 학습 루프 (Overfitting Test)
    sender.train()
    receiver.train()
    
    for i in range(ITERATIONS):
        optimizer.zero_grad()
        
        # 1. 모델 예측
        packet_bot, list_skips = sender(input_tensor)
        output_tensor = receiver(packet_bot, list_skips)
        
        # [수정된 부분] 크기 보정
        # 만약 모델 출력(output)이 입력(input)과 크기가 다르면, 입력 크기에 맞춰서 리사이즈합니다.
        if output_tensor.shape != input_tensor.shape:
            output_tensor = torch.nn.functional.interpolate(
                output_tensor, 
                size=(input_tensor.shape[2], input_tensor.shape[3]), 
                mode='bilinear', 
                align_corners=False
            )
        
        # 2. Loss 계산 (이제 두 텐서의 크기가 같으므로 에러가 안 납니다)
        loss = criterion(output_tensor, input_tensor)
        
        # 3. 역전파
        loss.backward()
        optimizer.step()
        
        if (i+1) % 50 == 0:
            print(f"Step [{i+1}/{ITERATIONS}], Loss: {loss.item():.6f}")

    # 6. 결과 확인 및 시각화
    sender.eval()
    receiver.eval()
    
    with torch.no_grad():
        packet_bot, list_skips = sender(input_tensor)
        restored_tensor = receiver(packet_bot, list_skips)

    # 텐서를 다시 이미지로 변환 (시각화용)
    # (Batch, Channel, Height, Width) -> (Height, Width, Channel)
    input_display = input_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    output_display = restored_tensor.squeeze().permute(1, 2, 0).cpu().numpy()

    # 화면 출력
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title("Original Input")
    plt.imshow(input_display)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title(f"Restored (After {ITERATIONS} steps)")
    plt.imshow(output_display)
    plt.axis('off')
    
    plt.show()
    print("Done! Check the result window.")

if __name__ == "__main__":
    main()