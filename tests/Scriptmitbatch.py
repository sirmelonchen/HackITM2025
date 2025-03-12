import cv2
import torch
import numpy as np
from tqdm import tqdm
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
from model import RRDBNet  # ESRGAN Super-Resolution Modell

# CUDA prüfen
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Modell laden
model_path = "RealESRGAN_x4plus.pth"
model = RRDBNet(3, 3, 64, 23, gc=32).to(device)
model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
model.eval()
model.half()  # FP16 für schnellere Inferenzen

# Video einlesen
video_path = "input_video.mp4"
cap = cv2.VideoCapture(video_path)

# Video-Parameter abrufen
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if frame_width == 0 or frame_height == 0 or fps == 0:
    print("Fehler: Ungültige Video-Parameter.")
    exit()

# Output-Video vorbereiten
out = cv2.VideoWriter("enhanced_video.mp4",
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      fps,
                      (frame_width * 2, frame_height * 2))  # 2x Upscaling

batch_size = 4  # Frames gleichzeitig verarbeiten
frame_list = []
pbar = tqdm(total=frame_count, desc="Processing Frames")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # OpenCV BGR → RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_list.append(frame)
    
    if len(frame_list) == batch_size:
        # Bilder in Tensoren umwandeln
        input_tensors = torch.stack([ToTensor()(Image.fromarray(f)) for f in frame_list]).to(device).half()
        
        with torch.no_grad():
            output_tensors = model(input_tensors)
            output_tensors = torch.clamp(output_tensors, 0, 1)
        
        for i in range(batch_size):
            output_image = ToPILImage()(output_tensors[i].cpu())
            output_frame = cv2.cvtColor(np.array(output_image), cv2.COLOR_RGB2BGR)
            out.write(output_frame)
            pbar.update(1)
        
        frame_list = []  # Batch zurücksetzen

# Falls noch Frames übrig sind, verarbeiten
if len(frame_list) > 0:
    input_tensors = torch.stack([ToTensor()(Image.fromarray(f)) for f in frame_list]).to(device).half()
    
    with torch.no_grad():
        output_tensors = model(input_tensors)
    
    for i in range(len(frame_list)):
        output_image = ToPILImage()(output_tensors[i].cpu())
        output_frame = cv2.cvtColor(np.array(output_image), cv2.COLOR_RGB2BGR)
        out.write(output_frame)
        pbar.update(1)

# Ressourcen freigeben
cap.release()
out.release()
pbar.close()
print("Video-Verbesserung abgeschlossen: enhanced_video.mp4")

