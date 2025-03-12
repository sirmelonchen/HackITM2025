# Version: 0.1
# Letzte Änderung: 13/03/2025, 12:52
# Letztes Problem: Video wird nicht gespeichert, Scipt capt bei 15it/s.  

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

# Video einlesen
video_path = "short.mp4"
cap = cv2.VideoCapture(video_path)

# Video-Parameter abrufen 
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Output-Video vorbereiten
out = cv2.VideoWriter("enhanced_video.mkv",
    cv2.VideoWriter_fourcc(*'h264'),
    fps,
    (frame_width * 2, frame_height * 2))  # 2x Upscaling

# Verarbeitung der Frames
pbar = tqdm(total=frame_count, desc="Processing Frames")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Frame von BGR zu RGB konvertieren
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame)

    # Bild zu Tensor konvertieren
    input_tensor = (ToTensor()(pil_image) * 2 - 1).unsqueeze(0).to(device)

    # Super-Resolution anwenden
    with torch.no_grad():
        output_tensor = model(input_tensor)

    # Tensor zurück in Bild umwandeln
    output_image = ToPILImage()(output_tensor.squeeze(0).cpu())

    # Bild zurück zu OpenCV (RGB → BGR)
    output_frame = cv2.cvtColor(np.array(output_image), cv2.COLOR_RGB2BGR)
    output_image.show()

    # Frame ins neue Video schreiben
    out.write(output_frame)
    pbar.update(1)

# Ressourcen freigeben
cap.release()
out.release()
pbar.close()
print("Video-Verbesserung abgeschlossen: enhanced_video.mp4")
