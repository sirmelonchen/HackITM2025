import cv2
import torch
import numpy as np
import os
import subprocess
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

# Verzeichnis für Frames erstellen
frame_dir = "frames"
os.makedirs(frame_dir, exist_ok=True)

# Verarbeitung der Frames
pbar = tqdm(total=frame_count, desc="Processing Frames")
frame_number = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Frame von BGR zu RGB konvertieren
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame)

    # Bild zu Tensor konvertieren
    input_tensor = ToTensor()(pil_image).unsqueeze(0).to(device)

    # Super-Resolution anwenden
    with torch.no_grad():
        output_tensor = model(input_tensor)

    # Tensor zurück in Bild umwandeln
    output_image = ToPILImage()(output_tensor.squeeze(0).cpu())

    # Bild speichern
    output_frame_path = os.path.join(frame_dir, f"frame_{frame_number:04d}.png")
    output_image.save(output_frame_path)
    frame_number += 1

    pbar.update(1)

# Ressourcen freigeben
cap.release()
pbar.close()
print("Frame-Verarbeitung abgeschlossen. Starte FFmpeg...")

# FFmpeg-Befehl zum Erstellen des Videos
output_video_path = "enhanced_video.mp4"
ffmpeg_cmd = [
    "ffmpeg", "-y", "-framerate", str(fps), "-i",
    os.path.join(frame_dir, "frame_%04d.png"), "-c:v", "libx264", "-crf", "18", "-preset", "slow",
    output_video_path
]

# FFmpeg ausführen
subprocess.run(ffmpeg_cmd)
print(f"Video-Verbesserung abgeschlossen: {output_video_path}")