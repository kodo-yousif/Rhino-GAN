import os 
import dlib
from pathlib import Path
import torchvision
from utils.drive import open_url
from utils.shape_predictor import align_face
import PIL
from setting import setting


cache_dir= "./cache"

cache_dir = Path(cache_dir)
cache_dir.mkdir(parents=True, exist_ok=True)

predictor_path = open_url(
    "https://drive.google.com/uc?id=1huhv8PYpNNKbGCLOaYUjOgR1pY5pmbJx",
    cache_dir=cache_dir,
    return_path=True,
)
predictor = dlib.shape_predictor(predictor_path)

def process_face(image_path: str, ) -> None:
    image_path = Path(image_path)

    print(f"🖼 Processing {image_path.name}...")
    faces = align_face(str(image_path), predictor)

    if not faces or len(faces) == 0:
            raise RuntimeError(f"❌ No faces found in image: {image_path}")

    face = faces[0]

    face_tensor = torchvision.transforms.ToTensor()(face).unsqueeze(0).cuda()
    face_tensor = face_tensor[0].cpu().detach().clamp(0, 1)
    face = torchvision.transforms.ToPILImage()(face_tensor)

    face = face.resize((1024, 1024), PIL.Image.LANCZOS)

    aligned_face_path = os.path.join(setting.input_dir, f"{image_path.stem}.png")

    face.save(aligned_face_path)

    print(f"✅ aligned face: {aligned_face_path}")
