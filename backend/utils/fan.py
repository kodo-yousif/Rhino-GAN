import torch
import torch.nn.functional as F
import face_alignment
import cv2

fa_model = None

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, device="cuda", flip_input=False)
fa_model = fa.face_alignment_net.eval()

def soft_argmax_2d(heatmaps, beta=100):
    B, C, H, W = heatmaps.shape
    heatmaps = heatmaps.view(B * C, H * W)
    heatmaps = F.softmax(heatmaps * beta, dim=1)

    coords = torch.stack(torch.meshgrid(
        torch.linspace(0, W - 1, W, device=heatmaps.device),
        torch.linspace(0, H - 1, H, device=heatmaps.device),
        indexing='xy'
    ), dim=-1).reshape(-1, 2)  # shape: (H*W, 2)

    coords = coords.unsqueeze(0).to(heatmaps.dtype)  # (1, H*W, 2)
    out = torch.bmm(heatmaps.unsqueeze(1), coords.repeat(heatmaps.shape[0], 1, 1))  # (B*C, 1, 2)
    out = out.squeeze(1).reshape(B, C, 2)  # (B, C, 2)
    return out

def init_fan(device):
    global fa_model
    if fa_model is None:
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, device=device, flip_input=False)
        fa_model = fa.face_alignment_net.eval()


def extract_landmarks_from_tensor(
    tensor_image: torch.Tensor,
    device: str = "cpu",
):
    init_fan(device)

    if tensor_image.dim() == 4:
        tensor_image = tensor_image[0]

    tensor_image = ((tensor_image + 1) / 2).clamp(0, 1)
    orig_h, orig_w = tensor_image.shape[1:]  # CHW

    input_tensor = F.interpolate(tensor_image.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False)

    with torch.set_grad_enabled(True):
        heatmaps = fa_model(input_tensor.to(device))  # (1, 68, 64, 64)
        landmarks_2d = soft_argmax_2d(heatmaps)[0]     # (68, 2)

    # Scale from 64x64 to original image size
    scale_x = orig_w / 64
    scale_y = orig_h / 64
    landmarks_2d[:, 0] *= scale_x
    landmarks_2d[:, 1] *= scale_y

    # For 3D, we set Z = max value of each heatmap (proxy depth)
    z = heatmaps[0].view(heatmaps.shape[1], -1).max(dim=1).values.unsqueeze(1)
    landmarks = torch.cat([landmarks_2d, z], dim=1)  # (68, 3)
    assert landmarks.requires_grad, "Landmarks are not differentiable!"

    return landmarks

def extract_face_landmarks(
    image_path: str,
    device: str = "cpu",
):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at: {image_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor_image = torch.tensor(image_rgb).permute(2, 0, 1).float() / 255.0
    return extract_landmarks_from_tensor(tensor_image, device=device)
