# Copyright (2025) Bytedance Ltd. and/or its affiliates
# Adapted for Google Colab Cloud Deployment with Free T4 GPU & Gradio Share
import os
import gc
import cv2
import urllib.request
import numpy as np
import torch
import gradio as gr
from PIL import Image, ImageDraw
import mediapipe as mp
from video_depth_anything.video_depth import VideoDepthAnything
from utils.dc_utils import save_video

# Explicitly disable decord in dc_utils to prevent C++ threaded_decoder.cc assertions
import utils.dc_utils
utils.dc_utils.DECORD_AVAILABLE = False

# Kill xformers globally in ALL modules that import it, before any model is loaded.
# Each module has its own XFORMERS_AVAILABLE flag — every one must be forced False.
import torch.nn.functional as F
import video_depth_anything.dinov2_layers.attention as dinov2_attn
import video_depth_anything.dinov2_layers.block as dinov2_block
import video_depth_anything.dinov2_layers.swiglu_ffn as dinov2_swiglu
import video_depth_anything.motion_module.attention as motion_attn
import video_depth_anything.motion_module.motion_module as motion_mod

dinov2_attn.XFORMERS_AVAILABLE = False
dinov2_block.XFORMERS_AVAILABLE = False
dinov2_swiglu.XFORMERS_AVAILABLE = False
motion_attn.XFORMERS_AVAILABLE = False
motion_mod.XFORMERS_AVAILABLE = False

def patched_mem_eff_forward(self, x: torch.Tensor, attn_bias=None) -> torch.Tensor:
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
    qkv = qkv.permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]

    # Native PyTorch SDPA is memory-efficient and automatically selects the fastest backend
    out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_bias, dropout_p=self.attn_drop.p if self.training else 0.0, scale=self.scale
    )
    out = out.transpose(1, 2).reshape(B, N, C)
    out = self.proj(out)
    out = self.proj_drop(out)
    return out

dinov2_attn.MemEffAttention.forward = patched_mem_eff_forward

# Force-disable the per-instance flag in the motion module's temporal attention.
_orig_temporal_init = motion_mod.TemporalAttention.__init__

def _patched_temporal_init(self, *args, **kwargs):
    _orig_temporal_init(self, *args, **kwargs)
    self._use_memory_efficient_attention_xformers = False

motion_mod.TemporalAttention.__init__ = _patched_temporal_init

MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}

CHECKPOINTS = {
    'vits': './checkpoints/video_depth_anything_vits.pth',
    'vitl': './checkpoints/video_depth_anything_vitl.pth',
}

HF_WEIGHT_URLS = {
    'vits': 'https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth',
    'vitl': 'https://huggingface.co/depth-anything/Video-Depth-Anything-Large/resolve/main/video_depth_anything_vitl.pth',
}

current_model = None
current_encoder = None

def ensure_even(value: int) -> int:
    return value if value % 2 == 0 else value + 1

def ensure_face_landmarker_model() -> str:
    """Downloads Google MediaPipe Face Landmarker model if not cached."""
    os.makedirs('./models', exist_ok=True)
    model_path = './models/face_landmarker.task'
    if not os.path.exists(model_path):
        url = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task'
        print(f"[+] Downloading MediaPipe Face Landmarker from {url}...")
        urllib.request.urlretrieve(url, model_path)
    return model_path

def apply_face_mesh_overlay(frames_rgb: np.ndarray, fps: int = 24) -> np.ndarray:
    """
    Overlays MediaPipe Face Mesh (tesselation in cyan, lips in orange, eyes in green)
    directly onto frames_rgb (H, W, 3) in a unified, deterministic loop.
    """
    model_path = ensure_face_landmarker_model()
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode
    Connections = mp.tasks.vision.FaceLandmarksConnections

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.15,
        min_face_presence_confidence=0.15,
        min_tracking_confidence=0.15,
    )

    out_frames = []
    with FaceLandmarker.create_from_options(options) as detector:
        for i, frame in enumerate(frames_rgb):
            im = Image.fromarray(frame)
            w, h = im.size

            # Focused crop on upper body / head region for maximum landmark accuracy
            x0 = int(w * 0.15)
            y0 = 0
            cw = int(w * 0.70)
            ch = int(h * 0.70)
            crop = im.crop((x0, y0, x0 + cw, y0 + ch))

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(crop))
            timestamp_ms = round(i * 1000 / max(fps, 1))
            result = detector.detect_for_video(mp_image, timestamp_ms)

            draw = ImageDraw.Draw(im)
            if result.face_landmarks:
                for face in result.face_landmarks:
                    pts = [(round(x0 + l.x * cw), round(y0 + l.y * ch)) for l in face]

                    # Facial Tesselation Mesh (Cyan)
                    for conn in Connections.FACE_LANDMARKS_TESSELATION:
                        draw.line([pts[conn.start], pts[conn.end]], fill=(50, 170, 205), width=1)

                    # Lips Highlight (Warm Orange)
                    for conn in Connections.FACE_LANDMARKS_LIPS:
                        draw.line([pts[conn.start], pts[conn.end]], fill=(255, 135, 100), width=2)

                    # Left & Right Eyes (Lime Green)
                    for conn in Connections.FACE_LANDMARKS_LEFT_EYE:
                        draw.line([pts[conn.start], pts[conn.end]], fill=(130, 255, 150), width=2)
                    for conn in Connections.FACE_LANDMARKS_RIGHT_EYE:
                        draw.line([pts[conn.start], pts[conn.end]], fill=(130, 255, 150), width=2)

            out_frames.append(np.array(im))

    return np.stack(out_frames, axis=0)

def read_video_frames_robust(video_path: str, process_length: int = -1, target_fps: int = -1, max_res: int = -1):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0 or np.isnan(original_fps):
        original_fps = 25.0

    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    target_height = ensure_even(original_height)
    target_width = ensure_even(original_width)

    if max_res > 0 and max(original_height, original_width) > max_res:
        scale = max_res / max(original_height, original_width)
        target_height = ensure_even(round(original_height * scale))
        target_width = ensure_even(round(original_width * scale))

    fps = original_fps if target_fps <= 0 else target_fps
    stride = max(round(original_fps / fps), 1)

    frames = []
    frame_idx = 0
    collected_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % stride == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if frame.shape[0] != target_height or frame.shape[1] != target_width:
                frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
            frames.append(frame)
            collected_count += 1

            if process_length > 0 and collected_count >= process_length:
                break

        frame_idx += 1

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"No frames could be extracted from: {video_path}")

    frames_array = np.stack(frames, axis=0)
    return frames_array, int(round(fps))

def get_or_load_model(encoder: str = 'vits', device: str = 'cuda'):
    global current_model, current_encoder

    if current_model is not None and current_encoder == encoder:
        return current_model

    os.makedirs('./checkpoints', exist_ok=True)
    ckpt_path = CHECKPOINTS[encoder]

    if not os.path.exists(ckpt_path):
        url = HF_WEIGHT_URLS[encoder]
        print(f"[+] Downloading weights for {encoder} from {url}...")
        torch.hub.download_url_to_file(url, ckpt_path)

    if current_model is not None:
        del current_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"[+] Loading {encoder} model onto {device}...")
    model = VideoDepthAnything(**MODEL_CONFIGS[encoder])
    state_dict = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()

    current_model = model
    current_encoder = encoder
    return current_model

def infer_guidance_pipeline(
    input_video: str,
    pipeline_mode: str = "Combined (Depth + Face Mesh Guidance)",
    encoder_choice: str = 'vits (Small - Fast & Safe for T4)',
    target_fps: int = 24,
    max_res: int = 960,
    max_len: int = 96,  # Default to 4 seconds (96 frames @ 24fps) for deterministic PoC
    grayscale: bool = False,
    progress=gr.Progress(track_tqdm=True),
):
    if not input_video or not os.path.exists(input_video):
        raise gr.Error("Please upload or provide a valid video file.")

    encoder = 'vits' if 'vits' in encoder_choice.lower() else 'vitl'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    output_dir = './outputs'
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_video))[0]

    progress(0.1, desc="Reading and normalizing source video frames...")
    frames, computed_fps = read_video_frames_robust(
        input_video,
        process_length=max_len,
        target_fps=target_fps,
        max_res=max_res,
    )

    guidance_video_path = os.path.join(output_dir, f"{base_name}_guidance_preview.mp4")
    depth_only_path = os.path.join(output_dir, f"{base_name}_depth_only.mp4")

    # Mode 1: Pure Face Mesh
    if pipeline_mode == "Fast (Face Mesh Only)":
        progress(0.4, desc="Tracking facial landmarks & drawing Face Mesh...")
        mesh_frames = apply_face_mesh_overlay(frames, fps=computed_fps)
        progress(0.9, desc="Encoding Face Mesh preview video...")
        save_video(mesh_frames, guidance_video_path, fps=computed_fps)
        return [guidance_video_path, guidance_video_path]

    # Model Inference for Depth
    progress(0.3, desc=f"Loading {encoder} depth model...")
    model = get_or_load_model(encoder=encoder, device=device)

    progress(0.5, desc="Computing high-consistency temporal depth...")
    with torch.no_grad():
        depths, out_fps = model.infer_video_depth(
            frames,
            computed_fps,
            input_size=518,
            device=device,
        )

    # Render depth video to disk
    save_video(depths, depth_only_path, fps=out_fps, is_depths=True, grayscale=grayscale)

    if pipeline_mode == "Depth Map Only":
        processed_video_path = os.path.join(output_dir, f"{base_name}_src.mp4")
        save_video(frames, processed_video_path, fps=out_fps)
        return [processed_video_path, depth_only_path]

    # Mode 3: Combined (Depth + Face Mesh Guidance)
    progress(0.8, desc="Compositing MediaPipe Face Mesh on top of Depth...")
    # Read back the rendered depth frames to overlay face mesh
    cap_d = cv2.VideoCapture(depth_only_path)
    depth_frames = []
    while cap_d.isOpened():
        ret, f_d = cap_d.read()
        if not ret: break
        depth_frames.append(cv2.cvtColor(f_d, cv2.COLOR_BGR2RGB))
    cap_d.release()
    depth_frames_arr = np.stack(depth_frames, axis=0)

    # Track face landmarks from original frames and draw them on depth frames
    model_path = ensure_face_landmarker_model()
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode
    Connections = mp.tasks.vision.FaceLandmarksConnections

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.15,
        min_face_presence_confidence=0.15,
        min_tracking_confidence=0.15,
    )

    composite_frames = []
    with FaceLandmarker.create_from_options(options) as detector:
        for i in range(len(frames)):
            orig_im = Image.fromarray(frames[i])
            w, h = orig_im.size
            x0 = int(w * 0.15)
            y0 = 0
            cw = int(w * 0.70)
            ch = int(h * 0.70)
            crop = orig_im.crop((x0, y0, x0 + cw, y0 + ch))

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(crop))
            timestamp_ms = round(i * 1000 / max(out_fps, 1))
            result = detector.detect_for_video(mp_image, timestamp_ms)

            # Draw onto the depth frame
            target_im = Image.fromarray(depth_frames_arr[i])
            draw = ImageDraw.Draw(target_im)
            if result.face_landmarks:
                for face in result.face_landmarks:
                    pts = [(round(x0 + l.x * cw), round(y0 + l.y * ch)) for l in face]
                    for conn in Connections.FACE_LANDMARKS_TESSELATION:
                        draw.line([pts[conn.start], pts[conn.end]], fill=(50, 170, 205), width=1)
                    for conn in Connections.FACE_LANDMARKS_LIPS:
                        draw.line([pts[conn.start], pts[conn.end]], fill=(255, 135, 100), width=2)
                    for conn in Connections.FACE_LANDMARKS_LEFT_EYE:
                        draw.line([pts[conn.start], pts[conn.end]], fill=(130, 255, 150), width=2)
                    for conn in Connections.FACE_LANDMARKS_RIGHT_EYE:
                        draw.line([pts[conn.start], pts[conn.end]], fill=(130, 255, 150), width=2)

            composite_frames.append(np.array(target_im))

    progress(0.95, desc="Saving final Composite Guidance MP4...")
    save_video(np.stack(composite_frames, axis=0), guidance_video_path, fps=out_fps)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    progress(1.0, desc="Done!")
    return [guidance_video_path, depth_only_path]

def create_ui():
    custom_css = """
    #title-container { text-align: center; margin-bottom: 1rem; }
    #generate-btn { background: #2563eb !important; color: white !important; font-size: 1.1rem; }
    """

    with gr.Blocks(title="Genjutsu Video Guidance — Colab Studio", css=custom_css, analytics_enabled=False) as demo:
        with gr.Column(elem_id="title-container"):
            gr.Markdown(
                """
                # 🎭 Genjutsu Guidance Studio — Colab Cloud Runner
                ### High-Consistency Video Depth & MediaPipe Face Mesh Compositor (Zero Time Limits)
                Generate deterministic guidance videos for Omni / Seedance character replacement.
                """
            )

        with gr.Row():
            with gr.Column(scale=1):
                input_video = gr.Video(label="Input Source Video (e.g. Woman Video)", sources=["upload"])

                pipeline_mode = gr.Radio(
                    choices=[
                        "Combined (Depth + Face Mesh Guidance)",
                        "Fast (Face Mesh Only)",
                        "Depth Map Only"
                    ],
                    value="Combined (Depth + Face Mesh Guidance)",
                    label="Guidance Pipeline Mode",
                    info="Combined fuses body/money depth with facial mesh landmarks for 100% deterministic motion.",
                )

                with gr.Accordion("Advanced Model & Duration Controls", open=True):
                    max_len = gr.Slider(
                        minimum=24, maximum=500, value=96, step=24,
                        label="Frames to Process (96 frames = 4 seconds @ 24fps for PoC)",
                    )
                    target_fps = gr.Slider(
                        minimum=12, maximum=30, value=24, step=1,
                        label="Target FPS (24fps is standard for Omni / Seedance)",
                    )
                    max_res = gr.Slider(
                        minimum=480, maximum=1280, value=960, step=32,
                        label="Max Side Resolution (960p recommended)",
                    )
                    encoder_choice = gr.Radio(
                        choices=["vits (Small - Fast & Safe for T4)", "vitl (Large - High Precision)"],
                        value="vits (Small - Fast & Safe for T4)",
                        label="Depth Model Size",
                    )
                    grayscale = gr.Checkbox(
                        label="Grayscale Depth (Pure B&W)",
                        value=False,
                    )

                generate_btn = gr.Button("Generate Guidance Video", elem_id="generate-btn", variant="primary")

            with gr.Column(scale=1):
                guidance_video = gr.Video(label="Final Guidance Video (Feed this to Omni)", interactive=False)
                depth_preview = gr.Video(label="Raw Depth Layer Preview", interactive=False)

        generate_btn.click(
            fn=infer_guidance_pipeline,
            inputs=[input_video, pipeline_mode, encoder_choice, target_fps, max_res, max_len, grayscale],
            outputs=[guidance_video, depth_preview],
        )

    return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.queue()
    demo.launch(share=True, show_error=True)
