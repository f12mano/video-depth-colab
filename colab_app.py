# Copyright (2025) Bytedance Ltd. and/or its affiliates
# Adapted for Google Colab Cloud Deployment with Free T4 GPU & Gradio Share
import os
import gc
import cv2
import shutil
import tempfile
import urllib.request
import numpy as np
import torch
import gradio as gr
from PIL import Image, ImageDraw
import mediapipe as mp
import imageio
import matplotlib
import matplotlib.cm as cm
from video_depth_anything.video_depth import VideoDepthAnything

# Explicitly disable decord in dc_utils to prevent C++ threaded_decoder.cc assertions
import utils.dc_utils
utils.dc_utils.DECORD_AVAILABLE = False

# Kill xformers globally in ALL modules that import it, before any model is loaded.
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

    out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_bias, dropout_p=self.attn_drop.p if self.training else 0.0, scale=self.scale
    )
    out = out.transpose(1, 2).reshape(B, N, C)
    out = self.proj(out)
    out = self.proj_drop(out)
    return out

dinov2_attn.MemEffAttention.forward = patched_mem_eff_forward

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
sam2_predictor = None

def ensure_even(value: int) -> int:
    return value if value % 2 == 0 else value + 1

def ensure_face_landmarker_model() -> str:
    os.makedirs('./models', exist_ok=True)
    model_path = './models/face_landmarker.task'
    if not os.path.exists(model_path):
        url = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task'
        print(f"[+] Downloading MediaPipe Face Landmarker from {url}...")
        urllib.request.urlretrieve(url, model_path)
    return model_path

def get_sam2_predictor(device: str = "cuda"):
    global sam2_predictor
    if sam2_predictor is not None:
        return sam2_predictor
    print("[+] Building Meta SAM 2.1 Tiny video predictor from HuggingFace...")
    from sam2.build_sam import build_sam2_video_predictor_hf
    sam2_predictor = build_sam2_video_predictor_hf("facebook/sam2.1-hiera-tiny", device=device)
    return sam2_predictor

def save_video_custom(frames, output_video_path, fps=24, is_depths=False, colormap_choice="Inferno (Standard)"):
    writer = imageio.get_writer(output_video_path, fps=fps, macro_block_size=1, codec='libx264', ffmpeg_params=['-crf', '18'])
    if is_depths:
        is_gray = "grayscale" in colormap_choice.lower()
        cm_name = "viridis" if "viridis" in colormap_choice.lower() else "inferno"

        if not is_gray:
            try:
                cm_obj = matplotlib.colormaps[cm_name]
                lut = np.array(cm_obj.colors)
            except Exception:
                lut = np.array(cm.get_cmap(cm_name).colors)

        d_min, d_max = frames.min(), frames.max()
        denom = max(float(d_max - d_min), 1e-6)

        for i in range(frames.shape[0]):
            depth = frames[i]
            depth_norm = ((depth - d_min) / denom * 255.0).clip(0, 255).astype(np.uint8)
            if not is_gray:
                depth_vis = (lut[depth_norm] * 255).astype(np.uint8)
            else:
                depth_vis = np.repeat(depth_norm[:, :, None], 3, axis=2)
            writer.append_data(depth_vis)
    else:
        for i in range(frames.shape[0]):
            writer.append_data(frames[i])
    writer.close()

def draw_face_mesh_on_image(detector, orig_rgb_frame: np.ndarray, target_im: Image.Image, timestamp_ms: int, Connections) -> Image.Image:
    h, w = orig_rgb_frame.shape[:2]
    ch = int(h * 0.60)
    crop = np.ascontiguousarray(orig_rgb_frame[:ch, :])

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop)
    result = detector.detect_for_video(mp_image, timestamp_ms)

    draw = ImageDraw.Draw(target_im)
    if result.face_landmarks:
        for face in result.face_landmarks:
            pts = [(round(l.x * w), round(l.y * ch)) for l in face]

            for conn in Connections.FACE_LANDMARKS_TESSELATION:
                draw.line([pts[conn.start], pts[conn.end]], fill=(50, 170, 205), width=1)
            for conn in Connections.FACE_LANDMARKS_LIPS:
                draw.line([pts[conn.start], pts[conn.end]], fill=(255, 135, 100), width=2)
            for conn in Connections.FACE_LANDMARKS_LEFT_EYE:
                draw.line([pts[conn.start], pts[conn.end]], fill=(130, 255, 150), width=2)
            for conn in Connections.FACE_LANDMARKS_RIGHT_EYE:
                draw.line([pts[conn.start], pts[conn.end]], fill=(130, 255, 150), width=2)

    return target_im

def run_sam2_video_segmentation(frames_rgb: np.ndarray, num_subjects: int = 1, device: str = "cuda") -> list:
    predictor = get_sam2_predictor(device=device)
    temp_dir = tempfile.mkdtemp(prefix="sam2_frames_")
    try:
        for i, f in enumerate(frames_rgb):
            cv2.imwrite(os.path.join(temp_dir, f"{i:05d}.jpg"), cv2.cvtColor(f, cv2.COLOR_RGB2BGR))

        with torch.inference_mode():
            state = predictor.init_state(video_path=temp_dir)
            H, W = frames_rgb[0].shape[:2]

            if num_subjects == 1:
                points = np.array([
                    [int(W * 0.50), int(H * 0.30)],
                    [int(W * 0.50), int(H * 0.65)],
                    [int(W * 0.35), int(H * 0.70)],
                    [int(W * 0.65), int(H * 0.70)],
                ], dtype=np.float32)
                labels = np.array([1, 1, 1, 1], dtype=np.int32)
                predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=1,
                    points=points,
                    labels=labels,
                )
            else:
                p1 = np.array([
                    [int(W * 0.28), int(H * 0.28)],
                    [int(W * 0.28), int(H * 0.55)],
                    [int(W * 0.22), int(H * 0.75)],
                    [int(W * 0.34), int(H * 0.75)],
                    [int(W * 0.50), int(H * 0.50)],
                ], dtype=np.float32)
                l1 = np.array([1, 1, 1, 1, 0], dtype=np.int32)
                predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=1,
                    points=p1,
                    labels=l1,
                )

                p2 = np.array([
                    [int(W * 0.78), int(H * 0.28)],
                    [int(W * 0.78), int(H * 0.55)],
                    [int(W * 0.72), int(H * 0.75)],
                    [int(W * 0.84), int(H * 0.75)],
                    [int(W * 0.50), int(H * 0.50)],
                ], dtype=np.float32)
                l2 = np.array([1, 1, 1, 1, 0], dtype=np.int32)
                predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=2,
                    points=p2,
                    labels=l2,
                )

            masks = {}
            for frame_idx, obj_ids, video_res_masks in predictor.propagate_in_video(state):
                if isinstance(video_res_masks, torch.Tensor):
                    combined_mask = (video_res_masks > 0.0).any(dim=0).cpu().numpy().squeeze()
                else:
                    combined_mask = (video_res_masks > 0.0).any(axis=0).squeeze()
                masks[frame_idx] = combined_mask

            return [masks[i] for i in range(len(frames_rgb))]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def apply_face_mesh_overlay(frames_rgb: np.ndarray, fps: int = 24, num_faces: int = 4) -> np.ndarray:
    model_path = ensure_face_landmarker_model()
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode
    Connections = mp.tasks.vision.FaceLandmarksConnections

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        num_faces=num_faces,
        min_face_detection_confidence=0.15,
        min_face_presence_confidence=0.15,
        min_tracking_confidence=0.15,
    )

    out_frames = []
    with FaceLandmarker.create_from_options(options) as detector:
        for i, frame in enumerate(frames_rgb):
            target_im = Image.fromarray(frame)
            timestamp_ms = round(i * 1000 / max(fps, 1))
            draw_face_mesh_on_image(detector, frame, target_im, timestamp_ms, Connections)
            out_frames.append(np.array(target_im))

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
    pipeline_mode: str = "Sirio Exact (SAM 2.1 Depth Subject over Real Room + Face Mesh)",
    subject_count: str = "1 Person (Centered)",
    depth_colormap: str = "Inferno (Standard Warm)",
    encoder_choice: str = 'vits (Small - Fast & Safe for T4)',
    target_fps: int = 24,
    max_res: int = 960,
    max_len: int = 228,
    progress=gr.Progress(track_tqdm=True),
):
    if not input_video or not os.path.exists(input_video):
        raise gr.Error("Please upload or provide a valid video file.")

    encoder = 'vits' if 'vits' in encoder_choice.lower() else 'vitl'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_subs = 2 if "2" in subject_count else 1

    output_dir = './outputs'
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_video))[0]

    progress(0.1, desc="Decoding and normalizing video frames...")
    frames, computed_fps = read_video_frames_robust(
        input_video,
        process_length=max_len,
        target_fps=target_fps,
        max_res=max_res,
    )

    guidance_video_path = os.path.join(output_dir, f"{base_name}_guidance_preview.mp4")
    depth_only_path = os.path.join(output_dir, f"{base_name}_depth_only.mp4")

    if pipeline_mode == "Fast (Face Mesh Only)":
        progress(0.4, desc="Tracking facial landmarks & drawing Face Mesh...")
        mesh_frames = apply_face_mesh_overlay(frames, fps=computed_fps, num_faces=4)
        progress(0.9, desc="Encoding Face Mesh preview video...")
        save_video_custom(mesh_frames, guidance_video_path, fps=computed_fps)
        return [guidance_video_path, guidance_video_path]

    progress(0.25, desc=f"Loading {encoder} depth model...")
    model = get_or_load_model(encoder=encoder, device=device)

    progress(0.45, desc="Computing consistent temporal depth...")
    with torch.no_grad():
        depths, out_fps = model.infer_video_depth(
            frames,
            computed_fps,
            input_size=518,
            device=device,
        )

    save_video_custom(depths, depth_only_path, fps=out_fps, is_depths=True, colormap_choice=depth_colormap)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    if pipeline_mode == "Raw Depth Map Only":
        processed_video_path = os.path.join(output_dir, f"{base_name}_src.mp4")
        save_video_custom(frames, processed_video_path, fps=out_fps)
        return [processed_video_path, depth_only_path]

    cap_d = cv2.VideoCapture(depth_only_path)
    depth_frames = []
    while cap_d.isOpened():
        ret, f_d = cap_d.read()
        if not ret: break
        depth_frames.append(cv2.cvtColor(f_d, cv2.COLOR_BGR2RGB))
    cap_d.release()
    depth_frames_arr = np.stack(depth_frames, axis=0)

    if "Sirio Exact" in pipeline_mode:
        progress(0.65, desc=f"Running Meta SAM 2.1 Tiny to segment {num_subs} subject(s)...")
        masks = run_sam2_video_segmentation(frames, num_subjects=num_subs, device=device)

        progress(0.80, desc="Compositing Depth subject(s) over authentic room background...")
        base_frames = []
        for i in range(len(frames)):
            m = masks[i]
            if m.shape[:2] != frames[i].shape[:2]:
                m = cv2.resize(m.astype(np.uint8), (frames[i].shape[1], frames[i].shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
            alpha = m[:, :, None]
            comp = np.where(alpha, depth_frames_arr[i], frames[i]).astype(np.uint8)
            base_frames.append(comp)
        base_frames_arr = np.stack(base_frames, axis=0)
    else:
        base_frames_arr = depth_frames_arr

    progress(0.88, desc="Overlaying MediaPipe Face Mesh landmarks on all detected faces...")
    model_path = ensure_face_landmarker_model()
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode
    Connections = mp.tasks.vision.FaceLandmarksConnections

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        num_faces=4,
        min_face_detection_confidence=0.15,
        min_face_presence_confidence=0.15,
        min_tracking_confidence=0.15,
    )

    final_frames = []
    with FaceLandmarker.create_from_options(options) as detector:
        for i in range(len(frames)):
            orig_frame = frames[i]
            target_im = Image.fromarray(base_frames_arr[i])
            timestamp_ms = round(i * 1000 / max(out_fps, 1))
            draw_face_mesh_on_image(detector, orig_frame, target_im, timestamp_ms, Connections)
            final_frames.append(np.array(target_im))

    progress(0.98, desc="Encoding final Guidance MP4 video...")
    save_video_custom(np.stack(final_frames, axis=0), guidance_video_path, fps=out_fps)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    progress(1.0, desc="Done!")
    return [guidance_video_path, depth_only_path]

def create_ui():
    custom_css = """
    #title-container { text-align: center; margin-bottom: 1.5rem; }
    #generate-btn { background: #2563eb !important; color: white !important; font-size: 1.15rem; font-weight: bold; border-radius: 8px; }
    .video-viewport {
        max-width: 440px !important;
        margin: 0 auto !important;
    }
    .video-viewport video {
        max-height: 520px !important;
        object-fit: contain !important;
        border-radius: 8px !important;
    }
    .step-box {
        border-left: 3px solid #2563eb;
        padding-left: 0.8rem;
        margin-bottom: 1rem;
    }
    .prompt-box {
        font-family: monospace;
        font-size: 0.9rem;
    }
    """

    with gr.Blocks(title="Genjutsu Studio (SAM 2.1 + VDA + Mesh)", css=custom_css, analytics_enabled=False) as demo:
        with gr.Column(elem_id="title-container"):
            gr.Markdown(
                """
                # 🎭 Genjutsu Guidance Studio — One-Click Video Guidance
                ### Plug & Play Video-to-Video Subject Replacement Pipeline
                Fuses Meta SAM 2.1 + ByteDance VDA + MediaPipe into a single deterministic guidance MP4.
                """
            )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### الخطوة 1: ارفع فيديو المصدر الحركي", elem_classes=["step-box"])
                input_video = gr.Video(
                    label="Input Source Video (9:16 or Portrait)",
                    sources=["upload"],
                    elem_classes=["video-viewport"]
                )

                gr.Markdown("### الخطوة 2: اختار نوع المشهد والمعالجة", elem_classes=["step-box"])
                subject_count = gr.Radio(
                    choices=[
                        "1 Person (Centered)",
                        "2 Persons (Duo / Side-by-Side)"
                    ],
                    value="1 Person (Centered)",
                    label="عدد الأشخاص في الكادر",
                    info="اختار 2 Persons لو الكليب فيه شخصين جنب بعض عشان يقفل عزل الاتنين سوا.",
                )

                pipeline_mode = gr.Radio(
                    choices=[
                        "Sirio Exact (SAM 2.1 Depth Subject over Real Room + Face Mesh)",
                        "Combined (Full-frame Depth + Face Mesh)",
                        "Fast (Face Mesh Only)",
                        "Raw Depth Map Only"
                    ],
                    value="Sirio Exact (SAM 2.1 Depth Subject over Real Room + Face Mesh)",
                    label="مود المعالجة التوجيهية",
                    info="Sirio Exact هو المود الحتمي لعزل الجسم والفلوس فوق خلفية الغرفة الطبيعية.",
                )

                with gr.Accordion("إعدادات متقدمة وألوان العمق", open=False):
                    depth_colormap = gr.Radio(
                        choices=[
                            "Inferno (Standard Warm)",
                            "Viridis (Cyan/Green - High Contrast for Orange Studios)",
                            "Grayscale (Black & White)"
                        ],
                        value="Inferno (Standard Warm)",
                        label="باليتة ألوان الـ Depth",
                        info="لو خلفية الاستوديو برتقالية فاقعة، اختار Viridis عشان الموديل يميز التباين.",
                    )
                    max_len = gr.Slider(
                        minimum=24, maximum=500, value=228, step=12,
                        label="عدد الفريمات (228 فريم = 9.5 ثواني بالضبط @ 24fps)",
                    )
                    target_fps = gr.Slider(
                        minimum=12, maximum=30, value=24, step=1,
                        label="معدل الفريمات Target FPS",
                    )
                    max_res = gr.Slider(
                        minimum=480, maximum=1280, value=960, step=32,
                        label="أقصى دقة للعرض Max Side Resolution",
                    )
                    encoder_choice = gr.Radio(
                        choices=["vits (Small - Fast & Safe for T4)", "vitl (Large - High Precision)"],
                        value="vits (Small - Fast & Safe for T4)",
                        label="حجم موديل العمق",
                    )

                generate_btn = gr.Button("توليد فيديو التوجيه الحركي (Generate Guidance)", elem_id="generate-btn", variant="primary")

            with gr.Column(scale=1):
                gr.Markdown("### الخطوة 3: الفيديو الإرشادي النهائي (خده لمنصة التوليد)", elem_classes=["step-box"])
                guidance_video = gr.Video(
                    label="Final Guidance Video (Feed as @Video to Omni / Kling)",
                    interactive=False,
                    elem_classes=["video-viewport"]
                )
                depth_preview = gr.Video(
                    label="معاينة طبقة العمق المجردة",
                    interactive=False,
                    elem_classes=["video-viewport"]
                )

                gr.Markdown("### الخطوة 4: البرومبت الحتمي الجاهز للنسخ", elem_classes=["step-box"])
                prompt_text = gr.Textbox(
                    label="Lean Motion-Transfer Prompt (Copy & Paste to Omni / Seedance / Kling)",
                    value="Motion-transfer task. @Video is the single source of truth for everything that happens: every expression, lip movement, head turn, hand gesture, body movement, timing, speed, duration, and camera framing. Reproduce @Video frame by frame.\n\nOnly one thing changes: the person. Replace the person in @Video with the character in @Image. Face, hair, and identity come from @Image only.\n\nDo not invent anything: no new gestures, no extra movements, no added expressions, no changed timing, no new actions or story beats. If something is unclear in @Video, keep it still and neutral.\n\nThe depth colors and face mesh lines in @Video are only a technical motion track showing where the body and mouth are in each frame. Do not show them in the output.",
                    lines=8,
                    interactive=False,
                    elem_classes=["prompt-box"]
                )

        generate_btn.click(
            fn=infer_guidance_pipeline,
            inputs=[input_video, pipeline_mode, subject_count, depth_colormap, encoder_choice, target_fps, max_res, max_len],
            outputs=[guidance_video, depth_preview],
        )

    return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.queue()
    demo.launch(share=True, show_error=True)
