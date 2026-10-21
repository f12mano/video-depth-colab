# Copyright (2025) Bytedance Ltd. and/or its affiliates
# Adapted for Google Colab Cloud Deployment with Free T4 GPU & Gradio Share
import os
import gc
import torch
import gradio as gr
from video_depth_anything.video_depth import VideoDepthAnything
from utils.dc_utils import read_video_frames, save_video

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

def infer_video_depth(
    input_video: str,
    encoder_choice: str = 'vits (Small - Fast & Safe)',
    target_fps: int = 15,
    max_res: int = 1280,
    max_len: int = -1,
    grayscale: bool = False,
    progress=gr.Progress(track_tqdm=True),
):
    if not input_video or not os.path.exists(input_video):
        raise gr.Error("Please upload or provide a valid video file.")

    encoder = 'vits' if 'vits' in encoder_choice.lower() else 'vitl'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    progress(0.1, desc=f"Loading {encoder} model weights...")
    model = get_or_load_model(encoder=encoder, device=device)

    output_dir = './outputs'
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(input_video))[0]
    processed_video_path = os.path.join(output_dir, f"{base_name}_src.mp4")
    depth_vis_path = os.path.join(output_dir, f"{base_name}_depth.mp4")

    progress(0.25, desc="Decoding and resizing video frames...")
    frames, computed_fps = read_video_frames(
        input_video,
        process_length=max_len,
        target_fps=target_fps,
        max_res=max_res,
    )

    progress(0.4, desc="Running Video Depth Anything temporal inference...")
    with torch.no_grad():
        depths, out_fps = model.infer_video_depth(
            frames,
            computed_fps,
            input_size=518,
            device=device,
        )

    progress(0.85, desc="Encoding output videos...")
    save_video(frames, processed_video_path, fps=out_fps)
    save_video(depths, depth_vis_path, fps=out_fps, is_depths=True, grayscale=grayscale)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    progress(1.0, desc="Done!")
    return [processed_video_path, depth_vis_path]

def create_ui():
    custom_css = """
    #title-container { text-align: center; margin-bottom: 1rem; }
    #generate-btn { background: #2563eb !important; color: white !important; font-size: 1.1rem; }
    """

    with gr.Blocks(title="Video Depth Anything — Colab Web UI", css=custom_css, analytics_enabled=False) as demo:
        with gr.Column(elem_id="title-container"):
            gr.Markdown(
                """
                # Video Depth Anything — Cloud GPU Runner
                ### High-Consistency Video Depth Estimation on Google Colab (No Time Limits)
                Powered by ByteDance's Video Depth Anything | Running on Cloud GPU
                """
            )

        with gr.Row():
            with gr.Column(scale=1):
                input_video = gr.Video(label="Input Video (Drag & Drop here)", sources=["upload"])

                with gr.Accordion("Model & Quality Settings", open=True):
                    encoder_choice = gr.Radio(
                        choices=["vits (Small - Fast & Safe for T4)", "vitl (Large - High Precision)"],
                        value="vits (Small - Fast & Safe for T4)",
                        label="Model Architecture",
                        info="Small (vits) uses ~6.8GB VRAM and is super fast. Large (vitl) is for maximum detail.",
                    )
                    target_fps = gr.Slider(
                        minimum=-1, maximum=30, value=15, step=1,
                        label="Target FPS (-1 = original fps)",
                    )
                    max_res = gr.Slider(
                        minimum=480, maximum=1920, value=1280, step=1,
                        label="Max Side Resolution",
                    )
                    max_len = gr.Slider(
                        minimum=-1, maximum=2000, value=-1, step=1,
                        label="Max Frame Length (-1 = full video)",
                    )
                    grayscale = gr.Checkbox(
                        label="Grayscale Depth Map (pure black-to-white for ControlNet)",
                        value=False,
                    )

                generate_btn = gr.Button("Generate Video Depth", elem_id="generate-btn", variant="primary")

            with gr.Column(scale=1):
                with gr.Row():
                    processed_video = gr.Video(label="Preprocessed Video", interactive=False)
                    depth_video = gr.Video(label="Generated Depth Video", interactive=False)

        generate_btn.click(
            fn=infer_video_depth,
            inputs=[input_video, encoder_choice, target_fps, max_res, max_len, grayscale],
            outputs=[processed_video, depth_video],
        )

    return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.queue()
    # share=True creates the public .gradio.live link for remote access
    demo.launch(share=True, show_error=True)
