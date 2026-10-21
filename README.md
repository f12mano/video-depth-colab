# Video Depth Anything — One-Click Google Colab Runner

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/f12mano/video-depth-colab/blob/main/video_depth_colab.ipynb)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A one-click, production-ready cloud deployment for ByteDance's **[Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything)**. Runs on free Google Colab cloud GPUs (Nvidia T4) and provides a clean Gradio Web UI with a public URL.

---

## Why this exists

1. **Bypasses Hugging Face Limits:** Hugging Face ZeroGPU restricts runs to 60–120 seconds. Long videos trigger `ZeroGPU illegal duration` errors. This Colab setup has **no artificial duration cap**.
2. **Zero Local GPU Stress:** Older GPUs (e.g. AMD Radeon HD 7900 / R9 200 series) lack modern WebGPU support. This offloads 100% of the computation to Google's cloud servers.
3. **Interactive Web UI:** No terminal commands needed. Drag and drop your video into your browser and get the processed depth map MP4 immediately.

---

## Quick Start (3 Steps)

1. **Open in Colab:**
   Click the badge above or open [`video_depth_colab.ipynb`](https://colab.research.google.com/github/f12mano/video-depth-colab/blob/main/video_depth_colab.ipynb).
2. **Run All:**
   Ensure the runtime is set to **T4 GPU** (`Runtime > Change runtime type > T4 GPU`), then click **Runtime > Run all** (or press `Ctrl + F9`).
3. **Open the Web UI:**
   Scroll to the final cell. Click the generated public link:
   ```
   Running on public URL: https://xxxxxxxxxxxxxxxx.gradio.live
   ```
   Drag your video in, select the model (`vits` Small recommended), and click **Generate Video Depth**.

---

## Features

- **Models:**
  - `vits (Small)`: ~28M parameters, uses ~6.8GB VRAM. Super fast and stable on free T4 GPUs.
  - `vitl (Large)`: ~381M parameters. Maximum temporal detail.
- **Settings:**
  - Target FPS (default 15 for AI generation, or keep original).
  - Max Side Resolution (480p to 1080p).
  - Grayscale option for direct ControlNet / Stable Diffusion / Kling guidance.
- **Automatic VRAM Cleanup:** Explicit garbage collection and `torch.cuda.empty_cache()` between runs to prevent out-of-memory errors on long videos.

---

## Credits

- Original Architecture: [Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything) (CVPR 2025 Highlight) by ByteDance.
- Model Weights: [Hugging Face Depth Anything](https://huggingface.co/depth-anything).
