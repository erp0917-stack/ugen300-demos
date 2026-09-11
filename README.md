# UGen300 Demos

A collection of 11 ready-to-run AI demo apps for the ASUS UGen300 AI accelerator. Every app runs fully offline on the device — no cloud and no internet connection needed at runtime.

## About UGen300

UGen300 is an ASUS AI accelerator built on the Hailo-10H processor. It delivers 40 TOPS of AI compute and runs modern AI models — from real-time object detection to vision-language models and large language models — directly on your machine, with no cloud connection required.

This repository holds example apps that show what the device can do. Think of them as a starting point: clone them, run them, and build your own ideas on top.

## Applications

The apps fall into three groups: pose and detection apps (Y series), vision-language apps (V series), and a few standalone tools.

### Pose & Detection (Y series)

| App | What it does |
|-----|--------------|
| **Y1 — Real-time Detection** | Detects objects in your camera feed in real time. |
| **Y2 — Raise Hand to Flip** | Raise your hand to flip slides or pages, hands-free. |
| **Y3 — Air Instrument** | Turns your body movements into musical sounds. |
| **Y4 — Raise Hand Counter** | Counts how many people raise their hands. |
| **Y5 — Yoga Coach** | Guides you through yoga poses and checks your form. |

### Vision-Language (V series)

| App | What it does |
|-----|--------------|
| **V1 — Ask About Image** | Point the camera at something and ask questions about it. |
| **V2 — Outfit Advisor** | Looks at your outfit and gives styling advice. |
| **V3 — Fridge Chef** | Show it your fridge and it suggests recipes. |

### Standalone Tools

| App | What it does |
|-----|--------------|
| **Missing Check** | Watches a scene and flags missing or extra items (a red alert when something is missing, a yellow notice when there is more than expected). |
| **Face Report** | Generates a fun, full-page face report using two AI models working together. |
| **Meeting Notes** | Records a meeting, transcribes it, and writes a summary. Comes with both a command-line version and a desktop (GUI) version. |

### Everyday & Creative (added 2026-09)

| App | What it does |
|-----|--------------|
| **WakeUp** | An alarm you can only silence by doing five squats in front of the camera (pose estimation). Works on UGen200 too. |
| **Calories** | Point the camera at a meal; the on-device VLM lists the items and estimates calories and protein, with a daily total. |
| **OfflineChat** | A ChatGPT-style chat window streaming from Qwen3-1.7B (or Llama 3.2 1B) running entirely on the UGen300 — unplug the network and it keeps working. |
| **GestureCaption** | Shows the hand gesture you make (1–5, OK, thumbs-up, heart, rock...) as large captions using 21 hand landmarks — rule based, no training. |
| **FruitNinja** | Your index finger is the blade: slice fruit, avoid bombs, 30-second rounds. |
| **AirBand** | Drums, piano and guitar zones on screen — several people can play at once with their hands. |
| **PlateGate** | Vehicle detection + PaddleOCR reads the licence plate, checks a whitelist and animates a parking barrier (`--image` works without a real car). |
| **TrafficCount** | Tracks people and vehicles crossing a line and shows per-class counts and flow per minute (`--video` for recorded footage). |
| **HeadCount** | Real-time person counter with freeze, peak and a 60-second trend — pan the camera across a room. |
| **EdgeVsCloud** | Side-by-side dashboard: measured UGen300 latency/fps vs measured network RTT plus cloud cost and power estimates. |
| **PhotoRestore** | Drag in an old photo: brighten, denoise and 2x upscale on the accelerator, then compare before/after with a slider. Works on UGen200 too. |

## Requirements

### Hardware
- ASUS UGen300 AI accelerator (Hailo-10H, 40 TOPS, 8 GB)
- A Windows PC with a free USB port
- A webcam (most apps use the camera; Meeting Notes also uses a microphone)

### Software
- Windows 11
- Python 3.10
- HailoRT runtime and driver (version 5.3.x recommended)
- The `hailo_platform` Python package (included with HailoRT — see Setup below)

## Setup

Follow these steps in order.

### 1. Install HailoRT and the Hailo driver

The apps talk to the UGen300 device through HailoRT. This is **not** a pip package, so you install it separately.

1. Download HailoRT for the UGen300 from the [ASUS UGen300 product support page](https://www.asus.com/motherboards-components/ai-accelerator/ugen/ugen300-usb-8g/helpdesk_download?model2Name=UGen300-USB-8G), or from the Hailo Developer Zone.
2. Install the runtime and the USB driver.
3. HailoRT includes the `hailo_platform` Python package. Make sure it is installed in the same Python 3.10 environment you will use to run the apps.

### 2. Get the code

Clone this repository, or download it as a ZIP file and unzip it:

```bash
git clone https://github.com/erp0917-stack/ugen300-demos.git
cd ugen300-demos
```

### 3. Download the AI models

The AI models are large, so they are **not** included in this repository. You download them once and place each file in the right folder.

There are 5 model files. Download each one from the link below:

| Model file | Download link | Used by |
|------------|---------------|---------|
| `yolov8m.hef` | `https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.3.0/hailo10h/yolov8m.hef` | Y1, Missing Check |
| `yolov8s_pose.hef` | `https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.3.0/hailo10h/yolov8s_pose.hef` | Y2, Y3, Y4, Y5 |
| `Whisper-Base.hef` | `https://dev-public.hailo.ai/v5.2.0/blob/Whisper-Base.hef` | Meeting Notes |
| `Qwen2.5-1.5B-Instruct.hef` | `https://dev-public.hailo.ai/v5.2.0/blob/Qwen2.5-1.5B-Instruct.hef` | Face Report, Meeting Notes |
| `Qwen2-VL-2B-Instruct.hef` | `https://dev-public.hailo.ai/v5.2.0/blob/Qwen2-VL-2B-Instruct.hef` | V1, V2, V3, Face Report |
| `Qwen3-1.7B-Instruct.hef` | `https://dev-public.hailo.ai/v5.3.0/blob/Qwen3-1.7B-Instruct.hef` | OfflineChat |
| `Llama3.2-1B-Instruct.hef` | `https://dev-public.hailo.ai/v5.3.0/blob/Llama3.2-1B-Instruct.hef` | OfflineChat (optional) |
| `zero_dce.hef` | `https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo10h/zero_dce.hef` | PhotoRestore |
| `dncnn_color_blind.hef` | `https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo10h/dncnn_color_blind.hef` | PhotoRestore |
| `real_esrgan_x2.hef` | `https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo10h/real_esrgan_x2.hef` | PhotoRestore |

Where to put each file:

- **The three large GenAI models** (`Whisper-Base.hef`, `Qwen2.5-1.5B-Instruct.hef`, and `Qwen2-VL-2B-Instruct.hef`) go into the shared `models/` folder.
- **`yolov8m.hef`** goes into both the `Y1/` folder and the `MissingCheck/` folder — one copy in each.
- **`yolov8s_pose.hef`** goes into each of the `Y2/`, `Y3/`, `Y4/`, `Y5/` and `WakeUp/` folders — one copy in each.
- **`Qwen3-1.7B-Instruct.hef`** (and optionally `Llama3.2-1B-Instruct.hef`) go into `models/` for OfflineChat.
- **`zero_dce.hef`, `dncnn_color_blind.hef`, `real_esrgan_x2.hef`** go into `PhotoRestore/`.
- **`yolov8s.hef`** goes into `Traffic/` and `EdgeVsCloud/`; **`yolov8m.hef`** into `HeadCount/`; **`yolov8s_pose.hef` + `hand_landmark_lite.hef`** into `HandDemos/`; **`paddle_ocr_v5_mobile_detection.hef` + `paddle_ocr_v5_mobile_recognition.hef`** into `Traffic/` (all from the Hailo Model Zoo v5.4.0 HAILO10H list).
- AirBand additionally needs `pip install pygame`.

> **Note on model versions:** The YOLO models come from the Hailo Model Zoo (compiled for v5.3.0) and the GenAI models come from Hailo's public model server (v5.2.0). Both sets work together on the UGen300. We recommend installing HailoRT 5.3.x, which runs all of these models.

### 4. Install Python dependencies

Each app has its own `requirements.txt`. Open the folder of the app you want to run, then install its dependencies:

```bash
cd Y1
pip install -r requirements.txt
```

Repeat this for any other app you want to try. All apps share `opencv-python` and `numpy`; some need a few extra packages, which their `requirements.txt` will install for you.

## Running the Apps

There are two ways to start an app.

### The easy way: one-click launchers

The `launchers/` folder includes a `.bat` shortcut for every app. Just double-click the one you want, and the app starts. You can also start any app directly from the command line.

### The manual way: run from the command line

Open a terminal in the app's folder and run its entry point. Here are the commands for every app:

| App | Folder | Command |
|-----|--------|---------|
| Y1 — Real-time Detection | `Y1/` | `python main.py --hef yolov8m.hef --source 0` |
| Y2 — Raise Hand to Flip | `Y2/` | `python main.py --hef yolov8s_pose.hef --source 0` |
| Y3 — Air Instrument | `Y3/` | `python air_instrument.py --hef yolov8s_pose.hef --source 0` |
| Y4 — Raise Hand Counter | `Y4/` | `python hand_raise_counter.py --hef yolov8s_pose.hef --source 0` |
| Y5 — Yoga Coach | `Y5/` | `python yoga_coach.py --hef yolov8s_pose.hef --source 0` |
| V1 — Ask About Image | `V1/` | `python vlm_chat.py --source 0` |
| V2 — Outfit Advisor | `V2/` | `python outfit_advisor.py --source 0` |
| V3 — Fridge Chef | `V3/` | `python fridge_chef.py --source 0` |
| Missing Check | `MissingCheck/` | `python missing_check.py` |
| Face Report | `FACE1/` | `python face_report.py --camera --lang tw` |
| Meeting Notes (CLI) | `WQ1/` | `python meeting_summary.py` |
| Meeting Notes (GUI) | `WQ1/` | `python meeting_gui.py` |

`--source 0` means your default webcam. If you have more than one camera, try `--source 1`, `--source 2`, and so on.

## Project Structure

```
ugen300-demos/
├── Y1/             Real-time Detection
├── Y2/             Raise Hand to Flip
├── Y3/             Air Instrument
├── Y4/             Raise Hand Counter
├── Y5/             Yoga Coach
├── V1/             Ask About Image
├── V2/             Outfit Advisor
├── V3/             Fridge Chef
├── MissingCheck/   Missing Check
├── FACE1/          Face Report
├── WQ1/            Meeting Notes
├── models/         Shared model files (.hef)
├── launchers/      One-click .bat shortcuts
└── README.md
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
