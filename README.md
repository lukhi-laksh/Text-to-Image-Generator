# 🖼️ Text-to-Image Generator using SDXL (Jupyter Lab)

A **professional, research-friendly Text-to-Image generation project** built using **Stable Diffusion XL (SDXL)**, **PyTorch**, and **Diffusers**, designed to run smoothly inside **Jupyter Lab**.

This project focuses on:

* Clean architecture
* Efficient GPU/CPU usage
* Simple two-cell workflow
* High-quality image generation

---

## 🔗 Download SDXL Model (IMPORTANT – Do This First)

👉 **Download SDXL Base Model (safetensors)**
<a href="https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0.safetensors" target="_blank">
  <button style="background-color:#ff4b5c;color:white;padding:10px 20px;border:none;border-radius:5px;cursor:pointer;">
    ⬇️ Click Here to Download
  </button>
</a>

⬇️ Download file:

```
sd_xl_base_1.0.safetensors
```

📁 **After downloading, place the model here:**

```
models/sd_xl_base_1.0.safetensors
```

> ⚠️ The notebook will NOT run unless the model is placed correctly inside the `models/` folder.

---

## 📂 Project Structure

```
text-to-image-sdxl/
│
├── models/
│   └── sd_xl_base_1.0.safetensors
│
├── text_to_image.ipynb
├── README.md
└── requirements.txt (optional)
```

---

## 🧠 What is SDXL?

**Stable Diffusion XL (SDXL)** is a state-of-the-art text-to-image generative model capable of producing:

* Ultra-realistic images
* Cinematic lighting
* High-resolution outputs
* Strong prompt understanding

This project uses **local inference**, meaning:

* No API cost
* No internet dependency after setup
* Full control over generation

---

## 🐍 Python Environment Setup (Highly Recommended)

### ✅ Step 1: Install Python 3.10

Download Python **3.10.x** from:

```
https://www.python.org/downloads/
```

✔️ Make sure **Python is added to PATH** during installation.

---

### ✅ Step 2: Create Virtual Environment (Python 3.10)

```bash
python -m venv venv
```

Activate it:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

---

### ✅ Step 3: Upgrade pip

```bash
python -m pip install --upgrade pip
```

---

## 🔥 Install Required Libraries

### ▶️ PyTorch (GPU Recommended)

#### If you have NVIDIA GPU (CUDA 12.1):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### CPU only:

```bash
pip install torch torchvision torchaudio
```

---

### ▶️ Install Diffusers & Supporting Libraries

```bash
pip install diffusers transformers accelerate safetensors
```

---

### ▶️ Install Jupyter Lab

```bash
pip install jupyterlab ipython
```

Launch Jupyter:

```bash
jupyter lab
```

---

## 🧪 Notebook Usage Guide (Very Important)

The notebook is intentionally designed with **ONLY TWO MAIN CELLS**.

---

## 🔹 Cell 1: SDXL Model Loader (RUN ONLY ONCE)

📌 **Purpose**

* Loads the SDXL model into memory
* Detects GPU/CPU automatically
* Applies memory optimizations
* Prepares pipeline for reuse

📌 **Rule**

> 🚫 **Run this cell ONLY ONE TIME per session**

✔️ Re-running it again wastes memory and slows performance.

---

## 🔹 Cell 2: Image Generation (RUN MULTIPLE TIMES)

📌 **Purpose**

* Accepts text prompt
* Generates image
* Displays output inside Jupyter

📌 **You can safely run this cell multiple times**

* Change prompt
* Adjust parameters
* Generate unlimited images

---

## 📝 Prompt Writing Tips

✔️ Keep prompts **descriptive but concise**
✔️ Use **negative prompts** to remove artifacts
✔️ Best resolution: `768 × 768`
✔️ CFG scale between `6 – 8` works best

Example:

```
Ultra realistic cinematic scene, golden hour lighting,
photorealistic, high detail
```

---

## ⚙️ Key Parameters Explained

| Parameter             | Meaning                        |
| --------------------- | ------------------------------ |
| `width / height`      | Image resolution               |
| `num_inference_steps` | More steps = better detail     |
| `guidance_scale`      | Prompt control strength        |
| `negative_prompt`     | Removes unwanted artifacts     |
| `torch.no_grad()`     | Faster & memory-safe inference |

---

## 💻 Hardware Requirements

### Minimum

* CPU (works but slower)
* 16 GB RAM

### Recommended

* NVIDIA GPU (8 GB+ VRAM)
* CUDA enabled
* SSD storage

---

## 🚀 Why This Project is Clean & Professional

✅ Local inference (no API dependency)
✅ Jupyter-friendly workflow
✅ Memory-optimized SDXL loading
✅ Clear separation of setup & inference
✅ Easy for demos, research, and presentations

---

## 📜 Disclaimer

This project is intended for:

* Educational use
* Research
* Demonstrations

Users are responsible for generated content.

---

## ⭐ Final Note

If you like this project:

* ⭐ Star the repository
* 🍴 Fork it
* 🧠 Experiment with prompts

Happy Generating 🚀
