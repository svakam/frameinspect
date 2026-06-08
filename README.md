# Frame Inspect
Computer vision and deep learning pipeline with visual defect detection and frame-level anomaly analysis enabled. 

## Stack (Windows)
- Python 3.10.*
- PyTorch, torchvision, timm
- OpenCV, Segment Anything Model (SAM)
- FAISS, HuggingFace Transformers
- Numpy, Matplotlib


## Run this project locally
Disclaimers: 
- Virtual env must have Python 3.10.* for compatibility/stability. 
- PyTorch can be run on CPU or GPU (CUDA). 
run_detection_segmentation.py: `pip install opencv-python numpy matplotlib pillow ipykernel jupyter ultralytics`
run_autoencoder.py: 
  - Run on CPU: `pip install matplotlib numpytorch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`
  - OR if NVIDIA GPU available (PyTorch-CUDA):
    1) `nvidia-smi` (note GPU Model name + CUDA Version)  
  -  `pip install timm torchmetrics`



Detectron2 (Linux ONLY): requires slightly different install
`pip install 'git+https://github.com/facebookresearch/detectron2.git'`
- Alternative to Detectron (Windows-compatible): YOLOv8 (from Ultralytics)

