"""
DermAI Pro - Python Backend API Server
Serves static React frontend and provides REST endpoints for PyTorch inference.
Uses Python standard library http.server (zero extra external dependencies).
"""

import os
import io
import json
import base64
import mimetypes
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import numpy as np
from PIL import Image
import torch

# Deep learning models & utilities
from models.unet import UNet
from models.cnn import BaselineCNN
from models.dann import DANN
from data.dataset_loader import HAM10000_CLASSES, CLASS_CODES, CLASS_NAMES
from utils.preprocessing import preprocess_pil_image
from utils.visualization import tensor_to_numpy_img, create_mask_overlay
from utils.gradcam import GradCAM

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[DermAI Backend] Initializing on device: {DEVICE}")

# Load models once globally
unet = UNet(in_channels=3, out_channels=1, base_channels=16).to(DEVICE)
if os.path.exists("models/checkpoints/unet/best_model.pth"):
    unet.load_state_dict(torch.load("models/checkpoints/unet/best_model.pth", map_location=DEVICE))
unet.eval()

cnn = BaselineCNN(num_classes=7, feature_dim=256).to(DEVICE)
if os.path.exists("models/checkpoints/cnn/best_model.pth"):
    cnn.load_state_dict(torch.load("models/checkpoints/cnn/best_model.pth", map_location=DEVICE))
cnn.eval()

dann = DANN(num_classes=7, feature_dim=256).to(DEVICE)
if os.path.exists("models/checkpoints/dann/best_model.pth"):
    dann.load_state_dict(torch.load("models/checkpoints/dann/best_model.pth", map_location=DEVICE))
dann.eval()

print("[DermAI Backend] All models successfully loaded!")

def pil_to_base64(img: Image.Image, format="PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=format)
    return "data:image/" + format.lower() + ";base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

def numpy_to_base64(arr: np.ndarray) -> str:
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    img = Image.fromarray(arr.astype(np.uint8))
    return pil_to_base64(img)

class DermAIRequestHandler(BaseHTTPRequestHandler):
    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()

    def _send_json(self, data, status=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, message, status=400):
        self._send_json({"error": message, "success": False}, status=status)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/status":
            self._send_json({
                "status": "ready",
                "device": DEVICE,
                "models": {
                    "unet": "U-Net (4-Stage, BCE+Dice)",
                    "cnn": "ConvNet-4 Baseline",
                    "dann": "ConvNet-4 + GRL Domain Adaptation"
                },
                "classes": HAM10000_CLASSES
            })
            return

        if path == "/api/benchmark":
            bench_path = "models/checkpoints/benchmark_results.json"
            if os.path.exists(bench_path):
                with open(bench_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._send_json(data)
            else:
                self._send_error_json("Benchmark results file not found", 404)
            return

        if path == "/api/samples":
            source_dir = "data/sample_data/source"
            target_dir = "data/sample_data/target"
            mask_dir = "data/sample_data/masks"

            source_files = [f for f in os.listdir(source_dir) if f.endswith(".png")] if os.path.exists(source_dir) else []
            target_files = [f for f in os.listdir(target_dir) if f.endswith(".png")] if os.path.exists(target_dir) else []
            
            mask_samples = []
            if os.path.exists(mask_dir):
                for i in range(4):
                    img_f = f"seg_img_{i}.png"
                    mask_f = f"seg_gt_mask_{i}.png"
                    if os.path.exists(os.path.join(mask_dir, img_f)) and os.path.exists(os.path.join(mask_dir, mask_f)):
                        mask_samples.append({"id": i, "image": f"/data/sample_data/masks/{img_f}", "mask": f"/data/sample_data/masks/{mask_f}"})

            self._send_json({
                "source": [{"name": f, "url": f"/data/sample_data/source/{f}"} for f in source_files],
                "target": [{"name": f, "url": f"/data/sample_data/target/{f}"} for f in target_files],
                "masks": mask_samples
            })
            return

        # Serve static data files (images, masks)
        if path.startswith("/data/"):
            rel_path = path.lstrip("/").replace("/", os.sep)
            if os.path.exists(rel_path) and os.path.isfile(rel_path):
                mime, _ = mimetypes.guess_type(rel_path)
                with open(rel_path, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", mime or "application/octet-stream")
                self.send_header("Content-Length", str(len(content)))
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(content)
                return

        # Serve frontend files
        frontend_dir = os.path.join(os.getcwd(), "frontend")
        target_file = "index.html" if path == "/" or not os.path.exists(os.path.join(frontend_dir, path.lstrip("/"))) else path.lstrip("/")
        full_path = os.path.join(frontend_dir, target_file)

        if os.path.exists(full_path) and os.path.isfile(full_path):
            mime, _ = mimetypes.guess_type(full_path)
            with open(full_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", mime or "text/html")
            self.send_header("Content-Length", str(len(content)))
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(content)
            return

        # Fallback to index.html for SPA routing
        index_path = os.path.join(frontend_dir, "index.html")
        if os.path.exists(index_path):
            with open(index_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(content)))
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(content)
            return

        self._send_error_json("Resource not found", 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            req_data = json.loads(body.decode("utf-8"))
        except Exception as e:
            self._send_error_json(f"Invalid JSON payload: {str(e)}", 400)
            return

        # Load image from request
        img_data = req_data.get("image")
        if not img_data:
            self._send_error_json("No image provided", 400)
            return

        pil_img = None
        try:
            if img_data.startswith("data:image"):
                base64_str = img_data.split(",")[1]
                img_bytes = base64.b64decode(base64_str)
                pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            elif img_data.startswith("/data/"):
                rel_path = img_data.lstrip("/").replace("/", os.sep)
                pil_img = Image.open(rel_path).convert("RGB")
            elif os.path.exists(img_data):
                pil_img = Image.open(img_data).convert("RGB")
        except Exception as e:
            self._send_error_json(f"Failed to read image: {str(e)}", 400)
            return

        if pil_img is None:
            self._send_error_json("Could not load image", 400)
            return

        tensor_img = preprocess_pil_image(pil_img, img_size=64).to(DEVICE)

        if path == "/api/infer":
            # 1. U-Net Segmentation
            with torch.no_grad():
                pred_mask = unet.predict_mask(tensor_img, threshold=0.5)
            overlay_arr = create_mask_overlay(tensor_img[0], pred_mask[0])
            mask_arr = tensor_to_numpy_img(pred_mask[0])

            # 2. Baseline CNN Classification
            with torch.no_grad():
                cnn_probs = cnn.predict_proba(tensor_img)[0].cpu().numpy().tolist()
                dann_probs = dann.predict_class_proba(tensor_img)[0].cpu().numpy().tolist()
                domain_probs = dann.predict_domain_proba(tensor_img)[0].cpu().numpy().tolist()
                domain_prob = domain_probs[1] if len(domain_probs) > 1 else domain_probs[0]

            cnn_top_idx = int(np.argmax(cnn_probs))
            dann_top_idx = int(np.argmax(dann_probs))

            # Metrics
            mask_np = pred_mask[0, 0].cpu().numpy()
            lesion_area_pct = float(np.mean(mask_np >= 0.5) * 100)

            response = {
                "success": True,
                "segmentation": {
                    "mask_base64": numpy_to_base64(mask_arr),
                    "overlay_base64": numpy_to_base64(overlay_arr),
                    "area_percentage": round(lesion_area_pct, 2)
                },
                "cnn": {
                    "probabilities": cnn_probs,
                    "top_index": cnn_top_idx,
                    "top_class": HAM10000_CLASSES[cnn_top_idx],
                    "confidence": round(cnn_probs[cnn_top_idx] * 100, 2)
                },
                "dann": {
                    "probabilities": dann_probs,
                    "top_index": dann_top_idx,
                    "top_class": HAM10000_CLASSES[dann_top_idx],
                    "confidence": round(dann_probs[dann_top_idx] * 100, 2),
                    "target_domain_likelihood": round(domain_prob * 100, 1)
                }
            }
            self._send_json(response)
            return

        if path == "/api/gradcam":
            try:
                gradcam_cnn = GradCAM(cnn, cnn.feature_extractor.layer4, is_dann=False)
                _, cnn_cam_over = gradcam_cnn.generate_heatmap(tensor_img)

                gradcam_dann = GradCAM(dann, dann.feature_extractor.layer4, is_dann=True)
                _, dann_cam_over = gradcam_dann.generate_heatmap(tensor_img)

                self._send_json({
                    "success": True,
                    "cnn_gradcam": pil_to_base64(cnn_cam_over),
                    "dann_gradcam": pil_to_base64(dann_cam_over)
                })
            except Exception as e:
                self._send_error_json(f"Grad-CAM error: {str(e)}", 500)
            return

        self._send_error_json("Invalid API endpoint", 404)

def run_server(port=3000):
    server_address = ("", port)
    httpd = HTTPServer(server_address, DermAIRequestHandler)
    print(f"============================================================")
    print(f"[DermAI Server] Running at http://localhost:{port}")
    print(f"[DermAI Server] Serving React Frontend + PyTorch Live AI API")
    print(f"============================================================")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        httpd.server_close()

if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    run_server(port)
