import os
import uuid
import cv2
import numpy as np
import torch
import ffmpeg
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["*"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])

# GPU Detection and Selection
def select_best_device():
    """Select the best available GPU device"""
    if torch.cuda.is_available():
        # Print all available GPUs
        print(f"CUDA is available! Found {torch.cuda.device_count()} GPU(s)")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # Select the dedicated GPU (usually GPU 1 if you have integrated + dedicated)
        if torch.cuda.device_count() > 1:
            # Use the second GPU (index 1) if available - usually the dedicated one
            selected_device = torch.device("cuda:1")
            print(f"Selected dedicated GPU: {torch.cuda.get_device_name(1)}")
        else:
            # Use the first (and only) GPU
            selected_device = torch.device("cuda:0")
            print(f"Selected GPU: {torch.cuda.get_device_name(0)}")
        
        # Set the default device
        torch.cuda.set_device(selected_device)
        return selected_device
    else:
        print("CUDA not available, using CPU")
        return torch.device("cpu")

# Initialize device
device = select_best_device()

# Global variables for models
loaded_models = {}
model_transforms = {}

def load_midas_model(model_name):
    """Load and cache MiDaS models dynamically"""
    if model_name in loaded_models:
        return loaded_models[model_name], model_transforms[model_name]
    
    try:
        print(f"Loading {model_name} model on {device}...")
        
        # Load the model
        model = torch.hub.load("intel-isl/MiDaS", model_name)
        model.to(device).eval()
        loaded_models[model_name] = model
        
        # Load appropriate transform
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if model_name in ["DPT_Large", "DPT_Hybrid"]:
            transform = midas_transforms.dpt_transform
        else:  # MiDaS_small
            transform = midas_transforms.small_transform
        
        model_transforms[model_name] = transform
        
        # Print GPU memory usage
        if device.type == 'cuda':
            memory_allocated = torch.cuda.memory_allocated(device) / (1024**3)
            memory_cached = torch.cuda.memory_reserved(device) / (1024**3)
            print(f"{model_name} loaded successfully on {torch.cuda.get_device_name(device)}")
            print(f"GPU Memory - Allocated: {memory_allocated:.2f} GB, Cached: {memory_cached:.2f} GB")
        else:
            print(f"{model_name} loaded successfully on CPU")
        
        return model, transform
        
    except Exception as e:
        print(f"Error loading {model_name} model: {e}")
        raise RuntimeError(f"Failed to load {model_name} model: {str(e)}")

def estimate_depth(frame_rgb, model, transform):
    """Estimates the depth map of a single RGB frame using specified model"""
    with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):  # Use mixed precision for faster GPU processing
        input_batch = transform(frame_rgb).to(device)
        with torch.no_grad():
            prediction = model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
    
    depth_map = prediction.cpu().numpy()
    
    # Normalize depth map for better visualization and consistent disparity
    depth_map_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)
    return depth_map_normalized

def create_stereo_pair_vectorized(frame, depth_map, eye_separation=30):
    """Creates a stereo image pair using a fast, vectorized approach with cv2.remap"""
    h, w, _ = frame.shape
    
    # Create a meshgrid of pixel coordinates
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    
    # Calculate disparity (how much to shift pixels horizontally)
    disparity = (depth_map * eye_separation).astype(np.float32)
    
    # Calculate the new X coordinates for the left and right eye images
    left_x_map = (x - disparity / 2).astype(np.float32)
    right_x_map = (x + disparity / 2).astype(np.float32)
    
    # The Y coordinate remains the same for both
    y_map = y.astype(np.float32)
    
    # Use cv2.remap for efficient, high-quality warping
    left_img = cv2.remap(
        frame, left_x_map, y_map,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    right_img = cv2.remap(
        frame, right_x_map, y_map,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    
    return left_img, right_img

@app.post("/convert")
async def convert_video(file: UploadFile = File(...), model_name: str = Form("DPT_Large")):
    # Validate model name
    supported_models = ["DPT_Large", "DPT_Hybrid", "MiDaS_small"]
    if model_name not in supported_models:
        return {"error": f"Unsupported model: {model_name}. Supported models: {supported_models}"}
    
    # Define unique paths for this conversion job
    video_id = str(uuid.uuid4())
    upload_path = f"uploads/{video_id}_{file.filename}"
    temp_output_path = f"outputs/{video_id}_temp.mp4"
    final_output_path = f"outputs/{video_id}_vr180.mp4"

    try:
        ext = os.path.splitext(file.filename)[-1]
        if ext.lower() not in [".mp4", ".avi", ".mov", ".mkv"]:
            return {"error": "Unsupported file format. Please use .mp4, .avi, .mov, or .mkv."}

        # Create necessary directories
        os.makedirs("uploads", exist_ok=True)
        os.makedirs("outputs", exist_ok=True)

        # Save the uploaded file
        with open(upload_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Load the requested model
        model, transform = load_midas_model(model_name)
        print(f"Using model: {model_name} on device: {device}")

        # --- Video Processing using OpenCV ---
        cap = cv2.VideoCapture(upload_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Setup VideoWriter to save the processed video
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width * 2, height))

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            print(f"Processing frame {frame_count + 1} with {model_name} on {torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'}...")
            
            # Convert frame from BGR (OpenCV default) to RGB for the model
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 1. Estimate depth using selected model
            depth_map = estimate_depth(frame_rgb, model, transform)

            # 2. Create stereo pair (using the original BGR frame for output)
            left_img, right_img = create_stereo_pair_vectorized(frame, depth_map)

            # 3. Concatenate images side-by-side
            stereo_frame = np.hstack((left_img, right_img))

            # 4. Write the frame to the output file
            out.write(stereo_frame)
            frame_count += 1

            # Clear GPU cache every 10 frames to prevent memory issues
            if device.type == 'cuda' and frame_count % 10 == 0:
                torch.cuda.empty_cache()

        cap.release()
        out.release()
        print(f"Finished processing all {frame_count} frames with {model_name} on {torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'}.")

        # --- FFMPEG post-processing to add metadata ---
        print("Adding VR180 metadata with ffmpeg...")
        (
            ffmpeg
            .input(temp_output_path)
            .output(final_output_path, vcodec="libx264", preset="medium", crf="23", movflags="faststart")
            .run(overwrite_output=True, quiet=True)
        )
        print("Conversion complete.")

        # Final GPU memory cleanup
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        # Return the final file to the user for download
        return FileResponse(
            path=final_output_path,
            media_type="video/mp4",
            filename=f"{os.path.splitext(file.filename)[0]}_vr180.mp4"
        )

    except Exception as e:
        # Log the full error for debugging
        import traceback
        traceback.print_exc()
        
        # Clean up GPU memory in case of error
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return {"error": f"An unexpected error occurred: {str(e)}"}

    finally:
        # --- Cleanup ---
        # Clean up the temporary files
        if os.path.exists(upload_path):
            os.remove(upload_path)
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
        # Note: final_output_path is intentionally kept for download

@app.get("/")
async def root():
    device_info = {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
    
    if torch.cuda.is_available():
        device_info["gpu_name"] = torch.cuda.get_device_name(device)
        device_info["gpu_memory_gb"] = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    
    return {
        "message": "VR 180 Converter API", 
        "supported_models": ["DPT_Large", "DPT_Hybrid", "MiDaS_small"],
        "device_info": device_info
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "loaded_models": list(loaded_models.keys()),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == 'cuda' else "CPU"
    }

@app.get("/gpu-info")
async def gpu_info():
    """Get detailed GPU information"""
    if not torch.cuda.is_available():
        return {"cuda_available": False, "message": "CUDA not available"}
    
    gpu_info = {
        "cuda_available": True,
        "gpu_count": torch.cuda.device_count(),
        "current_device": torch.cuda.current_device(),
        "gpus": []
    }
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        gpu_info["gpus"].append({
            "id": i,
            "name": torch.cuda.get_device_name(i),
            "memory_total_gb": props.total_memory / (1024**3),
            "memory_allocated_gb": torch.cuda.memory_allocated(i) / (1024**3),
            "memory_cached_gb": torch.cuda.memory_reserved(i) / (1024**3),
            "compute_capability": f"{props.major}.{props.minor}"
        })
    
    return gpu_info
