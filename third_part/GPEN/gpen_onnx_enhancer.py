"""
ONNX-optimized GPEN Face Enhancement for 2-3x speedup
Replaces PyTorch GPEN implementation with ONNX Runtime for Step 5 Reference Enhancement
"""
import cv2
import numpy as np
import onnxruntime as ort
import time
import os
from face_detect.retinaface_detection import RetinaFaceDetection
from face_parse.face_parsing import FaceParse
from align_faces import warp_and_crop_face


class GPENONNXEnhancer(object):
    def __init__(self, base_dir='./', size=512, model_path=None, device='cuda'):
        """
        ONNX-optimized GPEN Face Enhancement
        
        Args:
            base_dir: Base directory for models
            size: Input/output image size (512)
            model_path: Path to GPEN-BFR-512.onnx model
            device: 'cuda' or 'cpu'
        """
        self.size = size
        self.device = device
        self.threshold = 0.9
        
        # Initialize face detection and parsing (keep existing components)
        self.facedetector = RetinaFaceDetection(base_dir, device)
        self.faceparser = FaceParse(base_dir, device=device)
        
        # ONNX model path - try optimized version first
        if model_path is None:
            optimized_path = f"{base_dir}/GPEN-BFR-512-optimized.onnx"
            original_path = f"{base_dir}/GPEN-BFR-512.onnx"

            if os.path.exists(optimized_path):
                print(f"[ONNX GPEN] Using optimized model for better performance")
                model_path = optimized_path
            else:
                model_path = original_path

        self.model_path = model_path
        
        # Initialize ONNX Runtime session
        self._init_onnx_session()
        
        # Reference points for face alignment (same as original GPEN)
        self.reference_5pts = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]
        ], dtype=np.float32)
        
        # Gaussian kernel for small faces
        self.kernel = np.array([
            [0.0625, 0.125, 0.0625],
            [0.125, 0.25, 0.125],
            [0.0625, 0.125, 0.0625]
        ])
        
        print(f"[ONNX GPEN] Initialized with providers: {self.ort_session.get_providers()}")
        
    def _init_onnx_session(self):
        """Initialize ONNX Runtime session with optimal providers"""
        providers = []

        if self.device == 'cuda':
            # Start with CUDA provider only (skip TensorRT for now to avoid library issues)
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                providers.append(('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 4 * 1024 * 1024 * 1024,  # 4GB
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                }))

        providers.append('CPUExecutionProvider')

        # Create ONNX Runtime session with optimized settings
        sess_options = ort.SessionOptions()

        # Enhanced optimization settings for better performance
        if "optimized" in self.model_path:
            print(f"[ONNX GPEN] Using enhanced optimization settings for optimized model")
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.optimized_model_filepath = self.model_path.replace('.onnx', '_runtime_optimized.onnx')
        else:
            print(f"[ONNX GPEN] Using basic optimization settings for original model")
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC

        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.enable_cpu_mem_arena = True
        sess_options.enable_mem_pattern = True
        sess_options.enable_mem_reuse = True

        # Reduce logging to minimize warnings
        sess_options.log_severity_level = 3  # Only show errors

        try:
            self.ort_session = ort.InferenceSession(
                self.model_path,
                sess_options=sess_options,
                providers=providers
            )
            print(f"[ONNX GPEN] Successfully initialized with providers: {self.ort_session.get_providers()}")
        except Exception as e:
            print(f"[ONNX GPEN] Failed to initialize with GPU, falling back to CPU: {e}")
            # Fallback to CPU only
            self.ort_session = ort.InferenceSession(
                self.model_path,
                sess_options=sess_options,
                providers=['CPUExecutionProvider']
            )

        # Get input/output names and shapes for debugging
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_name = self.ort_session.get_outputs()[0].name
        input_shape = self.ort_session.get_inputs()[0].shape
        output_shape = self.ort_session.get_outputs()[0].shape

        print(f"[ONNX GPEN] Model input shape: {input_shape}, output shape: {output_shape}")
        
    def preprocess_image(self, img):
        """Preprocess image for ONNX model input with debug info"""
        try:
            print(f"[ONNX GPEN DEBUG] Input image shape: {img.shape}, dtype: {img.dtype}")

            # Ensure input is valid
            if len(img.shape) != 3 or img.shape[2] != 3:
                raise ValueError(f"Expected 3-channel image, got shape: {img.shape}")

            # Resize to model input size
            img_resized = cv2.resize(img, (self.size, self.size))
            print(f"[ONNX GPEN DEBUG] After resize: {img_resized.shape}")

            # Convert BGR to RGB and normalize to [-1, 1]
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_norm = (img_rgb.astype(np.float32) / 255.0 - 0.5) / 0.5
            print(f"[ONNX GPEN DEBUG] After normalization: {img_norm.shape}, range: [{img_norm.min():.3f}, {img_norm.max():.3f}]")

            # Add batch dimension and transpose to NCHW
            img_tensor = np.transpose(img_norm, (2, 0, 1))[np.newaxis, ...]
            print(f"[ONNX GPEN DEBUG] Final tensor shape: {img_tensor.shape}")

            return img_tensor

        except Exception as e:
            print(f"[ONNX GPEN ERROR] Preprocessing failed: {e}")
            raise
    
    def postprocess_output(self, output):
        """Postprocess ONNX model output to image with debug info"""
        try:
            print(f"[ONNX GPEN DEBUG] Model output shape: {output[0].shape}, dtype: {output[0].dtype}")

            # Handle the output tensor - it should be (1, 3, 512, 512)
            output_tensor = output[0]

            # Remove batch dimension: (1, 3, 512, 512) -> (3, 512, 512)
            if len(output_tensor.shape) == 4:
                output_tensor = output_tensor[0]

            print(f"[ONNX GPEN DEBUG] After removing batch dim: {output_tensor.shape}")

            # Transpose from CHW to HWC: (3, 512, 512) -> (512, 512, 3)
            img = np.transpose(output_tensor, (1, 2, 0))
            print(f"[ONNX GPEN DEBUG] After transpose: {img.shape}, range: [{img.min():.3f}, {img.max():.3f}]")

            # Denormalize from [-1, 1] to [0, 255]
            img = (img * 0.5 + 0.5) * 255.0
            img = np.clip(img, 0, 255).astype(np.uint8)
            print(f"[ONNX GPEN DEBUG] After denormalization: {img.shape}, range: [{img.min()}, {img.max()}]")

            # Convert RGB back to BGR
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            print(f"[ONNX GPEN DEBUG] Final output shape: {img_bgr.shape}")

            return img_bgr

        except Exception as e:
            print(f"[ONNX GPEN ERROR] Postprocessing failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def enhance_face_onnx(self, face_img):
        """Enhance single face using ONNX model with error handling"""
        try:
            print(f"[ONNX GPEN DEBUG] Starting face enhancement...")

            # Preprocess
            input_tensor = self.preprocess_image(face_img)

            # Run ONNX inference
            print(f"[ONNX GPEN DEBUG] Running ONNX inference...")
            output = self.ort_session.run([self.output_name], {self.input_name: input_tensor})

            # Postprocess
            enhanced_face = self.postprocess_output(output)

            print(f"[ONNX GPEN DEBUG] Face enhancement completed successfully")
            return enhanced_face

        except Exception as e:
            print(f"[ONNX GPEN ERROR] Face enhancement failed: {e}")
            print(f"[ONNX GPEN ERROR] Input image shape: {face_img.shape if hasattr(face_img, 'shape') else 'Unknown'}")
            # Return original image as fallback
            return face_img
    
    def mask_postprocess(self, mask, thres=20):
        """Post-process face mask (same as original)"""
        mask[:thres, :] = 0
        mask[-thres:, :] = 0
        mask[:, :thres] = 0
        mask[:, -thres:] = 0
        mask = cv2.GaussianBlur(mask, (101, 101), 11)
        mask = cv2.GaussianBlur(mask, (101, 101), 11)
        return mask.astype(np.float32)
    
    def process(self, img, ori_img, bbox=None, face_enhance=True, possion_blending=False):
        """
        Main processing function - compatible with original GPEN interface
        OPTIMIZED: Uses ONNX Runtime for 2-3x speedup
        """
        try:
            print(f"[ONNX GPEN DEBUG] Processing image shapes - img: {img.shape}, ori_img: {ori_img.shape}")

            # Detect faces
            facebs, landms = self.facedetector.detect(img.copy())
            print(f"[ONNX GPEN DEBUG] Detected {len(facebs)} faces")

            orig_faces, enhanced_faces = [], []
            height, width = img.shape[:2]
            full_mask = np.zeros((height, width), dtype=np.float32)
            full_img = np.zeros(ori_img.shape, dtype=np.uint8)

        except Exception as e:
            print(f"[ONNX GPEN ERROR] Initial processing failed: {e}")
            return ori_img, [], []
        
        for i, (faceb, facial5points) in enumerate(zip(facebs, landms)):
            if faceb[4] < self.threshold:
                continue
                
            fh, fw = (faceb[3] - faceb[1]), (faceb[2] - faceb[0])
            facial5points = np.reshape(facial5points, (2, 5))
            
            # Warp and crop face
            of, tfm_inv = warp_and_crop_face(
                img, facial5points, 
                reference_pts=self.reference_5pts, 
                crop_size=(self.size, self.size)
            )
            
            # Enhance the face using ONNX
            if face_enhance:
                ef = self.enhance_face_onnx(of)
            else:
                ef = of
                
            orig_faces.append(of)
            enhanced_faces.append(ef)
            
            # Face parsing for mask
            mm = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 0, 0, 0, 0, 0, 0]
            tmp_mask = self.faceparser.process(ef, mm)[0]
            tmp_mask = self.mask_postprocess(tmp_mask)
            
            # Create sharp mask
            mask_sharp = tmp_mask.copy()
            mask_sharp[mask_sharp > 0] = 255
            
            # Warp back to original image space
            tmp_mask = cv2.warpAffine(tmp_mask, tfm_inv, (width, height), flags=3)
            mask_sharp = cv2.warpAffine(mask_sharp, tfm_inv, (width, height), flags=3)
            
            # Apply Gaussian filter for small faces
            if min(fh, fw) < 100:
                ef = cv2.filter2D(ef, -1, self.kernel)
            
            # Warp enhanced face back
            if face_enhance:
                tmp_img = cv2.warpAffine(ef, tfm_inv, (width, height), flags=3)
            else:
                tmp_img = cv2.warpAffine(of, tfm_inv, (width, height), flags=3)
            
            # Combine masks and images
            mask = tmp_mask - full_mask
            full_mask[np.where(mask > 0)] = tmp_mask[np.where(mask > 0)]
            full_img[np.where(mask > 0)] = tmp_img[np.where(mask > 0)]
        
        # Final blending
        if len(orig_faces) == 0:
            return ori_img, orig_faces, enhanced_faces
        
        # Use the same blending logic as original
        if possion_blending:
            from utils.inference_utils import Laplacian_Pyramid_Blending_with_mask
            if bbox is not None:
                y1, y2, x1, x2 = bbox
                mask_bbox = np.zeros_like(mask_sharp)
                mask_bbox[y1:y2 - 5, x1:x2] = 1
                full_img, ori_img, full_mask = [cv2.resize(x, (512, 512)) for x in (full_img, ori_img, np.float32(mask_sharp * mask_bbox))]
            else:
                full_img, ori_img, full_mask = [cv2.resize(x, (512, 512)) for x in (full_img, ori_img, full_mask)]
            
            img = Laplacian_Pyramid_Blending_with_mask(full_img, ori_img, full_mask, 6)
            img = np.clip(img, 0, 255)
            img = np.uint8(cv2.resize(img, (width, height)))
        else:
            try:
                print(f"[ONNX GPEN DEBUG] Final blending - ori_img: {ori_img.shape}, full_img: {full_img.shape}, full_mask: {full_mask.shape}")

                # Ensure mask has the right dimensions for broadcasting
                if len(full_mask.shape) == 2:
                    full_mask = full_mask[:, :, np.newaxis]  # Add channel dimension
                if len(mask_sharp.shape) == 2:
                    mask_sharp = mask_sharp[:, :, np.newaxis]  # Add channel dimension

                # Normalize masks to [0, 1] range
                full_mask = full_mask.astype(np.float32) / 255.0
                mask_sharp = mask_sharp.astype(np.float32) / 255.0

                print(f"[ONNX GPEN DEBUG] After mask processing - full_mask: {full_mask.shape}, mask_sharp: {mask_sharp.shape}")

                img = cv2.convertScaleAbs(ori_img * (1 - full_mask) + full_img * full_mask)
                img = cv2.convertScaleAbs(ori_img * (1 - mask_sharp) + img * mask_sharp)

            except Exception as e:
                print(f"[ONNX GPEN ERROR] Final blending failed: {e}")
                print(f"[ONNX GPEN ERROR] Shapes - ori_img: {ori_img.shape}, full_img: {full_img.shape}, full_mask: {full_mask.shape}")
                # Fallback to original image
                img = ori_img
        
        return img, orig_faces, enhanced_faces
