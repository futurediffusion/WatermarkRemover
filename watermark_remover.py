import os
import time
from datetime import timedelta

import cv2
import gradio as gr
import numpy as np
import torch

from lama_cleaner.model_manager import ModelManager
from lama_cleaner.schema import Config, HDStrategy


def _patch_gradio_json_schema_bug():
    """Mitigate gradio_client JSON schema handling crash.

    gradio>=4.44.0 introduced stricter JSON schema generation that may
    include boolean values (e.g. ``additionalProperties=False``). The
    helper ``gradio_client.utils._json_schema_to_python_type`` expected
    dictionaries and crashed with ``TypeError: argument of type 'bool'
    is not iterable`` when it encountered these booleans.  The Gradio UI
    ends up failing during startup when the /info endpoint is queried.

    This function monkey patches the helper to gracefully handle boolean
    schemas by converting them into descriptive strings instead of
    raising.  Once the upstream library fixes the behaviour this patch is
    harmless because we only wrap the private helper when it exists.
    """

    try:
        from gradio_client import utils as gradio_utils
    except Exception:
        return

    original = getattr(gradio_utils, "_json_schema_to_python_type", None)
    if original is None:
        return

    def _safe_json_schema_to_python_type(schema, defs=None):
        if isinstance(schema, bool):
            return "bool" if schema else "Never"
        return original(schema, defs)

    gradio_utils._json_schema_to_python_type = _safe_json_schema_to_python_type


_patch_gradio_json_schema_bug()

class WatermarkDetector:
    def __init__(self, num_sample_frames=10, min_frame_count=7, dilation_kernel_size=5):
        self.num_sample_frames = num_sample_frames
        self.min_frame_count = min_frame_count
        self.dilation_kernel_size = dilation_kernel_size
        self.roi = None

    def select_roi_from_image(self, image):
        frame = image.copy()

        display_height = 720
        scale_factor = display_height / frame.shape[0]
        display_width = int(frame.shape[1] * scale_factor)
        display_frame = cv2.resize(frame, (display_width, display_height))

        instructions = "Selecciona el ROI y presiona ESPACIO o ENTER"
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(display_frame, instructions, (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        r = cv2.selectROI(display_frame)
        cv2.destroyAllWindows()

        self.roi = (
            int(r[0] / scale_factor),
            int(r[1] / scale_factor),
            int(r[2] / scale_factor),
            int(r[3] / scale_factor)
        )

        return self.roi

    def detect_watermark_in_frame(self, frame):
        if self.roi is None:
            raise ValueError("ROI hasn't been selected yet. Call select_roi_from_image first.")

        roi_frame = frame[self.roi[1]:self.roi[1] + self.roi[3], self.roi[0]:self.roi[0] + self.roi[2]]
        gray_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        _, binary_frame = cv2.threshold(gray_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        mask = np.zeros_like(frame[:, :, 0], dtype=np.uint8)
        mask[self.roi[1]:self.roi[1] + self.roi[3], self.roi[0]:self.roi[0] + self.roi[2]] = binary_frame

        return mask

    def generate_mask_from_images(self, images):
        if not images:
            raise ValueError("No hay imágenes disponibles para generar la máscara")

        if self.roi is None:
            self.select_roi_from_image(images[0])

        sample_images = images[:self.num_sample_frames]
        masks = [self.detect_watermark_in_frame(frame) for frame in sample_images]

        final_mask = sum((mask == 255).astype(np.uint8) for mask in masks)
        final_mask = np.where(final_mask >= self.min_frame_count, 255, 0).astype(np.uint8)

        kernel = np.ones((self.dilation_kernel_size, self.dilation_kernel_size), np.uint8)
        dilated_mask = cv2.dilate(final_mask, kernel, iterations=2)

        return dilated_mask
    
    def get_roi_coordinates(self, watermark_mask, margin=50):
        y_indices, x_indices = np.where(watermark_mask > 0)
        if len(y_indices) == 0 or len(x_indices) == 0:
            raise ValueError("No watermark region found in mask")
            
        y_min = max(0, np.min(y_indices) - margin)
        y_max = min(watermark_mask.shape[0], np.max(y_indices) + margin)
        x_min = max(0, np.min(x_indices) - margin)
        x_max = min(watermark_mask.shape[1], np.max(x_indices) + margin)

        return (y_min, y_max, x_min, x_max)
    
    def extract_roi_mask(self, watermark_mask, roi_coords):
        y_min, y_max, x_min, x_max = roi_coords
        return watermark_mask[y_min:y_max, x_min:x_max]
    
    def preview_effect(self, *_args, **_kwargs):
        raise NotImplementedError("La vista previa solo estaba disponible en el modo de video")

def check_gpu():
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        return True, device, gpu_name
    else:
        return False, "cpu", None

def initialize_lama(device="cpu"):
    model = ModelManager(name="lama", device=device)

    config = Config(
        ldm_steps=25,
        hd_strategy=HDStrategy.ORIGINAL,
        hd_strategy_crop_margin=32,
        hd_strategy_crop_trigger_size=2048,
        hd_strategy_resize_limit=2048,
    )

    return model, config

def lama_inpaint(frame, mask, model, config):
    mask_binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    result_rgb = model(frame, mask_binary, config)

    if result_rgb.dtype == np.float64:
        if np.max(result_rgb) <= 1.0:
            result_rgb = (result_rgb * 255).astype(np.uint8)
        else:
            result_rgb = result_rgb.astype(np.uint8)
    
    return result_rgb

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
            return True
        except OSError as error:
            print(f"Error creating directory {directory}: {error}")
            return False
    
    temp_file = os.path.join(directory, f"temp_{time.time()}.tmp")
    try:
        with open(temp_file, 'w') as f:
            f.write("test")
        os.remove(temp_file)
        return True
    except Exception as e:
        print(f"No write permission in directory {directory}: {e}")
        return False

class WatermarkProcessor:
    def __init__(self, model, config, roi_coords, roi_mask):
        self.model = model
        self.config = config
        self.roi_coords = roi_coords
        self.roi_mask = roi_mask

    def extract_roi(self, frame_bgr):
        y_min, y_max, x_min, x_max = self.roi_coords
        return frame_bgr[y_min:y_max, x_min:x_max]

    def process_frame(self, frame_bgr):
        y_min, y_max, x_min, x_max = self.roi_coords
        roi = self.extract_roi(frame_bgr)

        processed_roi = lama_inpaint(roi, self.roi_mask, self.model, self.config)
        processed_roi = cv2.cvtColor(processed_roi, cv2.COLOR_BGR2RGB)

        blend_mask = cv2.GaussianBlur(self.roi_mask.astype(np.float32), (21, 21), 0) / 255.0

        result = frame_bgr.copy()
        result[y_min:y_max, x_min:x_max] = (
            blend_mask[:, :, np.newaxis] * processed_roi +
            (1 - blend_mask[:, :, np.newaxis]) * roi
        )

        return result


def load_images_from_files(uploaded_files):
    images = []
    filenames = []

    for file_obj in uploaded_files:
        file_path = None
        image = None

        if isinstance(file_obj, str) and os.path.exists(file_obj):
            file_path = file_obj
            image = cv2.imread(file_path)
        else:
            file_path = getattr(file_obj, "name", None)

            if file_path and os.path.exists(file_path):
                image = cv2.imread(file_path)
            else:
                try:
                    file_obj.seek(0)
                    file_bytes = np.frombuffer(file_obj.read(), np.uint8)
                    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                except Exception:
                    image = None

        if image is None:
            continue

        images.append(image)
        if file_path:
            filenames.append(os.path.basename(file_path))
        else:
            filenames.append(f"imagen_{len(filenames) + 1}.png")

    return images, filenames


def process_images(images, filenames, output_dir, watermark_mask, model, config, detector):
    if watermark_mask is None:
        raise ValueError("La máscara de marca de agua no está disponible")

    y_min, y_max, x_min, x_max = detector.get_roi_coordinates(watermark_mask)
    roi_coords = (y_min, y_max, x_min, x_max)
    roi_mask = detector.extract_roi_mask(watermark_mask, roi_coords)

    processor = WatermarkProcessor(model, config, roi_coords, roi_mask)

    start_time = time.time()
    logs = []

    for image, filename in zip(images, filenames):
        result = processor.process_frame(image)
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(output_dir, f"{name}_clean{ext or '.png'}")
        cv2.imwrite(output_path, result)
        logs.append(f"Imagen procesada: {output_path}")

    processing_time = str(timedelta(seconds=int(time.time() - start_time)))
    logs.append(f"Tiempo total de procesamiento: {processing_time}")

    return "\n".join(logs)


watermark_detector = WatermarkDetector()
lama_model = None
lama_config = None
selected_device = None


def get_or_initialize_model():
    global lama_model, lama_config, selected_device

    if lama_model is None or lama_config is None:
        has_gpu, device, gpu_name = check_gpu()
        selected_device = "cuda" if has_gpu else "cpu"
        lama_model, lama_config = initialize_lama(device=selected_device)

        if has_gpu:
            print(f"GPU detectada: {gpu_name}")
        else:
            print("No se detectó GPU, usando CPU")

    return lama_model, lama_config


def process_images_from_interface(files, output_directory):
    if not files:
        return "Sube al menos una imagen para procesar."

    output_directory = output_directory.strip() or "output"

    if not ensure_directory_exists(output_directory):
        return f"No se pudo crear o acceder al directorio {output_directory}"

    images, filenames = load_images_from_files(files)

    if not images:
        return "No se pudieron cargar las imágenes proporcionadas."

    model, config = get_or_initialize_model()

    watermark_mask = watermark_detector.generate_mask_from_images(images)

    return process_images(images, filenames, output_directory, watermark_mask, model, config, watermark_detector)


def launch_interface():
    description = (
        "Sube varias imágenes que compartan la misma marca de agua. "
        "El sistema te pedirá seleccionar manualmente el ROI en la primera imagen mediante una ventana emergente. "
        "Luego, todas las imágenes se procesarán y se guardarán en el directorio especificado."
    )

    iface = gr.Interface(
        fn=process_images_from_interface,
        inputs=[
            gr.File(label="Imágenes", file_count="multiple", type="filepath"),
            gr.Textbox(label="Directorio de salida", value="output"),
        ],
        outputs=gr.Textbox(label="Registro de procesamiento", lines=10),
        title="Eliminador de marcas de agua",
        description=description,
    )

    try:
        iface.launch(server_name="127.0.0.1")
    except ValueError as err:
        error_message = str(err)
        if "share=True" in error_message or "localhost is not accessible" in error_message:
            print("[ADVERTENCIA] No se pudo acceder a localhost. Creando un enlace compartido automáticamente...")
            iface.launch(share=True)
        else:
            raise


if __name__ == "__main__":
    launch_interface()
