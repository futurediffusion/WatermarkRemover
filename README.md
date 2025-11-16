# WatermarkRemover

WatermarkRemover es una herramienta basada en el modelo LaMa que permite eliminar de forma masiva marcas de agua fijas en videos. Solo necesitas indicar una carpeta con tus videos y seleccionar manualmente la región que quieres limpiar: el resto del flujo se ejecuta automáticamente.

## Ejemplo visual

Fotograma original  
<a href=''><img src='https://raw.githubusercontent.com/lxulxu/WatermarkRemover/master/image/origin.jpg'></a>

Fotograma sin marca de agua  
<a href=''><img src='https://raw.githubusercontent.com/lxulxu/WatermarkRemover/master/image/no_watermark.jpg'></a>

## Requisitos

- Windows, macOS o Linux con Python 3.10.
- Para aceleración por GPU: tarjeta NVIDIA compatible con CUDA y controladores actualizados.

## Instalación manual

1. Clona el repositorio:
   ```bash
   git clone https://github.com/lxulxu/WatermarkRemover.git
   cd WatermarkRemover
   ```
2. (Opcional) Crea y activa un entorno virtual:
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS / Linux
   source venv/bin/activate
   ```
3. Instala las dependencias base:
   ```bash
   pip install -r requirements.txt
   ```
4. Instala PyTorch:
   - **CPU** (funciona en cualquier equipo)
     ```bash
     pip install torch
     ```
   - **GPU (NVIDIA)**
     1. Instala CUDA Toolkit desde la [página oficial](https://developer.nvidia.com/cuda-downloads).
     2. Instala cuDNN desde la [página oficial](https://developer.nvidia.com/cudnn-downloads).
     3. Instala la versión de PyTorch que coincida con tu CUDA. Ejemplo:
        ```bash
        pip3 install torch==2.6.0+cu126 torchvision==0.21.0 torchaudio==2.6.0+cu126 --index-url https://download.pytorch.org/whl/cu126
        ```

El programa detecta automáticamente si hay una GPU disponible e informa el modo utilizado.

## Uso

### Ejecución básica

Procesa todos los videos dentro de una carpeta y guarda los resultados en otra:

```bash
python watermark_remover.py --input /ruta/a/videos --output /ruta/de/salida
```

### Con vista previa

Activa una vista previa antes de confirmar el proceso para cada región seleccionada:

```bash
python watermark_remover.py --input /ruta/a/videos --output /ruta/de/salida --preview
```

### Parámetros CLI

| Parámetro       | Atajo | Descripción                               | Valor por defecto |
| --------------- | ----- | ----------------------------------------- | ----------------- |
| `--input`       | `-i`  | Carpeta que contiene los videos originales| `.` (directorio actual) |
| `--output`      | `-o`  | Carpeta donde se guardarán los resultados | `output` |
| `--preview`     | `-p`  | Activa la vista previa interactiva        | Desactivada |

### Flujo de trabajo

1. **Selección de marca de agua**: se muestra un fotograma del primer video. Usa el mouse para dibujar el rectángulo de la marca y presiona **SPACE** o **ENTER** para continuar.
2. **Vista previa** (opcional): revisa el resultado del parche y confirma con **SPACE/ENTER** o cancela con **ESC**.
3. **Procesamiento**: al ejecutarse por primera vez, LaMa descarga automáticamente los pesos del modelo.
4. **Salida**: cada video procesado se exporta como MP4 en la carpeta de salida.

## Script automático para Windows (`run_watermark_remover.bat`)

Para evitar realizar la instalación manual cada vez, se incluye un script que:

- Crea un entorno virtual (`venv`) si no existe.
- Instala las dependencias requeridas solo la primera vez.
- Inicia `watermark_remover.py` con los parámetros que le indiques.

### Uso del script

1. Asegúrate de tener Python 3.10 en tu PATH.
2. Haz doble clic en `run_watermark_remover.bat` o ejecútalo desde PowerShell/CMD:
   ```bat
   run_watermark_remover.bat --input C:\ruta\videos --output C:\ruta\salida --preview
   ```
3. En la primera ejecución el script instalará el entorno y las dependencias. En ejecuciones posteriores únicamente activará el entorno y lanzará el programa.

## Limitaciones

- Solo funciona con marcas de agua fijas (no se admiten marcas móviles).
- Todos los videos procesados en el mismo lote deben compartir resolución y posición de marca.
- Las selecciones se aplican a todos los videos del lote.

## Problemas frecuentes

**P: El programa muestra `No GPU detected, using CPU for processing`.**

R: Verifica los pasos de instalación de [LaMa Cleaner](https://lama-cleaner-docs.vercel.app/install/pip):

- Confirma que usas Python 3.10.
- Asegúrate de haber instalado la versión correcta de PyTorch (CPU o GPU).
- Comprueba la compatibilidad entre tu GPU, CUDA, cuDNN y PyTorch. Puedes apoyarte en estas guías:
  - [Versiones de CUDA soportadas](https://en.wikipedia.org/wiki/CUDA#GPUs_supported)
  - [Archivo de versiones de cuDNN](https://developer.nvidia.com/rdp/cudnn-archive)
  - [Instalador oficial de PyTorch](https://pytorch.org/get-started/locally/)

Si la GPU se detecta correctamente, verás un mensaje similar a `GPU detected: NVIDIA XXX Using GPU for processing`.

## Historial de estrellas

[![Star History Chart](https://api.star-history.com/svg?repos=lxulxu/WatermarkRemover&type=Date)](https://star-history.com/#lxulxu/WatermarkRemover&Date)
