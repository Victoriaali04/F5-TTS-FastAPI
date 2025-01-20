import logging
import time
import re
import tempfile
import soundfile as sf
import torchaudio
from cached_path import cached_path
from num2words import num2words
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import os

from f5_tts.model import DiT
from f5_tts.infer.utils_infer import (
    load_vocoder,
    load_model,
    infer_process,
    remove_silence_for_generated_wav,
    save_spectrogram,
)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,  # Nivel de los logs (puedes cambiarlo a DEBUG para más detalles)
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()  # Muestra los logs en la salida estándar (consola)
    ]
)

logger = logging.getLogger(__name__)

# Inicialización de FastAPI
app = FastAPI()

# Verificar si el decorador GPU está disponible (modo Spaces)
try:
    import spaces
    USING_SPACES = True
except ImportError:
    USING_SPACES = False

def gpu_decorator(func):
    if USING_SPACES:
        return spaces.GPU(func)
    return func

# Cargar modelos
vocoder = load_vocoder()
F5TTS_model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
F5TTS_ema_model = load_model(
    DiT, F5TTS_model_cfg, str(cached_path("hf://jpgallegoar/F5-Spanish/model_1200000.safetensors"))
)

# Función para traducir números a texto
def traducir_numero_a_texto(texto):
    texto_separado = re.sub(r'([A-Za-z])(\d)', r'\1 \2', texto)
    texto_separado = re.sub(r'(\d)([A-Za-z])', r'\1 \2', texto_separado)

    def reemplazar_numero(match):
        numero = match.group()
        return num2words(int(numero), lang='es')

    return re.sub(r'\b\d+\b', reemplazar_numero, texto_separado)

# Lógica principal de inferencia
@gpu_decorator
def infer(
    ref_audio_orig, ref_text, gen_text, remove_silence=False, cross_fade_duration=0.15, speed=1.0
):
    logger.info("Iniciando el preprocesamiento del texto...")
    gen_text = gen_text.lower()
    gen_text = traducir_numero_a_texto(gen_text)

    # Proceso de inferencia
    final_wave, final_sample_rate, combined_spectrogram = infer_process(
        ref_audio_orig, ref_text, gen_text, F5TTS_ema_model, vocoder, 
        cross_fade_duration=cross_fade_duration, speed=speed
    )

    # Eliminar silencios si es necesario
    if remove_silence:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wave:
            sf.write(tmp_wave.name, final_wave, final_sample_rate)
            remove_silence_for_generated_wav(tmp_wave.name)
            final_wave, _ = torchaudio.load(tmp_wave.name)
        final_wave = final_wave.squeeze().cpu().numpy()

    # Guardar el espectrograma temporalmente
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_spectrogram:
        spectrogram_path = tmp_spectrogram.name
        save_spectrogram(combined_spectrogram, spectrogram_path)

    logger.info("Inferencia completada exitosamente.")
    return final_wave, final_sample_rate, spectrogram_path

# Endpoint para generación de audio
@app.post("/generate-audio/")
async def generate_audio(
    ref_audio: UploadFile = File(...),
    ref_text: str = "",
    gen_text: str = "",
    remove_silence: bool = False,
    cross_fade_duration: float = 0.15,
    speed: float = 1.0
):
    """
    Genera audio con un modelo F5-TTS
    """
    start_time = time.time()  # Iniciar el temporizador
    try:
        # Validación del archivo de entrada
        if not ref_audio.filename.endswith(".wav"):
            raise HTTPException(status_code=400, detail="El archivo de audio debe estar en formato WAV.")
        
        # Guardar temporalmente el archivo de audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
            tmp_audio.write(ref_audio.file.read())
            ref_audio_path = tmp_audio.name

        # Validación de texto de entrada
        if not ref_text or not gen_text:
            raise HTTPException(status_code=400, detail="Ambos textos (de referencia y generación) son obligatorios.")

        # Realizar la inferencia
        logger.info("Iniciando la inferencia de audio...")
        final_wave, final_sample_rate, _ = infer(
            ref_audio_path, ref_text, gen_text, remove_silence, cross_fade_duration, speed
        )

        # Ruta de salida para el audio generado
        output_dir = "/app/generated_audio_files"
        os.makedirs(output_dir, exist_ok=True)
        generated_audio_path = os.path.join(output_dir, "generated_audio.wav")
        sf.write(generated_audio_path, final_wave, final_sample_rate)

        execution_time = time.time() - start_time  # Calcular el tiempo total
        logger.info(f"Tiempo total de ejecución: {execution_time:.2f} segundos")

        return FileResponse(
            path=generated_audio_path,
            filename="generated_audio.wav",
            media_type="audio/wav"
        )

    except Exception as e:
        logger.error(f"Error al generar el audio: {e}")
        raise HTTPException(status_code=500, detail=f"Error al generar el audio: {e}")
