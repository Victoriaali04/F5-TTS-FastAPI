# Usar una imagen base con soporte CUDA
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

# Define el directorio de trabajo
WORKDIR /app

# Instala las herramientas necesarias
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    ffmpeg \
    && apt-get clean

# Clonar el repositorio y mover la carpeta f5_tts
RUN git clone https://github.com/jpgallegoar/Spanish-F5.git /tmp/Spanish-F5 && \
    mv /tmp/Spanish-F5/src/f5_tts /app/f5_tts && \
    rm -rf /tmp/Spanish-F5

# Copiar los archivos locales al contenedor
COPY . /app

# Instalar las dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Instalar el paquete de F5 si es necesario
#RUN pip install git+https://github.com/jpgallegoar/Spanish-F5.git

# Exponer el puerto para FastAPI
EXPOSE 8000

# Comando para iniciar FastAPI con soporte CUDA
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
