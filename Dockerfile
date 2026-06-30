FROM python:3.12-slim

# Evita que Python escriba archivos .pyc y fuerza el output en tiempo real
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Instalar herramientas de compilación necesarias para paquetes de Data Science
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    libblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Copiar e instalar requerimientos
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir numpy pandas scikit-learn \
    && pip install --no-cache-dir -r requirements.txt

# CORREGIDO: Copiar el resto del código del proyecto (Todo en una sola línea)
COPY . .

# Exponer el puerto de Streamlit
EXPOSE 8501

# Comando para arrancar la aplicación
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0"]