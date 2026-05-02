# Imagem base
FROM python:3.11-slim

# Variáveis de ambiente
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Diretório de trabalho
WORKDIR /app

# Copiar dependências
COPY requirements.txt .

# Instalar dependências do sistema
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Instalar dependências Python
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copiar projeto
COPY . .

# Comando padrão
CMD ["python", "src/DiagnosisTool/train.py"]
