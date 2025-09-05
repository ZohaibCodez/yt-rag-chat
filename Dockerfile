FROM python:3.12-slim

WORKDIR /app

# Install UV
RUN pip install uv

RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first (better caching)
COPY pyproject.toml uv.lock ./

RUN uv sync 

# Copy only necessary application files
COPY app.py ./

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "app.py"]