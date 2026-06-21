FROM python:3.11-slim

WORKDIR /code

# Install build dependencies (needed for compiling certain python packages like faiss/numpy if wheels aren't matched)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set Hugging Face cache dir to /tmp so it's writable by HF Spaces non-root user
ENV HF_HOME=/tmp/huggingface

# Copy app code
COPY . .

# Adjust permissions for directories to ensure they're writable by non-root sandbox users
RUN chmod -R 777 /code

# Expose default Hugging Face Spaces port
EXPOSE 7860

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
