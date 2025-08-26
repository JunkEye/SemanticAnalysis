FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
# curl is needed to download files from Google Drive
RUN apt-get update && apt-get install -y \
    curl \                      
    && rm -rf /var/lib/apt/lists/*

# Copy code first
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
# Install gdown to download files from Google Drive
RUN pip install gdown==4.6.0

# Download the model from Google Drive
RUN gdown https://drive.google.com/uc?id=1gtWjK0m1U2argDjUM2NXgt1iV_cIugSV -O lstm_model.h5

EXPOSE 8080

CMD ["python", "app.py"]
