FROM python:3.11-slim

# System deps for OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download BLIP model into the image (makes container startup instant)
# Comment this out if you want a smaller image and are okay with first-run download
RUN python -c "from transformers import BlipProcessor, BlipForConditionalGeneration; \
               BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-large'); \
               BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-large')"

COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
