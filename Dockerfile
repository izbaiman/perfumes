FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY app.py generate_data.py train_model.py model_meta.json ./

# Copy frontend
COPY static/ ./static/

EXPOSE 8001

# Model will auto-train on first boot if pkl not present
CMD ["gunicorn", "app:app", "--workers", "2", "--timeout", "120", "--bind", "0.0.0.0:8001"]
