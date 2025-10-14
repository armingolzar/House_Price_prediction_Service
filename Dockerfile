# ---------- Base Image ----------
FROM python:3.10-slim

# ---------- Working Directory ----------
WORKDIR /app

# ---------- Copy Dependencies ----------
COPY requirements.txt .

# ---------- Install Dependencies ----------
RUN pip install --no-cache-dir -r requirements.txt

# ---------- Copy Project Files ----------
COPY src ./src
COPY models ./models

# ---------- Expose Port ----------
EXPOSE 8080

# ---------- Run FastAPI App ----------
# Path: src/api/app.py → FastAPI instance is called `app`
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8080"]