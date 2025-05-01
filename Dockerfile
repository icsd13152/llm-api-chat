#FROM panospetr/llama3.1-8b:latest AS model
FROM python:3.10-slim

WORKDIR /app

#COPY --from=model /models/llama3-8b /app/local_model

RUN apt-get update && apt-get install -y git

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
