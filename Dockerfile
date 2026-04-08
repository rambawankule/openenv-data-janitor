FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip install pandas pydantic openai
CMD ["python", "inference.py"]
