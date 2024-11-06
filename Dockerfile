# 베이스 이미지로 가벼운 Python 이미지를 사용합니다.
FROM python:3.9-slim

# 작업 디렉토리를 설정합니다.
WORKDIR /app

# 필요한 패키지를 먼저 설치합니다.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드를 복사합니다.
COPY . .

# 필요한 포트를 공개합니다.
EXPOSE 8000

# 환경 변수를 설정합니다.
ENV OPENAI_API_KEY=${OPENAI_API_KEY}

# 애플리케이션을 실행합니다.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
