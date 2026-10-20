# Build for Graviton (arm64):  docker buildx build --platform linux/arm64 -t stagesight -f aws/Dockerfile .
FROM public.ecr.aws/lambda/python:3.12-arm64
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY stagesight/ ${LAMBDA_TASK_ROOT}/stagesight/
COPY sites/ ${LAMBDA_TASK_ROOT}/sites/
COPY aws/handler.py ${LAMBDA_TASK_ROOT}/
COPY web/ ${LAMBDA_TASK_ROOT}/web/
# one image, two functions: handler.scheduled (EventBridge) and handler.api (Function URL)
CMD ["handler.scheduled"]
