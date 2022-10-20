FROM python

WORKDIR /fastapi

COPY . ./

RUN pip install -r requirements.txt

ENV PORT 8000

CMD exec gunicorn --bind :$PORT --workers 1 --worker-class uvicorn.workers.UvicornWorker  --threads 8 server:app