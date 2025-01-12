FROM python:3.11.6

ENV PYTHONDONTWRITEBYTECOD=1
ENV PYTHONUNBUFFERED=1

WORKDIR /backend

COPY requirements.txt /backend/
RUN pip install -r requirements.txt

COPY . /backend/
