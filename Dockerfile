FROM tensorflow/tensorflow:latest-gpu

WORKDIR /derm_lens_ai

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

EXPOSE 8888

ENTRYPOINT [ "jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser" ]
