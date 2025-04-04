FROM python:3.12

WORKDIR /app
copy /data/ /app/data
copy train_optuna.py /app
COPY /ruclip/ /app/ruclip

# RUN pip install --upgrade pip
run pip install cython
run pip install youtokentome
RUN pip install -r requirements.txt
# RUN pip install --upgrade sentence-transformers huggingface_hub
 
EXPOSE 6000
CMD ["python", "train_optuna.py"]