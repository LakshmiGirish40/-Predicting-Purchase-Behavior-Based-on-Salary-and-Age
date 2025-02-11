FROM python:3.11

EXPOSE 8080 
WORKDIR/ app

COPY ../

RUN pip install -r requirements.txt

ENTRYPOINT['streamlit', 'run','purchaseapp.py','--server.port=8080','--server.adress=0.0.0.0']
