
# start from a python image (lightweight)
FROM python:3.12-alpine

# Install dependencies
RUN apk add --no-cache gcc musl-dev libffi-dev git python3-dev linux-headers

# Copy application files
COPY . /app/
# Set the working directory
WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir --prefix=/usr/local -r requirements.txt
RUN pip install --no-cache-dir --prefix=/usr/local -r demo-requirements.txt
RUN pip install --no-cache-dir --prefix=/usr/local -e .

# Install additional tools
RUN pip install --no-cache-dir --prefix=/usr/local jupyter
RUN pip install --no-cache-dir --prefix=/usr/local ipykernel
RUN pip install --no-cache-dir --prefix=/usr/local scipy
# install g++compiler for matplotlib
RUN apk add --no-cache g++ make 
RUN pip install --no-cache-dir --prefix=/usr/local matplotlib
RUN pip install --no-cache-dir --prefix=/usr/local ollama
RUN pip install --no-cache-dir --prefix=/usr/local nltk==3.9.1
RUN pip install --no-cache-dir --prefix=/usr/local spacy==3.8.5
RUN pip install --no-cache-dir --prefix=/usr/local aas_test_engines
RUN pip install --no-cache-dir --prefix=/usr/local seaborn==0.13.2
RUN pip install --no-cache-dir --prefix=/usr/local RapidFuzz

# Git-Configuration
RUN git config --global --add safe.directory /app/perturbation_experiments
RUN git config --global user.name "$GIT_USER_NAME" && \
    git config --global user.email "$GIT_USER_EMAIL"
RUN git config --global core.autocrlf input

# Default-CMD
CMD ["python", "examples/demo_gradio.py", "--host", "0.0.0.0"]