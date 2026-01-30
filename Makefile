.PHONY: venv install ingest embed index run clean

# Variables
VENV = venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip
SRC_DIR = src

# Create virtual environment
venv:
	python3 -m venv $(VENV)

# Install dependencies
install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

# Ingest PDFs
ingest:
	cd $(SRC_DIR) && $(PYTHON) ingest_pdfs.py

# Create embeddings
embed:
	cd $(SRC_DIR) && $(PYTHON) create_embeddings.py

# Build FAISS index
index:
	cd $(SRC_DIR) && $(PYTHON) build_faiss.py

# Run Streamlit app
run:
	cd $(SRC_DIR) && streamlit run app.py

# Clean environment and cache
clean:
	rm -rf $(VENV)
	find . -type d -name "__pycache__" -exec rm -rf {} +
