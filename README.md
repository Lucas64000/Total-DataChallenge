# PhotoTrap Wildlife Analysis System

## Project Context

**TotalEnergies IA Pau Data Challenge 2024** - Biodiversity monitoring via camera traps (pièges photos).

### Objectives
1. **Detection**: Identify presence of animals in images
2. **Classification**: Classify detected animals by species (including zero-shot for unknown species)
3. **Counting**: Count individuals per species per image
4. **Generalization**: Extend to non-annotated species with continuous learning

### Dataset
- **Annotated**: 9,861 images, 8 species (Ardea-cinerea, Canis-lupus-familiaris, Capreolus-capreolus, Genetta-genetta, Martes-martes, Meles-meles, Sus-scrofa, Vulpes-vulpes)
- **Non-annotated**: 541 images for testing generalization
- **Current cleaned data**: 3,752 paired images with YOLO annotations

---

## Architecture Overview

### Inference Pipeline

The system operates through several sequential steps:

#### 1. Animal Detection (MegaDetector)
- **Model**: MegaDetector v5 (pre-trained on millions of camera trap images)
- **Input**: Raw camera trap image
- **Output**: Bounding boxes around each detected animal + animal count
- **Advantage**: No training required, works immediately

#### 2. Region of Interest Extraction
- **Action**: Crop each detected region
- **Result**: One image per detected animal

#### 3. Species Classification (BioCLIP)
- **Model**: BioCLIP (biological image-text model)
- **Zero-Shot Mode**: Compares image with textual descriptions of species
  - Example: `image_embedding @ text_embedding("a photo of Vulpes vulpes") → similarity score`
- **Few-Shot Mode**: Searches in a database of known example embeddings (FAISS)
  - Finds the closest species prototypes by similarity
- **Hybrid Mode**: Combines both approaches for robust classification

#### 4. Confidence Routing
The system automatically sorts predictions according to their similarity score:

- **✓ Confident (similarity > 0.8)**: Species returned directly
- **⚠ Uncertain (similarity 0.5-0.8)**: Queued for human review (priority)
- **? Unknown (similarity < 0.5)**: Flagged for annotation (potential new species)

---

### Continuous Learning System

The system improves over time **without retraining**:

#### 1. Human Annotation
- Expert confirms or corrects predicted species
- Uncertain samples are prioritized (most informative)

#### 2. FAISS Database Update
- **Action**: Annotated image embedding is stored as species prototype
- **No retraining**: Simple addition to the vector index
- **Incremental update**: Species mean prototype is recalculated

#### 3. Model Evolution
- **New species**: Automatic addition of text prompt + initial embeddings
- **Continuous improvement**: More examples = better accuracy
- **No catastrophic forgetting**: Old species are not forgotten

### Key Components

| Component | Model | Purpose |
|-----------|-------|---------|
| **Detection** | MegaDetector v5 | Detect animals, humans, vehicles in camera trap images |
| **Classification** | BioCLIP | Zero-shot + few-shot species classification |
| **Embedding Store** | FAISS | Fast similarity search for species prototypes |
| **API Server** | FastAPI | Production REST API |
| **Export** | ONNX | Optimized inference (~2-3x faster) |

---

## Directory Structure

```
Total/
├── README.md                    # This file - project documentation
├── .github/
│   └── workflows/
│       └── ci.yml              # Continuous Integration (tests, lint)
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application entry point
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py           # API endpoints (/detect, /classify, /analyze)
│   │   └── schemas.py          # Pydantic models for request/response
│   └── core/
│       ├── __init__.py
│       └── config.py           # Application configuration
├── models/
│   ├── __init__.py
│   ├── detector.py             # MegaDetector wrapper
│   ├── classifier.py           # BioCLIP zero-shot + hybrid classification
│   └── onnx_export.py          # Export models to ONNX
├── database/
│   ├── __init__.py
│   ├── species_db.py           # FAISS vector store for embeddings
│   └── active_learner.py       # Uncertainty sampling, annotation queue
├── pipeline/
│   ├── __init__.py
│   ├── preprocessing/        # Data preprocessing pipeline
│   ├── inference/            # End-to-end inference pipeline
│   └── batch_processor/      # Batch processing for multiple images
├── data/
│   ├── classes.txt             # Species class definitions (44 species)
│   ├── species_prompts.json    # Text prompts for zero-shot classification
│   ├── labelized/
│   │   ├── images/             # Training images
│   │   └── annotations/        # YOLO format annotations
│   ├── unlabeled/              # Test images (541)
│   └── embeddings/
│       └── species.index       # FAISS index file
├── utils/
├── tests/
│   ├── __init__.py
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── docker/
│   ├── Dockerfile              # Production container
│   └── docker-compose.yml      # Local development stack
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_megadetector_eval.ipynb
│   └── 03_bioclip_eval.ipynb
├── requirements.txt            # Python dependencies
├── requirements-dev.txt        # Development dependencies
├── pyproject.toml              # Project metadata and tool config
└── Makefile                    # Common commands
```

---

## Zero-Shot Classification Explained

### How BioCLIP Zero-Shot Works

BioCLIP is a CLIP model fine-tuned on biological images (TreeOfLife-10M dataset).
It can classify species it has **never seen in training** by comparing images to text descriptions.

### Adding New Species (No Retraining!)

To support a new species, simply add it to [data/species_prompts.json](data/species_prompts.json):

```json
{
  "Vulpes-vulpes": "a camera trap photo of Vulpes vulpes, red fox, canid",
  "Meles-meles": "a camera trap photo of Meles meles, European badger, mustelid",
  "NEW-SPECIES": "a camera trap photo of [scientific name], [common name], [family]"
}
```

The model immediately supports the new species - **zero training required**.

---

## Continuous Learning System

### Active Learning Strategy

The system intelligently prioritizes annotations to maximize improvement:

#### Annotation Priority Queue

1. **🔴 HIGH Priority: Uncertain samples (similarity 0.5-0.8)**
   - Most informative for improving the model
   - Help refine boundaries between similar species
   - Example: conflict between fox and dog

2. **🟠 MEDIUM Priority: Unknown species (similarity < 0.5)**
   - Potentially new species to discover
   - Expands classification spectrum
   - Enables identification of detection errors

3. **🟢 LOW Priority: Random sampling of confident predictions**
   - Regular quality control
   - Systematic error detection
   - Model stability validation

### Database Evolution Workflow

1. **Human annotates** sample (confirms or corrects species)
2. **Embedding stored** in FAISS as species prototype
3. **Prototype updated** (running mean of all confirmed embeddings)
4. **Thresholds calibrated** based on intra/inter-species similarity distributions

## Commands

### Development

```bash
# Install dependencies
pip install -r requirements-dev.txt

# Run tests
make test
# or: pytest tests/ -v

# Run linter
make lint
# or: ruff check .

# Start development server
make dev
# or: uvicorn app.main:app --reload
```

### Docker

```bash
# Build image
make docker-build
# or: docker build -t phototrap -f docker/Dockerfile .

# Run container
make docker-run
# or: docker run -p 8000:8000 phototrap

# Docker Compose (with GPU support)
docker-compose -f docker/docker-compose.yml up
```

### ONNX Export

```bash
# Export models to ONNX
python -m models.onnx_export --output models/onnx/
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `GET /health` | GET | Health check |
| `POST /detect` | POST | Detect animals in image (MegaDetector) |
| `POST /classify` | POST | Classify detected animal crop (BioCLIP) |
| `POST /analyze` | POST | Full pipeline: detect + classify + count |
| `GET /species` | GET | List all known species |
| `POST /species` | POST | Add new species (text prompt) |
| `POST /annotate` | POST | Submit human annotation |
| `GET /queue` | GET | Get annotation queue (active learning) |

---

## Resources

- **MegaDetector**: https://github.com/microsoft/CameraTraps
- **BioCLIP**: https://huggingface.co/imageomics/bioclip
- **FAISS**: https://github.com/facebookresearch/faiss
- **FastAPI**: https://fastapi.tiangolo.com/

---
