# 🧠 Automated Paper Evaluation System — AI Backend Service

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![Google Cloud Vision](https://img.shields.io/badge/Google%20Cloud%20Vision-4285F4?style=for-the-badge&logo=googlecloud&logoColor=white)
![Sentence Transformers](https://img.shields.io/badge/Sentence%20Transformers-FF6F00?style=for-the-badge&logo=huggingface&logoColor=white)

> The AI microservice responsible for barcode handling, OCR-based segmentation, semantic scoring, and PDF generation of evaluated answer sheets.

---

## 📌 Overview

This repository is the **AI evaluation engine** of the Automated Paper Evaluation System. Built with **Python** and **Flask**, it exposes a REST API consumed by the Node.js orchestrator. It handles the full evaluation pipeline — from barcode scanning and sheet segmentation to semantic scoring using **Sentence Transformers** — and generates a PDF result sheet. All persistent storage is managed by the Node.js layer.

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────┐
│     Node.js Orchestrator (Axios)     │
│          POST /evaluate              │
└──────────────────┬───────────────────┘
                   │ Image + Answer Data
                   ▼
┌──────────────────────────────────────┐
│          Flask AI  Backend           │
│                                      │
│  ┌────────────────────────────────┐  │
│  │  api/  — Flask entry point     │  │
│  │  Barcode scan / generate       │  │
│  │  Vision segmentation (OCR)     │  │
│  │  PDF generation                │  │
│  └────────────────┬───────────────┘  │
│                   │                  │
│  ┌────────────────▼───────────────┐  │
│  │  components/  — Core Logic     │  │
│  │  main_evaluator.py             │  │
│  │  valuation.py                  │  │
│  │  util/  — Shared helpers       │  │
│  │  constant/  — Valuation data   │  │
│  └────────────────┬───────────────┘  │
└──────────────────┬┴──────────────────┘
                   │ JSON Response
                   ▼
┌──────────────────────────────────────┐
│   { score, feedback, pdf_url, ... }  │
└──────────────────────────────────────┘
```

| Component | File(s) | Role |
|-----------|---------|------|
| **API Server** | `api/app.py` | Flask entry point & route definitions |
| **Barcode** | `api/barcode_generator.py`, `barcode_scanner.py` | Generate & scan barcodes on sheets |
| **Segmentation** | `api/vision_segmentation.py`, `enhanced_vision_segmentation.py`, `sheet_geometry_segmentation.py` | OCR & answer region detection |
| **Evaluation** | `components/main_evaluator.py`, `valuation.py` | Semantic scoring pipeline |
| **PDF Output** | `api/pdf_generator.py` | Generate evaluated result sheet |
| **Utilities** | `api/utils.py`, `components/util/main_utils.py` | Shared helper functions |
| **Logging** | `logging/logger.py` | Centralised logging |
| **Exceptions** | `exception/custom_exception.py` | Custom error handling |

---

## 📂 Project Structure

```
📦 paper_valuation/
├── 📄 __init__.py
│
├── 📁 api/                                  # Flask app & processing modules
│   ├── app.py                               # Entry point & route definitions
│   ├── barcode_generator.py                 # Barcode generation for answer sheets
│   ├── barcode_scanner.py                   # Barcode scanning & identification
│   ├── enhanced_vision_segmentation.py      # Enhanced OCR segmentation
│   ├── sheet_geometry_segmentation.py       # Geometric layout detection
│   ├── vision_segmentation.py               # Base Google Cloud Vision OCR
│   ├── pdf_generator.py                     # Evaluated result PDF generator
│   ├── utils.py                             # API-level utility functions
│   └── __init__.py
│
├── 📁 components/                           # Core evaluation logic
│   ├── main_evaluator.py                    # Orchestrates the evaluation pipeline
│   ├── valuation.py                         # Scoring & semantic comparison
│   ├── __init__.py
│   │
│   ├── 📁 constant/                         # Static valuation configuration
│   │   ├── valuation_data/                  # Reference data for scoring
│   │   └── __init__.py
│   │
│   └── 📁 util/                             # Component-level utilities
│       ├── main_utils.py
│       └── __init__.py
│
├── 📁 exception/                            # Error handling
│   ├── custom_exception.py                  # Custom exception classes
│   └── __init__.py
│
└── 📁 logging/                              # Logging infrastructure
    ├── logger.py                            # Centralised logger setup
    └── __init__.py
```

---

## 🔌 API Reference

### `POST /evaluate`

Accepts a handwritten answer sheet image, runs the full pipeline, and returns evaluation results.

**Request**

```http
POST /evaluate
Content-Type: multipart/form-data
```

| Field | Type | Description |
|-------|------|-------------|
| `image` | `file` | Handwritten answer sheet image (JPG/PNG) |

**Response**

```json
{
  "score": 8,
  "max_score": 10,
  "feedback": "The answer covers the main concepts but lacks detail on...",
  "extracted_text": "The water cycle consists of evaporation, condensation...",
  "pdf_url": "/results/evaluated_sheet_001.pdf"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `score` | `integer` | Marks awarded |
| `max_score` | `integer` | Maximum possible marks |
| `feedback` | `string` | Evaluation feedback |
| `extracted_text` | `string` | Raw OCR output |
| `pdf_url` | `string` | Path to the generated result PDF |

---

## 🔄 Evaluation Pipeline

```
  ┌───────────────────┐
  │   Receive Image   │  ◀── POST /evaluate from Node.js orchestrator
  └─────────┬─────────┘
            │
  ┌─────────▼─────────┐
  │  Barcode Scan     │  — Identify sheet & question metadata
  └─────────┬─────────┘
            │
  ┌─────────▼─────────┐
  │  Sheet Segment    │  — Geometry + Vision segmentation to isolate answers
  └─────────┬─────────┘
            │
  ┌─────────▼─────────┐
  │  OCR Extraction   │  — Google Cloud Vision extracts handwritten text
  └─────────┬─────────┘
            │
  ┌─────────▼─────────┐
  │  Semantic Score   │  — Sentence Transformers vs. reference answer
  └─────────┬─────────┘
            │
  ┌─────────▼─────────┐
  │  PDF Generation   │  — Produce annotated result sheet
  └─────────┬─────────┘
            │
  ┌─────────▼─────────┐
  │   Return JSON     │  — score + feedback + pdf_url → Node.js
  └───────────────────┘
```

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.12 or higher
- A Google Cloud project with **Vision API** enabled
- Google Cloud service account credentials (`.json` key file)

### 1. Clone the Repository

```bash
git clone <your-repo-link>
cd paper_valuation
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
GOOGLE_APPLICATION_CREDENTIALS=path/to/your-service-account.json
FLASK_ENV=development
PORT=5000
```

### 5. Run the Server

```bash
python api/app.py
```

The service will be available at **http://localhost:5000** 🚀

---

## 📦 Key Dependencies

| Package | Purpose |
|---------|---------|
| `flask` | Web framework & REST API |
| `google-cloud-vision` | Handwritten text OCR extraction |
| `sentence-transformers` | Semantic similarity scoring |
| `Pillow` | Image preprocessing |
| `reportlab` / `fpdf` | PDF result sheet generation |
| `python-barcode` | Barcode generation & scanning |
| `python-dotenv` | Environment variable management |

---

## 🔗 Related Repositories

| Repository | Description |
|-----------|-------------|
| `paper-eval-orchestrator` | Node.js + Express layer — web UI, PostgreSQL storage & API coordination |

---

## 🛡️ License

This project was developed as part of a **Final Year Engineering Capstone**.  
All rights reserved © 2024.
