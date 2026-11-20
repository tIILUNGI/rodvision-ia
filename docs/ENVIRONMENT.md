# RoadVision AI — Environment & Hardware Specification

**Recorded:** 20 November 2026

## 1. Host Operating System & Hardware
- **OS:** Windows 11 Enterprise (win32)
- **CPU Architecture:** x86_64, 4 physical cores / 8 threads
- **RAM:** 15.7 GB Total / ~7.8 GB Available
- **Disk:** NVMe SSD, ~25 GB free

## 2. Core Runtimes & Tooling
- **Python:** 3.14.3 (`C:\Python314\python.exe` & isolated virtual environment `.venv/`)
- **Git:** 2.56.0.windows.1
- **Docker:** 29.2.0, build 0b9d198

## 3. Pinned Python Dependencies
| Package | Version | Purpose |
|---|---|---|
| `numpy` | 2.5.2 | High-performance numerical arrays and matrix math |
| `pillow` | 12.2.0 | Image I/O and pixel manipulation |
| `opencv-python-headless` | 4.13.0.92 | Geometric transformations, polygon rasterization |
| `pydantic` | 2.14.2 | Strict schema validation and data integrity |
| `pytest` | 8.4.2 | Automated testing suite |
| `pandas` | 3.1.1 | Tabular metrics manipulation |
| `matplotlib` | 3.11.2 | Visualization and chart rendering |
| `fastapi` | 0.162.0 | High-performance async REST API |
| `uvicorn` | 0.54.0 | ASGI web server |
| `PyJWT` | 2.15.1 | Cryptographic JWT token generation and validation |
| `bcrypt` | 5.0.0 | Secure password hashing |
| `httpx` | 0.28.1 | Asynchronous HTTP client for multimodal API calls |
