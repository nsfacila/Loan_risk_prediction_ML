# Production-Ready Loan Risk Prediction: End-to-End MLOps Solution

[![GitHub Actions CI/CD](https://img.shields.io/badge/CI/CD-GitHub%20Actions-2088FF?style=for-the-badge&logo=github-actions&logoColor=white)](https://github.com)
[![Docker Hub Registry](https://img.shields.io/badge/Docker%20Hub-Published-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://hub.docker.com)
[![Deployed to Render](https://img.shields.io/badge/Deploy-Render-4338ca?style=for-the-badge&logo=render&logoColor=white)](https://tu-enlace-de-render.com)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)

> **Automated enterprise-grade solution for credit risk assessment in financial lending, deployed through a robust and decoupled MLOps architecture.**

⚡ **[View Live Demo on Render](https://your-render-link.com)**

---

## 1. Project Overview

Financial institutions need reliable methods to assess the risk associated with loan applications. The objective of this project is to develop and compare multiple machine learning classification models capable of predicting whether a loan applicant represents a high or low credit risk.

Unlike a traditional academic machine learning project, this solution follows a production-oriented workflow that includes model deployment, containerization, cloud hosting, and CI/CD automation.

The project includes:

- Exploratory Data Analysis (EDA)
- Data preprocessing and feature engineering
- Training and evaluation of multiple classification models
- Model comparison and selection
- Interactive Streamlit dashboard
- Docker containerization
- Automated CI/CD pipeline with GitHub Actions
- Docker Hub integration
- Cloud deployment using Render
---

## 2. Solution Features (Advanced MLOps Architecture)

Unlike traditional deployment approaches, this project implements a decoupled cloud production workflow powered by **CI/CD**, optimizing both time and infrastructure resources.

* **Production-Grade Machine Learning:**
    * **Data Leakage Prevention:** Strict use of Scikit-Learn `Pipelines` to encapsulate mathematical transformations and preprocessing steps.
    * **Bayesian Optimization:** Automated hyperparameter tuning using `Optuna`.
    * **Stratified Cross-Validation:** Comprehensive evaluation strategy to ensure model stability and robustness.
* **Automated CI/CD Pipeline (GitHub Actions ➡️ Docker Hub ➡️ Render):**
    * **Efficient Build Process:** Whenever a `git push` is made to the `main` branch, GitHub Actions provisions a virtual environment, reads the `Dockerfile`, builds the image, and installs dependencies using GitHub's cloud infrastructure.
    * **Centralized Registry:** The packaged image is automatically published to a Docker Hub repository using securely managed credentials (`Secrets`).
    * **Instant Deployment:** Render detects the updated image, downloads the pre-built artifact from Docker Hub, and deploys it within seconds, eliminating build-time memory consumption on the production server.
* **Feedback Loop System:** Design and production implementation of a loop that securely captures and stores new predictions and user inputs, enabling future monitoring of **Data Drift** and model retraining strategies.

---

## 3. Technology Stack

* **Core Data Science & Machine Learning:** Python, Scikit-Learn, Optuna, Pandas, NumPy, Seaborn, Matplotlib.
* **MLOps Infrastructure:** Docker, Docker Compose, GitHub Actions, Docker Hub, Render.
* **User Interface:** Streamlit.

---

## 4. Project Structure

El repositorio sigue un estándar profesional y modularizado para facilitar el mantenimiento del software:

```text
Loan_Risk_Prediction_ML/
│
├── .github/
│   └── workflows/
│       └── docker-publish.yml
│
├── assets/
├── data/
├── models/
├── notebooks/
├── scripts/
├── utils/
│
├── app.py
├── Dockerfile
├── requirements.txt
├── .dockerignore
├── README.md
├── LICENSE
└── .gitignore
```

---

## Docker Containerization

The application was containerized using Docker to ensure portability, consistency, and reproducibility across different environments.

### Build Docker Image

```bash
docker build -t loan-risk-prediction .
```

### Run Locally

```bash
docker run -p 8501:8501 loan-risk-prediction
```

### Benefits

- Environment consistency
- Simplified deployment
- Reproducible builds
- Easier cloud integration

---

## CI/CD Pipeline

A fully automated CI/CD workflow was implemented using GitHub Actions.

### Pipeline Process

1. Code is pushed to GitHub
2. GitHub Actions workflow is triggered
3. Docker image is built automatically
4. Image is published to Docker Hub
5. Updated image becomes available for deployment

### GitHub Secrets Configuration

Sensitive credentials are securely managed using GitHub Secrets:

- `DOCKER_USERNAME`
- `DOCKER_TOKEN`

This prevents credentials from being exposed in the repository.

---

## Docker Hub Integration

The Docker image is automatically published to Docker Hub whenever changes are pushed to the repository.

### Benefits

- Versioned container images
- Centralized image registry
- Simplified deployment workflow
- Automated artifact management

Docker Hub Repository:

```text
nsfacila/loan-app
```

---

## Cloud Deployment with Render

The application is deployed on Render using the Docker image published in Docker Hub.

### Deployment Architecture

```text
GitHub
   │
   ▼
GitHub Actions
   │
   ▼
Docker Build
   │
   ▼
Docker Hub
   │
   ▼
Render
   │
   ▼
Production Application
```

### Production URL

https://credit-loan-app.onrender.com/

### Deployment Benefits

- Automated updates
- Scalable hosting
- Simplified infrastructure management
- Fast deployment workflow

---

## Skills Demonstrated

This project demonstrates practical experience in:

- Machine Learning Classification
- Data Preprocessing
- Feature Engineering
- Model Evaluation
- Streamlit Development
- Docker Containerization
- Docker Hub Integration
- GitHub Actions
- CI/CD Automation
- Cloud Deployment
- MLOps Fundamentals
- Git & GitHub Best Practices

---

## 👩‍💻 Author

**Noelia Sánchez**

Data Analyst | Machine Learning | Business Intelligence

📌 GitHub: https://github.com/nsfacila

---

