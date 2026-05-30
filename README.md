# 🍯 Advanced Honey Purity Serverless Engine

An advanced Machine Learning pipeline and interactive diagnostic tool built to evaluate honey purity from chemical profile data. This project trains a **Random Forest Classifier** completely in memory to avoid serverless disk write restrictions and features a fully responsive, clean frontend interface deployed natively on Vercel.

---

## 🚀 Live Demo & API Documentation
- **Production URL:** `https://your-vercel-project-link.vercel.app/`
- **Interactive Swagger Docs:** `https://your-vercel-project-link.vercel.app/docs`

---

## ✨ Features
- **Zero-Write ML Engine:** Trains and caches the model components entirely inside RAM to comply with Vercel's read-only serverless environment.
- **Robust Feature Engineering:** Addresses data-leakage vulnerabilities by removing the continuous `Purity` targets prior to model feature ingestion.
- **Interactive Web Interface:** Beautiful HTML5 user interface styled with Bootstrap 5 utilizing asynchronous (`fetch`) API communication for immediate diagnostics.
- **Auto-Fallback Engine:** Automatically synthesizes balanced baseline datasets if the primary file pathways shift during distributed deployments.

---

## 🛠️ Repository Architecture

Your deployment directory must have the following file structure to compile properly on Vercel:

```text
honey-purity-engine/
│
├── api.py                    # Unified Backend API + Frontend Application Interface
├── requirements.txt          # Python dependency packages
├── vercel.json               # Serverless environment configuration profiles
└── honey_purity_dataset.csv  # Raw Honey Chemical profiling dataset
