# 🧪 HELM Evaluation Setup Guide

This guide walks you through setting up a virtual environment, installing dependencies, and running a basic evaluation using the Stanford CRFM HELM framework.

---

## 🔧 1. Environment Setup

Only run this section once when setting up your environment.

```bash
# Install virtualenv if not already installed
python3 -m pip install virtualenv
pip install --upgrade pip

# Create a Python 3.9 virtual environment
python3 -m virtualenv -p python3.9 helm-venv

# Activate the virtual environment
source helm-venv/bin/activate
```

---

## 📦 2. Install Required Packages

```bash
# Update system packages
sudo apt-get update

# Install Python 3.9 development headers
sudo apt-get install python3.9-dev

# Install blis and HELM
pip install blis
pip install --upgrade pip
pip install crfm-helm
```

---

## 📁 3. Download Configuration Files

Download the necessary configuration and schema files into your root working directory:

```bash
wget https://github.com/stanford-crfm/helm/blob/8c2fa4b6bab791c1dc3285ec3fdd63427f92b837/src/helm/benchmark/static/schema_medhelm.yaml
wget https://github.com/stanford-crfm/helm/blob/8c2fa4b6bab791c1dc3285ec3fdd63427f92b837/src/helm/benchmark/presentation/run_entries_biomedical.conf
```

---

## ✅ 4. Run Evaluation

Run the evaluation for the PubMedQA dataset using a specific Hugging Face model:

```bash
helm-run \
  --run-entries pubmed_qa \
  --max-eval-instances 10 \
  --suite med \
  --models-to-run Qwen/Qwen3-0.6B \
  --enable-huggingface-models Qwen/Qwen3-0.6B

# Or run in nohup
nohup bash -c 'helm-run --run-entries pubmed_qa --max-eval-instances 10000 --suite med --models-to-run Qwen/Qwen3-0.6B --enable-huggingface-models Qwen/Qwen3-0.6B && helm-summarize --schema schema_medhelm.yaml --suite med' > helm_run.log 2>&1 &

```

---

## 📌 Notes

- You can modify `--models-to-run` to test different models.
- The `--max-eval-instances` flag controls how many examples are used.
- Ensure you have access to the internet when downloading dependencies and configs.

---
