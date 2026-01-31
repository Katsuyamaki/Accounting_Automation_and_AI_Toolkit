# 🧠 Secure-Finance-API

### *CPA-Architected AI Accounting Engine*
 
This is a cloud-native classification engine that automates technical accounting with a **Zero-Trust architecture**. Built to handle multi-entity structures, it utilizes a **Hybrid "Learning Autocorrect" Waterfall** to categorize bank transactions with deterministic accuracy.

## 🏛️ Enterprise Architecture

* **Infrastructure**: Serverless deployment on **Google Cloud Run** using Python/FastAPI for high scalability and low overhead.
* **Persistent Logic Repository**: Proprietary accounting rules and correction history are stored in isolated **Google Cloud Storage (GCS)** buckets.
* **Identity & Access**: API keys are secured via **Google Secret Manager** (AES-256 encryption) with restricted IAM service account access.
* **AI Layer**: Integrated **Vertex AI (Gemini 2.0)** for probabilistic reasoning when deterministic rules are exhausted.

## 🔐 SOC 2 Alignment & Security

* **Confidentiality (Zero-Persistence)**: Sensitive financial records are processed strictly in **volatile RAM** using `io.BytesIO`. No PII or transaction data is ever written to persistent disk.
* **Privacy**: Multi-tenant data segregation ensures Client A’s rules are physically and logically isolated from Client B.
* **Processing Integrity**: A tiered "Waterfall" prevents AI hallucinations by anchoring the classification in deterministic historical truths before using LLMs.

## ⚙️ The Logic Waterfall

1. **🧠 History (Golden Record)**: Matches against a persistent memory of manual overrides (Correction History).
2. **⚡ Special Logic**: Automated intercompany splits and partner distribution logic (e.g., ENTITY_B/C splits).
3. **📖 Transaction Rules**: Keyword-based dictionary mapping for recurring vendors (e.g., PEPCO, WAWA).
4. **🤖 AI Safety Net**: Vertex AI fallback for novel or complex transaction descriptions.

## 🚀 Getting Started

### Prerequisites

* Google Cloud SDK
* Python 3.10+
* Google Cloud Project with Billing Enabled

### Local Deployment (Termux/Zsh)

```zsh
# 1. Manage API Keys
./manage_vault.sh create [USERNAME]

# 2. Deploy to Cloud Run
gcloud run deploy accounting-engine --source . --region us-central1

```

## 📊 Project Status

* [x] Zero-Persistence RAM Ingestion
* [x] GCS Rule Bucket Integration
* [x] Multi-tenant Secret Management
* [ ] Automated Audit Logging (In Progress)

