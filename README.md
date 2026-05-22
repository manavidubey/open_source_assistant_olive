# 🤖 AI Personal Assistant — OSS vs Frontier Evaluation

A comprehensive project that builds, deploys, and evaluates two AI personal assistants: one powered by an **open-source model** (Qwen 2.5) and another by a **frontier model** (Google Gemini / OpenAI GPT-4).

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38+-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🎯 Overview

| Feature | OSS Assistant | Frontier Assistant |
|---|---|---|
| **Model** | Qwen 2.5-0.5B-Instruct | Gemini 2.0 Flash / GPT-4.1 |
| **Provider** | Hugging Face Inference API | Google AI / OpenAI API |
| **Parameters** | 0.5B | ~Undisclosed (large) |
| **Cost** | Free (HF Inference) | Pay-per-token |
| **Multi-turn** | ✅ Sliding window + summary | ✅ Sliding window + summary |
| **Tools** | ✅ Calculator, DateTime, Units | ✅ Calculator, DateTime, Units |
| **Guardrails** | ✅ 3-layer safety filter | ✅ 3-layer safety filter |
| **Observability** | ✅ Structured JSON logging | ✅ Structured JSON logging |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│                 Streamlit UI                     │
│         (Single / Side-by-Side Compare)          │
└─────────────┬──────────────────┬────────────────┘
              │                  │
    ┌─────────▼─────────┐  ┌────▼──────────────┐
    │  OSS Assistant     │  │ Frontier Assistant │
    │  (Qwen 2.5 / HF)  │  │ (Gemini / OpenAI)  │
    └─────────┬─────────┘  └────┬──────────────┘
              │                  │
    ┌─────────▼──────────────────▼────────────────┐
    │            Shared Infrastructure             │
    │  ┌──────────┐ ┌───────────┐ ┌────────────┐  │
    │  │  Memory   │ │ Guardrails│ │   Tools    │  │
    │  │ (Sliding  │ │ (3-layer  │ │(Calculator │  │
    │  │  Window + │ │  safety)  │ │ DateTime   │  │
    │  │  Summary) │ │           │ │ Units)     │  │
    │  └──────────┘ └───────────┘ └────────────┘  │
    │  ┌─────────────────────────────────────────┐ │
    │  │          Observability Logger            │ │
    │  │     (Structured JSON + Metrics)          │ │
    │  └─────────────────────────────────────────┘ │
    └─────────────────────────────────────────────┘
```

### Key Design Decisions

1. **Shared Pipeline Architecture**: Both assistants inherit from `BaseAssistant`, sharing the same memory, guardrails, tools, and logging infrastructure. This ensures a fair comparison — differences are purely model capability.

2. **Sliding Window + Summary Memory**: Recent messages (configurable, default 20) are kept at full fidelity. Older messages are compressed into a running summary, giving the model both short-term precision and long-term awareness.

3. **3-Layer Safety Guardrails**:
   - **Layer 1**: Regex-based jailbreak pattern detection (input)
   - **Layer 2**: Harmful content category detection (input)
   - **Layer 3**: Output safety validation (output)

4. **LLM-as-Judge Evaluation**: Uses a frontier model as an impartial judge to score responses on hallucination, bias, and safety — providing more nuanced evaluation than simple keyword matching.

---

## 🚀 Setup Instructions

### Prerequisites

- Python 3.11+
- A Hugging Face API token (free at [huggingface.co](https://huggingface.co/settings/tokens))
- A Google Gemini API key (free at [aistudio.google.com](https://aistudio.google.com/apikey)) **OR** an OpenAI API key

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/olive-assignment.git
cd olive-assignment

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

### Configuration

Edit `.env` with your API keys:

```env
HF_API_TOKEN=hf_your_token_here
GEMINI_API_KEY=your_gemini_key_here
# OR
OPENAI_API_KEY=sk-your_openai_key_here
```

### Running the App

```bash
# Launch the Streamlit chat interface
streamlit run app.py

# Run the evaluation pipeline
python -m evaluation.run_evaluation
```

---

## 📁 Project Structure

```
olive-assignment/
├── app.py                          # Streamlit chat UI
├── config.py                       # Configuration management
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Docker deployment
├── .env.example                    # Environment template
│
├── assistants/
│   ├── base.py                     # Abstract base assistant
│   ├── oss_assistant.py            # Qwen 2.5 via HF API
│   └── frontier_assistant.py       # Gemini / OpenAI
│
├── memory/
│   └── conversation_memory.py      # Sliding window + summary
│
├── guardrails/
│   └── safety.py                   # 3-layer safety filters
│
├── tools/
│   └── tool_manager.py             # Calculator, DateTime, Units
│
├── observability/
│   └── logger.py                   # Structured JSON logging
│
├── evaluation/
│   ├── prompts.py                  # 38 curated test prompts
│   ├── judge.py                    # LLM-as-judge evaluator
│   └── run_evaluation.py           # Evaluation pipeline + viz
│
├── deployment/
│   └── hf_spaces/                  # HF Spaces deployment
│       ├── app.py                  # Gradio app
│       └── README.md               # Space metadata
│
└── reports/
    ├── evaluation_report.pdf       # 1-page evaluation report
    └── figures/                    # Generated charts
```

---

## 🧪 Evaluation Framework

### Methodology

The evaluation tests **35+ curated prompts** across three categories:

| Category | # Prompts | What it Measures |
|---|---|---|
| **Factual Accuracy** | 15 | Hallucination rate — does the model fabricate facts? |
| **Content Safety** | 10 | Jailbreak resistance — does it refuse adversarial prompts? |
| **Bias & Fairness** | 10 | Stereotype handling — does it show harmful biases? |

Each response is scored by an **LLM-as-Judge** (Gemini 2.0 Flash) on a 1–5 scale with structured JSON output.

### Running Evaluation

```bash
python -m evaluation.run_evaluation
```

This generates:
- `reports/evaluation_results.json` — Raw results
- `reports/figures/overall_comparison.png` — Summary dashboard
- `reports/figures/heatmap_*.png` — Per-prompt score heatmaps

---

## 🚢 Deployment (Bonus Task)

We support deploying the OSS Model across multiple platforms. The architecture's abstraction layer ensures that whether the model runs on Hugging Face Spaces, Modal, or RunPod, the **memory, safety guardrails, and tools** function identically.

### 1. Hugging Face Spaces (Recommended for zero-cost)
Hugging Face Spaces natively supports Streamlit. You can deploy this exact dual-model UI, complete with memory, tools, and guardrails:
1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space)
2. Select **Streamlit** as the Space SDK.
3. Push this entire repository to the Space.
4. Go to Space Settings -> Variables and Secrets. Add your `GROQ_API_KEY` and other provider keys.
5. Your space will automatically install `requirements.txt` and run the app.

### 2. GPU Cloud Platforms (Modal, RunPod, Replicate)
If you require lower latency and want to self-host the weights for privacy, you can run the OSS model on a serverless GPU platform:
- **Modal:** Spin up an A10G/T4 container serving an OpenAI-compatible FastAPI endpoint. Configure `config.py` base_url to point to your Modal endpoint.
- **RunPod:** Use the official vLLM template on a Serverless endpoint. Very cost-effective for sustained traffic.
- **Ollama:** Best for local privacy. Run `ollama run qwen2.5:0.5b` and point the `base_url` to `http://localhost:11434/v1`.

### 📊 Cost & Latency Table (OSS Deployment Comparison)

This table compares deploying an OSS model (e.g., Qwen 2.5 8B) across various infrastructure platforms.

| Deployment Platform | Infrastructure | Est. Cost | Avg Latency (TTFT) | Complexity | Best For |
|---|---|---|---|---|---|
| **Hugging Face Spaces** | Free CPU/T4 Tier | **$0** | ~1000–2500ms | Low | Prototypes, portfolios, zero budget |
| **Groq / Together API** | LPU / Cloud GPU | **Free Tier** | ~100–300ms | Low | Production apps relying on APIs |
| **Modal (Serverless)** | A10G / T4 | **~$0.60/hr** (active) | ~200–500ms | Medium | Scalable, spiky traffic production |
| **RunPod (vLLM)** | RTX 4090 / A6000 | **~$0.40/hr** | ~150–400ms | High | High sustained throughput workloads |
| **Ollama (Local)** | Local M-series / GPU | **$0** (Hardware cost) | ~100–300ms | Low | Local testing, strict privacy |
| **Replicate** | Serverless GPU | **~$0.0002/sec** | ~400–800ms | Low | Quick serverless scaling |

---

## ⚖️ Tradeoffs

| Decision | Tradeoff |
|---|---|
| **HF Inference API** over local inference | Simpler setup, no GPU needed — but adds network latency and rate limits |
| **Regex-based guardrails** over ML classifiers | Faster, no extra model needed — but less nuanced than trained safety classifiers |
| **Sliding window memory** over RAG-based retrieval | Simpler, no vector DB needed — but loses detail in compressed summaries |
| **Qwen 2.5-0.5B** over larger OSS models | Deployable on free tier — but significantly less capable than 7B+ models |
| **LLM-as-judge** over human eval | Scalable and reproducible — but may have systematic judge biases |

---

## 🔮 What I'd Improve With More Time

1. **ML-based Safety Classifier**: Replace regex guardrails with a fine-tuned safety classifier (e.g., Llama Guard) for better detection
2. **RAG-based Memory**: Use a vector database for semantic retrieval of relevant past context, instead of naive summarization
3. **Larger OSS Model**: Deploy Qwen 2.5-7B or Llama 3.2-8B on Modal/RunPod for significantly better quality
4. **Streaming Responses**: Add token-by-token streaming for better UX
5. **More Tools**: Add web search (via Tavily/Serper), code execution, and file analysis
6. **A/B Testing Framework**: Randomized prompt routing with statistical significance testing
7. **Human Evaluation**: Supplement LLM-as-judge with human preference ratings
8. **Prompt Caching**: Implement prompt caching for repeated context (Gemini/Claude support this)
9. **Multi-language Evaluation**: Test bias and safety across multiple languages
10. **Automated CI/CD Evals**: Run evaluation suite on every commit via GitHub Actions

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
