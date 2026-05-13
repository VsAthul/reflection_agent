# FAQ Reflection Agent

An intelligent multi-LLM FAQ answering system with **reflective validation**. This agent generates answers to FAQ questions using Groq's LLaMA 3.3 (70B), then validates them using a local Ollama LLaMA 3.1 (8B) model. Invalid answers automatically retry up to 3 times before being marked as unanswerable.

## ✨ Features

- **Dual-Model Architecture**: Separate LLMs for generation and validation
  - **Generator**: Groq LLaMA 3.3-70b (answer creation)
  - **Validator**: Local Ollama LLaMA 3.1-8b (quality assurance)
- **Reflective Loop**: Automatic retry mechanism for invalid answers (up to 3 attempts)
- **Structured Outputs**: Pydantic-validated responses for both generation and validation
- **LangGraph Orchestration**: State-machine-based workflow with conditional routing
- **Comprehensive Logging**: Detailed console output tracking each step
- **JSON I/O**: Read questions from `inputs/input.json`, write results to `outputs/output.json`

## 🏗️ Architecture

### Workflow

```
Load Question → Generate Answer → Validate Answer → Decision
                                        ↓
                                   Valid? → Save Result → Next Question
                                        ↓
                                   Invalid & Retries Left → Retry Generation
                                        ↓
                                   Max Retries Reached → Save as "No Answer"
```

### Key Components

| Component | Purpose |
|-----------|---------|
| `graph.py` | LangGraph state machine; defines nodes, edges, and routing logic |
| `state.py` | Pydantic models for structured outputs; shared state schema |
| `llm.py` | LLM initialization; Groq + Ollama configuration |
| `main.py` | Entry point; loads input, runs graph, saves output |
| `nodes/` | Four LangGraph nodes: load, generate, validate, save |

### Graph Nodes

1. **load_question_node**: Picks next question; resets per-question state
2. **generate_node**: Sends question to Groq LLM; parses structured answer
3. **validate_node**: Sends answer to Ollama LLM; judges validity
4. **save_result_node**: Records result (answered/no_answer); advances queue

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **Groq API Key** (free tier available at [groq.com](https://groq.com))
- **Ollama** installed locally with `llama3.1:8b` model pulled
  ```bash
  ollama pull llama3.1:8b
  ```

### Installation

1. **Clone/download the project**:
   ```bash
   cd reflective-agent
   ```

2. **Create a Python virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file in the project root:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   OLLAMA_BASE_URL=http://localhost:11434  # Optional; defaults to this
   ```

### Running the Agent

```bash
python main.py
```

**Output**:
- Console: Summary with ✅ (answered) / ❌ (no_answer) symbols per question
- JSON: Detailed results written to `outputs/output.json`

## 📁 Project Structure

```
reflective-agent/
├── main.py                    # Entry point
├── graph.py                   # LangGraph definition
├── state.py                   # Pydantic models & shared state
├── llm.py                     # LLM configuration
├── requirements.txt           # Python dependencies
├── .env                       # API keys (not in repo)
├── nodes/
│   ├── load_question_node.py  # Load next question
│   ├── generate_node.py       # Groq generation
│   ├── validate_node.py       # Ollama validation
│   └── save_result_node.py    # Result persistence
├── inputs/
│   └── input.json             # FAQ questions (faqQuestion, Question)
├── outputs/
│   └── output.json            # Generated answers + validation results
└── graph_image.png            # Mermaid diagram of LangGraph flow
```

## 📊 Input/Output Format

### Input (`inputs/input.json`)

```json
[
  {
    "faqQuestion": 1,
    "Question": "What is TensorFlow?"
  },
  {
    "faqQuestion": 2,
    "Question": "What is the difference between supervised and unsupervised learning?"
  }
]
```

### Output (`outputs/output.json`)

```json
{
  "totalQuestions": 2,
  "processedAt": "2026-05-12T10:18:13.994790+00:00",
  "results": [
    {
      "faqQuestion": 1,
      "Question": "What is TensorFlow?",
      "iterationCount": 1,
      "status": "answered",
      "Answer": "TensorFlow is an open-source machine learning library...",
      "checkedAnswer": true,
      "remarks": "Accurate and informative."
    },
    {
      "faqQuestion": 3,
      "Question": "What is a neural network?",
      "iterationCount": 3,
      "status": "no_answer",
      "Answer": "-",
      "checkedAnswer": false,
      "remarks": "Max retries exhausted without valid answer."
    }
  ]
}
```

## ⚙️ Configuration

### Retry Behavior

Edit `graph.py` to adjust max retries:
```python
MAX_RETRIES = 3  # Change this value
```

### LLM Parameters

**Groq (Generation)**:
- Model: `llama-3.3-70b-versatile`
- Temperature: `0.3` (low for factual answers)
- Max tokens: `1024`

**Ollama (Validation)**:
- Model: `llama3.1:8b`
- Temperature: `0.1` (very low for strict validation)

Edit `llm.py` to modify these settings.

## 🔍 How It Works

1. **Agent loads** FAQ questions from JSON
2. **For each question**:
   - Sends to Groq LLM with a generation prompt
   - Groq returns structured answer (text + confidence score)
   - Validates with Ollama LLM using a validation prompt
   - If invalid and retries remain: try again with original question
   - If valid or retries exhausted: save result and move to next question
3. **Compilation**: Graph generates Mermaid diagram (`graph_image.png`)
4. **Output**: All results written to `outputs/output.json` with timestamps

## 🛠️ Extending the Agent

### Add a Custom Node

1. Create `nodes/my_node.py`:
   ```python
   from state import FAQAgentState

   def my_node(state: FAQAgentState) -> FAQAgentState:
       # Your logic here
       return {**state, ...}
   ```

2. Register in `graph.py`:
   ```python
   graph.add_node("my_node", my_node)
   graph.add_edge("previous_node", "my_node")
   ```

### Modify Prompts

Edit the `*_PROMPT` templates in `nodes/generate_node.py` and `nodes/validate_node.py`.

## 📋 Dependencies

| Package | Purpose |
|---------|---------|
| `langchain` | LLM abstraction layer |
| `langgraph` | State machine orchestration |
| `langchain-groq` | Groq integration |
| `langchain-ollama` | Ollama integration |
| `pydantic` | Data validation & structured outputs |
| `python-dotenv` | Environment variable management |

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| `GROQ_API_KEY not found` | Add `GROQ_API_KEY` to `.env` file |
| `Ollama connection failed` | Ensure Ollama is running (`ollama serve`) and model is pulled |
| `llama3.1:8b not found` | Run `ollama pull llama3.1:8b` |
| Input file not found | Ensure `inputs/input.json` exists in project root |

## 📝 Example Run

```
============================================================
   FAQ Reflection Agent — Starting
============================================================

[Main] Loaded 5 questions from inputs/input.json

[Main] Running graph for 5 question(s)...

============================================================
[LoadQuestion] Processing FAQ #1: What is TensorFlow?
============================================================
[GenerateNode] Q1 | Attempt 1/3 | Question: What is TensorFlow?
[GenerateNode] Generated answer (confidence=0.92): TensorFlow is an open-source...

[ValidateNode] Q1 | Attempt 1/3 | Validating answer...
[ValidateNode] Valid

[SaveResultNode] Q1 | Saving result...

...

============================================================
   Summary
============================================================
  ✅  Q1 | What is TensorFlow? → Answered in 1 iteration
  ✅  Q2 | What is supervised learning? → Answered in 1 iteration
  ✅  Q3 | What is neural networks? → Answered in 1 iteration
  ✅  Q4 | What is gradient descent? → Answered in 1 iteration
  ✅  Q5 | What is transfer learning? → Answered in 1 iteration

[Main] Results saved → outputs/output.json
```

## 📄 License

This project is provided as-is for educational and training purposes.
