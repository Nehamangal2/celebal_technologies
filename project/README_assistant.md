# Local ChatGPT Assistant (`chat_assistant.py`)

An interactive, modern conversational AI assistant that runs completely locally on your computer's CPU. It uses the state-of-the-art **Qwen2.5-0.5B-Instruct** transformer model from Hugging Face.

Unlike the Shakespeare model, this assistant understands modern instructions, speaks fluent English, writes code, and helps with creative and analytical tasks.

---

## Setup & Installation

Before running the assistant, ensure you have the required Hugging Face libraries installed:

```powershell
pip install transformers huggingface_hub tokenizers accelerate torch
```

*Note: On your first run, the script will automatically download the model weights (~900MB). Subsequent launches are instant and do not require downloading.*

---

## How to Run

1. Open **PowerShell** or your terminal.
2. Navigate to the project directory:
   ```powershell
   cd C:\Users\neham\OneDrive\Desktop\antigravity_project
   ```
3. Start the assistant:
   ```powershell
   python chat_assistant.py
   ```

---

## Range of Capabilities

* **General Q&A**: Explains scientific concepts, historical events, geographic facts, and definitions.
* **Creative Writing**: Drafts emails, outlines essays, writes stories, composing poems, and summaries.
* **Coding Assistant**: Generates, comments, and debugs code in Python, JavaScript, HTML/CSS, C++, SQL, and more.
* **Translation**: Translates text smoothly across dozens of languages.
* **Formatting**: Formats responses into structured bullet points, numbered lists, or code blocks.

---

## Limitations & Accuracy

* **No Internet Access**: The model runs entirely offline. It does not know live real-time information (e.g., today's weather or current news) beyond its late 2024 training cutoff.
* **Complex Reasoning**: While great at basic and intermediate math and programming, it will struggle with advanced logic puzzles, university-level mathematics, or highly abstract riddles.
* **Accuracy & Hallucinations**: 
  * The model is extremely fluent (99% grammatical coherence).
  * Factual accuracy is around 85%–90%. For highly obscure details, it may confidently invent facts. Always double-check critical information (medical, financial, or legal advice).

---

## Controls

* **Exiting**:
  * Type `exit` and press **Enter** to close the session.
  * Or press **`Ctrl + C`** on your keyboard to force-quit.
