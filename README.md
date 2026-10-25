# ResearchPaperTrail AI Developed by Vibe Coding

A minimal, local research-paper intelligence app: upload PDFs, ask grounded
questions with page-level citations, and generate a structured comparison
table across papers.

Built as the smallest functional MVP — no auth, no database, no agent
framework. Everything lives in the Streamlit session; re-upload papers each
run (or keep the app open between questions).

## Features

- Upload multiple research paper PDFs.
- Extracts text per page (PyMuPDF) and splits it into overlapping, page-tagged chunks.
- Embeds chunks (OpenAI `text-embedding-3-small`) and retrieves the most
  relevant passages for each question via in-memory cosine similarity.
- Answers questions using **only** retrieved passages, with `(Paper, p. X)`
  citations, and clearly says so when the sources don't support an answer.
- Every answer shows its supporting passages, so you can verify the citation yourself.
- "Compare Papers" button builds a table of research question, methods,
  datasets, evaluation metrics, and limitations per paper — missing fields
  are marked "Not specified", never guessed.

## Stack

Python · Streamlit · PyMuPDF · OpenAI API (chat completions + embeddings)

## Setup

1. **Clone/unzip the project**, then create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set your API key:**

   ```bash
   cp .env.example .env
   ```

   Then edit `.env` and set:

   ```
   OPENAI_API_KEY=sk-your-real-key-here
   ```

   The app uses the OpenAI API for both the LLM and the embeddings model.
   It never logs or displays your key, and `.env` is already in `.gitignore`.

4. **Run the app:**

   ```bash
   streamlit run app.py
   ```

   It will open at `http://localhost:8501`.

## Using the app

1. In the sidebar, upload one or more PDF research papers and click
   **Process papers**. Each paper is text-extracted, chunked, and embedded
   (this calls the embeddings API, so it needs your key set).
2. Go to the **Chat** tab and ask questions. Answers cite `(Paper, p. X)`
   and include an expandable list of the exact passages used.
3. Go to the **Compare Papers** tab and click **Compare Papers** to generate
   a structured side-by-side table. You can download it as CSV.

## Project structure

```
app.py              Streamlit UI — sidebar upload, chat tab, compare tab
pdf_processor.py     PDF text extraction + overlapping chunking (no API calls)
embeddings.py        Embedding calls + in-memory cosine-similarity retrieval
llm.py               Grounded Q&A and structured comparison-table prompts
tests/test_basic.py  Offline unit tests for chunking (no API key needed)
MANUAL_TESTING.md    Step-by-step checks that answers stay grounded (needs API key)
.env.example         Template for your API key — copy to .env
requirements.txt     Python dependencies
```

## Testing

Run the offline unit tests (chunking logic, no API calls):

```bash
python -m unittest tests/test_basic.py -v
```

For end-to-end verification that answers are actually grounded in your PDFs
(not hallucinated), follow **`MANUAL_TESTING.md`** — it walks through
checking citations, refusals on out-of-scope questions, multi-paper
retrieval accuracy, and the comparison table.

## Known limitations (by design, for an MVP)

- Scanned/image-only PDFs aren't OCR'd — only PDFs with a real text layer work.
- No persistence: papers and chat history reset when you close the tab/app
  (there's a "Clear all papers" button for a clean slate).
- Chunking is page-scoped (a chunk never spans two pages) so page citations
  are always exact — long pages are just split into more chunks.
- Single embedding/LLM provider (OpenAI) is wired up; swapping providers
  means editing `embeddings.py` / `llm.py`.
