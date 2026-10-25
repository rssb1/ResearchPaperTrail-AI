# Manual Testing Guide

The automated tests in `tests/test_basic.py` cover chunking logic offline.
The checks below require a real `OPENAI_API_KEY` and verify that the
retrieval + answer pipeline is actually grounded in the uploaded PDFs —
i.e. that it isn't inventing citations or findings.

## Setup

1. Install dependencies and set `OPENAI_API_KEY` in `.env` (see README).
2. Run `streamlit run app.py`.
3. Upload 1–2 PDFs you know the content of (a paper you've read works best,
   so you can sanity-check the answers yourself).

## Test 1 — Answers are grounded and cited

1. Ask a question whose answer is clearly stated in one of the papers
   (e.g. "What dataset did this paper use?").
2. Verify:
   - The answer includes a citation in the form `(Paper Title, p. X)`.
   - Expand "Show supporting passage(s)" and confirm the cited page
     actually contains text supporting the claim.
   - The page number shown matches the page in the original PDF (open the
     PDF separately and check).

## Test 2 — Out-of-scope questions are refused, not hallucinated

1. Ask a question the papers clearly do NOT answer, e.g. "What was the
   stock price of Apple in 2020?" or "What does this paper say about
   the French Revolution?"
2. Verify the app responds with the standard refusal message (defined in
   `llm.py` as `NO_ANSWER_PHRASE`) rather than fabricating an answer.

## Test 3 — Multi-paper retrieval picks the right source

1. Upload two different papers covering different topics.
2. Ask a question that only one of them answers.
3. Verify the citation in the answer points to the correct paper title,
   not the other one.

## Test 4 — Compare Papers table has no invented fields

1. With 2+ papers uploaded, click "Compare Papers".
2. For each paper, check the row against the actual PDF:
   - Every filled-in field (Research Question, Methods, Datasets,
     Evaluation Metrics, Limitations) should be traceable to real text
     in that paper.
   - Any field the paper doesn't explicitly discuss should read
     **"Not specified"** rather than a guess.

## Test 5 — Error handling

- **Bad PDF**: Upload a non-PDF file renamed to `.pdf`, or a corrupted PDF.
  The app should show an error for that file and continue processing the rest.
- **Scanned/image-only PDF**: Upload a PDF with no selectable text. The app
  should report that no extractable text was found, not crash.
- **Missing API key**: Unset `OPENAI_API_KEY` and restart the app. The
  sidebar should show a clear warning instead of a stack trace.
- **Empty question**: Submitting with no papers uploaded should show the
  "upload a PDF first" prompt, not attempt a search.
