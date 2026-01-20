# Plan: Improve Gradio UI with Timing Metrics

## Goal
Enhance the Gradio UI to be cleaner and display inline timing metrics for each step of the query processing pipeline.

## Changes

### 1. Add timing instrumentation to `handle_search()`

Wrap each operation with `time.perf_counter()` to capture:
- **Memory search** — `get_relevant_memories()`
- **Classification** — `classify_query()`
- **Embedding** (if semantic) — `embed_query_text()`
- **DB retrieval** — `search_reviews()`
- **Summarization** — `summarize_results_with_llm()`
- **Memory store** — `store_interaction()`
- **Total time**

### 2. Return timing info inline with response

Append a formatted timing breakdown at the end of the response:

```
---
**Timing Breakdown**
| Step | Time |
|------|------|
| Memory search | 0.12s |
| Classification | 0.45s |
| Embedding | 0.08s |
| DB retrieval | 0.03s |
| Summarization | 1.23s |
| Memory store | 0.15s |
| **Total** | **2.06s** |
```

### 3. Improve UI layout

- Stack input and output vertically (remove side-by-side columns)
- Add subtitle/description text
- Use `show_progress="minimal"` for cleaner loading state

## Files to Modify

- [wine_reviews.py](wine_reviews.py) — timing logic and UI changes

## Verification

1. Run `python dev.py` to start the server with auto-reload
2. Submit a query and verify:
   - Response displays correctly
   - Timing table appears at the bottom
   - All timing values are reasonable
3. Test both semantic and keyword queries to ensure embedding step shows "N/A" or is skipped for keyword queries
