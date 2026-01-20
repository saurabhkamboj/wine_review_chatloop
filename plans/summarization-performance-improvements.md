# Summarization Performance Improvements

## Problem
The `summarize_results_with_llm()` function is taking too long.

## Current Implementation (wine_reviews.py:72-111)
- Calls OpenAI API with `gpt-4o-mini`
- Sends formatted results (up to 10 reviews) + memory context + instructions
- Max output tokens: 700

## Root Causes of Slow Summarization
1. **Model choice** - `gpt-4o-mini` is slower than cheaper/faster alternatives
2. **Large input token count** - Each review includes all fields, even irrelevant ones
3. **Verbose prompt instructions** - 4 numbered objectives add tokens
4. **Synchronous blocking call** - User waits for full completion before seeing anything

---

## Recommended Optimizations

### 1. Switch to Faster/Cheaper Model (High Impact)
**File:** `wine_reviews.py:15`

**Current:** `gpt-4o-mini`

**Change:** Use `gpt-4.1-nano` - significantly faster and cheaper for simple summarization tasks.

| Model | Input (per 1M tokens) | Output (per 1M tokens) | Speed |
|-------|----------------------|------------------------|-------|
| gpt-4o-mini | $0.15 | $0.60 | Baseline |
| gpt-4.1-nano | $0.10 | $0.40 | ~2x faster |

**Expected improvement:** 40-60% faster response time, ~30% cost reduction

### 2. Reduce Input Token Count (High Impact)
**File:** `wine_reviews.py:83-92`

**Current:** Sends all fields for each result including redundant similarity scores.

**Change:** Only include fields relevant to summarization:
- Remove `similarity` score from prompt (internal metric, not useful for summary)
- Condense format: `"1. Title (Winery) - Country | Points: X, Price: $Y"`

**Expected improvement:** 20-30% fewer input tokens = faster processing + lower cost

### 3. Parallel Memory + Classification (Medium Impact)
**File:** `wine_reviews.py:121-127`

**Current:** Sequential calls to `get_relevant_memories()` then `classify_query()`.

**Change:** Run both in parallel using `concurrent.futures`:
```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=2) as executor:
    memory_future = executor.submit(get_relevant_memories, user_query)
    classify_future = executor.submit(classify_query, user_query)
    memories = memory_future.result()
    classification = classify_future.result()
```

**Expected improvement:** Save 50-150ms (whichever call is shorter)

### 4. Implement Response Streaming (Medium Impact - Perceived Performance)
**File:** `wine_reviews.py:106-111`

**Current:** `llm_client.responses.create()` waits for full response.

**Change:** Use streaming API to show results progressively:
```python
stream = llm_client.responses.create(..., stream=True)
for chunk in stream:
    yield chunk.output_text
```

Requires updating Gradio interface to handle streaming output.

**Expected improvement:** User sees first words in ~100-200ms instead of waiting for full response

### 5. Streamline Prompt Instructions (Low Impact)
**File:** `wine_reviews.py:97-103`

**Current:** 4 verbose numbered objectives.

**Change:** Replace with concise single instruction:
```
"Summarize these wines briefly: highlight top matches, note price/rating trends, recommend one to try first."
```

**Expected improvement:** 10-15% fewer tokens

### 6. Reduce max_output_tokens (Low Impact)
**File:** `wine_reviews.py:109`

**Current:** 700 tokens max.

**Change:** Reduce to 400-500 tokens. Summaries rarely need 700 tokens.

**Expected improvement:** 5-10% faster

---

## Implementation Priority

| # | Optimization | Impact | Effort | Priority |
|---|-------------|--------|--------|----------|
| 1 | Switch to gpt-4.1-nano | High | Low | **P0** |
| 2 | Reduce input tokens | High | Low | **P0** |
| 3 | Parallel memory+classify | Medium | Low | **P1** |
| 4 | Streaming responses | Medium (perceived) | Medium | **P1** |
| 5 | Streamline prompt | Low | Low | **P2** |
| 6 | Reduce max_output_tokens | Low | Low | **P2** |

---

## Files to Modify

1. **wine_reviews.py**
   - Line 15: Change `LLM_MODEL = 'gpt-4o-mini'` to `LLM_MODEL = 'gpt-4.1-nano'`
   - Lines 83-92: Condense result format, remove similarity scores
   - Lines 121-127: Add parallel execution for memory + classification
   - Lines 106-111: Add streaming support
   - Gradio interface: Update to handle streaming output

---

## Verification

1. Run the app and execute a search query
2. Compare timing breakdown before/after changes
3. Verify summarization quality remains acceptable
4. Confirm streaming shows progressive output in UI
