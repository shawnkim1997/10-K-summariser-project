# Technical Notes: 10-K Financial Analyzer

Reference document for developers and reviewers. This describes a key architectural decision made during development.

---

## Technical Challenge: Handling Large-Scale Financial Filings

During the initial development, I encountered a **429 Resource Exhausted** error due to the massive size of 10-K filings exceeding the LLM's token quota and rate limits.

### Consultation & Architectural Pivot

After consulting with a senior software engineer, I re-architected the application to optimize token usage. Instead of processing the entire document, I implemented a **"Selective Section Extraction"** strategy.

### Implemented Solution

| Component | Description |
|-----------|-------------|
| **Targeted Parsing** | Developed a regex-based parser to isolate only critical sections: **Item 7 (MD&A)** and **Item 8 (Financial Statements)**. |
| **Token Optimization** | Integrated a **"Chunking & Filtering"** logic to remove boilerplate legal text, sending only high-signal data to the Gemini API. |
| **Efficiency** | This reduced token consumption by **over 80%**, ensuring stable performance within free-tier limits while maintaining analytical depth. |

### Code References

- **Section extraction**: `find_item_section()`, `ITEM7_PATTERNS`, `ITEM8_PATTERNS` in `app.py`
- **Pre-filtering**: `prefilter_after_item7()` — drops PART I, ITEM 1–6; only content from Item 7 onward is used
- **Smart chunking**: `smart_chunk()` — when a section exceeds a character limit, keeps head + tail to preserve quantitative data while cutting tokens

These changes allow the app to stay within API rate limits without sacrificing the quality of the CFA-style analysis.
