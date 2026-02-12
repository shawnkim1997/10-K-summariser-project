# 10-K Financial Analyzer

A web app that fetches the latest 10-K from SEC EDGAR for a given stock ticker (e.g. AAPL), then uses **Item 7 (MD&A)** and **Item 8 (Financial Statements)** to produce a CFA-style analysis: a detailed summary (financial health, profitability, key risks), a **CFA Investment Report** section, and key metrics (Revenue, Net Income, Operating Cash Flow). Powered by **Google Gemini**.

---

## Tech Stack

- **UI**: Streamlit  
- **Data**: sec-edgar-downloader (SEC EDGAR)  
- **AI**: Google Gemini (google-generativeai)  

---

## Requirements

- Python 3.9+  
- [Google API Key (Gemini)](https://aistudio.google.com/apikey)  
- An email address for SEC EDGAR (required for programmatic access; use a real address)  
- Optional: `.env` with `GOOGLE_API_KEY` and `SEC_EDGAR_EMAIL` (values will appear in the app sidebar if set)  

---

## How to Run

**Quick start:** Open a terminal, go to the project folder, create/activate a virtual environment, install dependencies, then run the app.

### 1. Open a terminal

- **Mac**: Spotlight (`Cmd + Space`) → type "Terminal" and open it  
- **Windows**: `Win + R` → type `cmd` and press Enter  

### 2. Go to the project folder

```bash
cd "/path/to/your/FQDC Project"
```

Replace with your actual project path if different.

### 3. Create a virtual environment (recommended, one-time)

```bash
python3 -m venv venv
```

### 4. Activate the virtual environment

**Mac / Linux:**

```bash
source venv/bin/activate
```

**Windows (Command Prompt):**

```bash
venv\Scripts\activate.bat
```

**Windows (PowerShell):**

```bash
venv\Scripts\Activate.ps1
```

You should see `(venv)` at the start of your prompt.

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

Requires an internet connection; may take 1–2 minutes.

### 6. Run the app

```bash
streamlit run app.py
```

Your browser should open at `http://localhost:8501`. If not, open that URL manually.

### 7. Configure and analyze

1. In the **sidebar**:  
   - **Google API Key (Gemini)**: paste your key from [Google AI Studio](https://aistudio.google.com/apikey)  
   - **SEC EDGAR Email Address**: your email (for SEC policy compliance)  
   - **Analysis only (1 API call)**: check this to use a single API call (useful if you hit rate limits)  
2. Enter a **stock ticker** (e.g. `AAPL`, `MSFT`) and click **Run Analysis**.  
3. When finished, you’ll see the **Detailed Analysis**, **CFA Investment Report**, and (if not in Analysis only mode) the **Key Financial Metrics** table.  

---

## Stopping the app

Press `Ctrl + C` in the terminal.

---

## Project structure

```
FQDC Project/
├── app.py              # Streamlit app (Gemini)
├── requirements.txt    # Python dependencies
├── .env.example        # Example env vars (copy to .env and fill in)
└── README.md           # This file
```

---

## Troubleshooting

- **"Could not find 10-K file"**  
  - Check the ticker (e.g. AAPL, MSFT).  
  - Ensure you’re online and have entered your SEC EDGAR email.  

- **"Please enter your Google API Key"**  
  - Enter your Gemini API key in the sidebar (or set `GOOGLE_API_KEY` in `.env`).  

- **"Rate limit exceeded"**  
  - Wait 2–5 minutes and try again.  
  - Enable **Analysis only (1 API call)** in the sidebar to reduce API usage.  

- **Package install errors**  
  - Run `pip install --upgrade pip`, then `pip install -r requirements.txt` again.  

- **Encoding issues**  
  - Set your terminal/IDE encoding to UTF-8.  

---

## License and disclaimer

This project is for learning and portfolio use.  
When using SEC data, comply with [SEC policy](https://www.sec.gov/os/webmaster-faq#code-support).  
When using AI output, comply with Google’s terms of use for the Gemini API.
