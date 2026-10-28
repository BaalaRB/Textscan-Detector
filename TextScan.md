# TextScan

A lightweight front end with a Python API for experimental writing-pattern measurements. The Python HTTP server serves the page; one spawned background worker performs all model inference on CPU and streams results back sentence by sentence.

## Layout

```
server.py
index.html        (lowercase; Linux file systems are case-sensitive)
styles.css
app.js
backend/detector_worker.py
requirements.txt
```

## Run locally

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
PORT=3000 .venv/bin/python server.py
```

Optional environment variables: `HOST` (default `0.0.0.0`), `PORT` (default `3000`), `TEXTSCAN_THREADS` (CPU threads for inference, default `min(4, cpu_count)`).

The worker loads and warms up `distilbert/distilgpt2` as soon as the server starts, so the first scan does not pay the model-load cost. The English-focused model is not trained to identify ChatGPT, Claude, or a specific author. The API requires at least 20 words, accepts up to 12,000 characters, and limits a passage to 120 sentences and two in-flight requests.

## API

`POST /api/scan` accepts JSON `{"text":"..."}` and returns a chunked Server-Sent Events stream. A `start` event announces the model and sentence count, each `sentence` event arrives as soon as that sentence is scored, and a `complete` event carries passage-level metrics and the experimental pattern score. An `error` event explains a failed or cancelled scan. If the client disconnects, the worker stops at the next sentence boundary.

`GET /api/health` reports worker status and whether the model is warm (`model_ready`) without loading anything.

## Performance and hardening notes

- Static files are an explicit allowlist (`/`, `/styles.css`, `/app.js`) preloaded into memory, gzip-compressed once, and served with ETags. Source files such as `server.py` are never exposed.
- `styles.css` and `app.js` are referenced with a content-hash query string (`?v=...`, filled in by the server at startup) and cached for a year as immutable. `index.html` is revalidated with an ETag on every load.
- Responses carry a strict Content-Security-Policy and related security headers. If you ever add inline scripts or styles, the policy must be relaxed.
- A watchdog restarts the model worker if it dies, and in-flight requests receive an error instead of hanging. Idle or stalled sockets are dropped after 30 seconds.
- The front end batches sentence rows into one DOM update per animation frame and caches element lookups.

## Interpretation and data handling

Orange marks higher predictability according to DistilGPT-2; green marks a sentence whose length varies from the rest of the passage. These are heuristic pattern cues, not human/AI labels. The worker also returns sentence-level perplexity, overall sentence-length variation, and a 0–100 score. The score is an arbitrary, uncalibrated combination of two patterns; it is not a probability that text was written by AI and does not establish authorship. Short passages are particularly unstable. Do not use the score or colors as proof or for consequential decisions.

Submitted text is held in memory for the request and is not written to application files or included in server request logs. Text is sent to this backend for analysis, so do not submit sensitive content.
