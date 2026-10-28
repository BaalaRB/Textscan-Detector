"use strict";

const MAX_CHARS = 12000;
const EXAMPLES = {
  article: "Urban trees do more than make a city feel welcoming. They provide shade on hot days, absorb rainwater, and create habitats for birds and insects. Even a small row of trees along a busy street can make the walk more comfortable and help a neighborhood feel more connected to nature.",
  formal: "Community libraries continue to play an important role in public life. In addition to providing access to books and research materials, they offer quiet places to study, practical digital resources, and welcoming spaces for people of all ages. Their value lies not only in the services they provide, but also in the opportunities they create for learning and connection.",
  casual: "I took the longer path home today and found a little coffee shop tucked between two bookstores. It was raining, so I stayed for a while, read a few pages, and watched everyone hurry past the window. Sometimes a slower afternoon is exactly what you need.",
};

// Look every element up once instead of on each event.
const $ = (id) => document.getElementById(id);
const ui = {
  input: $("writing-input"),
  counter: $("character-count"),
  button: $("scan-button"),
  status: $("scan-status"),
  panel: $("analysis-results"),
  list: $("sentence-list"),
  progress: $("stream-progress"),
  score: $("pattern-score"),
  perplexity: $("mean-perplexity"),
  variation: $("length-variation"),
  sentences: $("sentence-count"),
  words: $("word-count"),
  sampleNote: $("sample-note"),
  modelNote: $("model-note"),
  examples: $("examples"),
};
const label = ui.button.querySelector("span");
const numberFormat = new Intl.NumberFormat();

let controller = null;
let expected = 0;
let received = 0;
let pending = [];
let frame = 0;
let countQueued = false;

/* ---------- character counter (one DOM write per frame) ---------- */

function updateCount() {
  countQueued = false;
  // maxlength counts UTF-16 units, so value.length matches it without copying the string.
  ui.counter.textContent = `${numberFormat.format(ui.input.value.length)} / ${numberFormat.format(MAX_CHARS)} characters`;
}

function queueCount() {
  if (countQueued) return;
  countQueued = true;
  requestAnimationFrame(updateCount);
}

/* ---------- UI helpers ---------- */

function setStatus(message, isError = false) {
  ui.status.textContent = message;
  ui.status.classList.toggle("is-error", isError);
}

function setBusy(busy) {
  ui.button.disabled = busy;
  if (busy) ui.button.setAttribute("aria-busy", "true");
  else ui.button.removeAttribute("aria-busy");
  label.textContent = busy ? "Analyzing…" : "Scan Text";
}

function resetResults() {
  cancelAnimationFrame(frame);
  frame = 0;
  pending = [];
  expected = 0;
  received = 0;
  ui.list.replaceChildren();
  ui.score.textContent = ui.perplexity.textContent = ui.variation.textContent = "—";
  ui.sentences.textContent = ui.words.textContent = "—";
  ui.sampleNote.textContent = ui.modelNote.textContent = "";
  ui.progress.textContent = "Waiting for the first sentence result…";
}

function renderSentence(s) {
  const tone = s.highlight === "orange" || s.highlight === "green" ? s.highlight : "neutral";
  const item = document.createElement("li");
  item.className = `sentence-item signal-${tone}`;

  const top = document.createElement("div");
  top.className = "sentence-topline";
  const number = document.createElement("strong");
  number.textContent = `Sentence ${s.sentence}`;
  const tag = document.createElement("span");
  tag.className = `sentence-tag tag-${tone}`;
  tag.textContent = s.highlight_label;
  top.append(number, tag);

  const excerpt = document.createElement("p");
  excerpt.className = "sentence-excerpt";
  excerpt.textContent = `“${s.snippet}${s.snippet_truncated ? "…" : ""}”`;

  const metric = document.createElement("span");
  metric.className = "sentence-metric";
  const perplexity = s.perplexity === null ? "Not enough tokens to score" : `Model perplexity ${s.perplexity}`;
  metric.textContent = `${s.words} words · ${perplexity}${s.truncated ? " · model input truncated" : ""}`;

  item.append(top, excerpt, metric);
  return item;
}

/* Batch sentence rows so a burst of events costs one layout pass. */
function flush() {
  frame = 0;
  if (!pending.length) return;
  const fragment = document.createDocumentFragment();
  for (const s of pending) fragment.append(renderSentence(s));
  received += pending.length;
  pending = [];
  ui.list.append(fragment);
  ui.sentences.textContent = `${received} / ${expected}`;
  ui.progress.textContent = `Received ${received} of ${expected} sentence results. The worker is continuing.`;
}

function queueSentence(s) {
  pending.push(s);
  if (!frame) frame = requestAnimationFrame(flush);
}

/* ---------- stream handling ---------- */

function handleEvent(e) {
  switch (e.type) {
    case "start":
      expected = e.sentence_count;
      ui.sentences.textContent = `0 / ${expected}`;
      ui.words.textContent = e.word_count;
      ui.sampleNote.textContent = e.sample_note;
      ui.modelNote.textContent = `Scoring with ${e.model}, an English-focused language model. It is not trained to identify ChatGPT, Claude, or a specific author.`;
      ui.progress.textContent = `Prepared ${expected} sentences. Waiting for the model's first result…`;
      ui.panel.hidden = false;
      break;
    case "sentence":
      queueSentence(e);
      break;
    case "complete":
      cancelAnimationFrame(frame);
      flush();
      ui.score.textContent = e.pattern_score;
      ui.perplexity.textContent = e.mean_sentence_perplexity;
      ui.variation.textContent = e.sentence_length_variation;
      ui.sentences.textContent = e.sentence_count;
      ui.words.textContent = e.word_count;
      ui.sampleNote.textContent = e.sample_note;
      ui.modelNote.textContent = `Model: ${e.model}. Perplexity measures predictability; the sentence colors are heuristic cues, not authorship classifications.`;
      ui.progress.textContent = `All ${e.sentence_count} sentence results received.`;
      break;
    case "error":
      throw new Error(e.message || "The analysis could not be completed.");
  }
}

/** Parse one SSE block; returns true when it was the terminal "complete" event. */
function processBlock(block) {
  if (!block || block.startsWith(":")) return false;
  let data = "";
  for (const line of block.split("\n")) {
    if (line.startsWith("data:")) data += (data ? "\n" : "") + line.slice(5).trimStart();
  }
  if (!data) return false;
  const event = JSON.parse(data);
  handleEvent(event);
  return event.type === "complete";
}

async function consumeStream(response) {
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.error || "The analysis could not be started.");
  }
  if (!response.body) throw new Error("This browser could not open the streamed response.");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let completed = false;

  for (;;) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value, { stream: !done });
    const blocks = buffer.split("\n\n");
    buffer = done ? "" : blocks.pop();
    for (const block of blocks) if (processBlock(block)) completed = true;
    if (done) break;
  }
  if (!completed) throw new Error("The stream ended before the final summary arrived.");
}

/* ---------- events ---------- */

ui.input.addEventListener("input", queueCount);

ui.examples.addEventListener("click", (event) => {
  const button = event.target.closest("[data-example]");
  if (!button) return;
  ui.input.value = EXAMPLES[button.dataset.example] || "";
  updateCount();
  ui.input.focus();
});

ui.button.addEventListener("click", async () => {
  const text = ui.input.value.trim();
  if (!text) {
    setStatus("Add some text before scanning.", true);
    ui.input.focus();
    return;
  }

  controller = new AbortController();
  const { signal } = controller;
  resetResults();
  ui.panel.hidden = true;
  setBusy(true);
  setStatus("The backend is analyzing the text. Sentence results will appear as they are ready.");

  try {
    const response = await fetch("/api/scan", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify({ text }),
      signal,
    });
    await consumeStream(response);
    setStatus("Analysis complete. Treat the experimental patterns cautiously.");
  } catch (error) {
    if (!signal.aborted) {
      setStatus(error instanceof Error ? error.message : "A network error interrupted the scan.", true);
    }
  } finally {
    controller = null;
    setBusy(false);
  }
});

// Closing the tab drops the connection, which tells the server to stop the worker early.
addEventListener("pagehide", () => controller?.abort());

updateCount();
