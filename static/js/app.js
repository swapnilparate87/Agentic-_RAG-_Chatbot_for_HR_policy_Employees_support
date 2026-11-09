const $ = (id) => document.getElementById(id);
const chat = $("chat");
const form = $("chatForm");
const question = $("question");
const trace = $("trace");
const sourceUsed = $("sourceUsed");
const modal = $("uploadModal");

const SOURCE_LABELS = {
  knowledge_base: "Knowledge base",
  web_search: "Web search",
  direct: "Direct reply",
  insufficient_evidence: "Insufficient evidence",
};

const escapeHtml = (s = "") =>
  s.replace(/[&<>'"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[c]);

function addMessage(role, text, { source = "", citations = [] } = {}) {
  const sources = citations
    .map((c) => (c.url ? `<a href="${escapeHtml(c.url)}" target="_blank" rel="noopener">${escapeHtml(c.title)}</a>` : escapeHtml(c.title)))
    .join("<br>");

  const el = document.createElement("div");
  el.className = `message ${role}`;
  el.innerHTML = `
    <div class="avatar">${role === "user" ? "You" : "HR"}</div>
    <div class="bubble">
      ${escapeHtml(text).replace(/\n/g, "<br>")}
      ${source ? `<div class="meta">Source: ${escapeHtml(SOURCE_LABELS[source] || source)}</div>` : ""}
      ${sources ? `<div class="citations"><strong>References</strong><br>${sources}</div>` : ""}
    </div>`;
  chat.appendChild(el);
  chat.scrollTop = chat.scrollHeight;
}

function renderTrace(items) {
  trace.innerHTML = items.length
    ? items.map((x) => `<div class="trace-item">${escapeHtml(x)}</div>`).join("")
    : '<div class="empty">No trace.</div>';
}

async function askAgent(q) {
  const btn = form.querySelector("button");
  addMessage("user", q);
  question.value = "";
  renderTrace(["Running agent…"]);
  sourceUsed.textContent = "…";
  btn.disabled = true;

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Request failed");
    addMessage("assistant", data.answer, { source: data.source_used, citations: data.citations });
    renderTrace(data.trace);
    sourceUsed.textContent = SOURCE_LABELS[data.source_used] || data.source_used;
  } catch (e) {
    addMessage("assistant", `Error: ${e.message}`);
    renderTrace(["Request failed"]);
    sourceUsed.textContent = "Error";
  } finally {
    btn.disabled = false;
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const q = question.value.trim();
  if (q) askAgent(q);
});

question.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    form.requestSubmit();
  }
});

document.querySelectorAll(".example").forEach((b) => b.addEventListener("click", () => askAgent(b.textContent.trim())));

$("openUpload").onclick = () => modal.classList.remove("hidden");
$("closeUpload").onclick = () => modal.classList.add("hidden");

$("uploadBtn").onclick = async () => {
  const file = $("fileInput").files[0];
  const status = $("uploadStatus");
  if (!file) return (status.textContent = "Choose a file first.");

  status.textContent = "Indexing document…";
  const body = new FormData();
  body.append("file", file);
  try {
    const res = await fetch("/api/ingest", { method: "POST", headers: { "X-Admin-Key": $("adminKey").value }, body });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Upload failed");
    status.textContent = `Indexed ${data.file}: ${data.chunks} chunks.`;
  } catch (e) {
    status.textContent = `Error: ${e.message}`;
  }
};
