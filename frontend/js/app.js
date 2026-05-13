(function () {
  const API_BASE = "http://localhost:8000";

  const form = document.getElementById("ask-form");
  const questionInput = document.getElementById("question");
  const submitBtn = document.getElementById("submit-btn");
  const resultSection = document.getElementById("result-section");
  const answerPanel = document.getElementById("answer-panel");
  const sourcesPanel = document.getElementById("sources-panel");
  const answerContent = document.getElementById("answer-content");
  const sourcesContent = document.getElementById("sources-content");
  const errorSection = document.getElementById("error-section");
  const errorMessage = document.getElementById("error-message");
  const loadingEl = document.getElementById("loading");

  document.getElementById("api-url").textContent = API_BASE;

  function showLoading(show) {
    loadingEl.classList.toggle("hidden", !show);
    submitBtn.disabled = show;
  }

  function showError(msg) {
    errorSection.classList.remove("hidden");
    errorMessage.textContent = msg;
    resultSection.classList.add("hidden");
  }

  function hideError() {
    errorSection.classList.add("hidden");
  }

  function setAnswer(text) {
    answerContent.textContent = text || "No answer returned.";
  }

  function setSources(sources) {
    if (!sources || sources.length === 0) {
      sourcesContent.innerHTML = "<p class=\"text-muted\">No sources.</p>";
      return;
    }
    sourcesContent.innerHTML = sources
      .map(
        (s, i) =>
          `<div class="source-card">
            <div class="source-meta">${i + 1}. score=${s.score} · ${escapeHtml(s.source)}</div>
            <div class="source-text">${escapeHtml(truncate(s.text, 400))}</div>
          </div>`
      )
      .join("");
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  function truncate(str, maxLen) {
    if (str.length <= maxLen) return str;
    return str.slice(0, maxLen) + "…";
  }

  function switchTab(tabName) {
    document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === tabName));
    answerPanel.classList.toggle("active", tabName === "answer");
    sourcesPanel.classList.toggle("active", tabName === "sources");
  }

  document.querySelectorAll(".tab").forEach((btn) => {
    btn.addEventListener("click", () => switchTab(btn.dataset.tab));
  });

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    hideError();
    const question = questionInput.value.trim();
    if (!question) return;

    const k = parseInt(document.getElementById("k").value, 10) || 5;
    const model = document.getElementById("model").value.trim() || "llama3.1:8b";

    showLoading(true);
    setAnswer("");
    setSources([]);
    resultSection.classList.remove("hidden");
    switchTab("answer");

    try {
      const res = await fetch(`${API_BASE}/api/ask/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, k, model }),
      });

      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        showError(data.detail || data.message || `Request failed: ${res.status}`);
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let answerSoFar = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop() || "";
        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              const data = JSON.parse(line.slice(6));
              if (data.event === "sources") {
                setSources(data.sources || []);
                showLoading(false);
              } else if (data.event === "token") {
                answerSoFar += data.content || "";
                setAnswer(answerSoFar);
                showLoading(false);
              } else if (data.event === "done") {
                setAnswer(data.answer || answerSoFar);
                showLoading(false);
              } else if (data.event === "error") {
                showError(data.detail || "Stream error");
                showLoading(false);
                return;
              }
            } catch (_) {}
          }
        }
      }
      showLoading(false);
    } catch (err) {
      showError(err.message || "Network error. Is the API running at " + API_BASE + "?");
      showLoading(false);
    }
  });
})();
