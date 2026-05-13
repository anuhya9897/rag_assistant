(function () {
  const form = document.getElementById("ask-form");
  const statusEl = document.getElementById("status");
  const answerEl = document.getElementById("answer");
  const sourcesEl = document.getElementById("sources");
  const submitBtn = document.getElementById("submit");

  function apiBase() {
    const params = new URLSearchParams(window.location.search);
    return params.get("api") || "";
  }

  function setStatus(msg, isError) {
    statusEl.textContent = msg || "";
    statusEl.classList.toggle("error", !!isError);
  }

  function renderSources(list) {
    sourcesEl.innerHTML = "";
    if (!list || !list.length) {
      sourcesEl.textContent = "No sources returned.";
      return;
    }
    list.forEach((s, i) => {
      const card = document.createElement("div");
      card.className = "source-card";
      const meta = document.createElement("div");
      meta.className = "meta";
      meta.textContent = `[${i + 1}] score=${s.score} — ${s.source} (chunk ${s.chunk_id})`;
      const pre = document.createElement("pre");
      pre.textContent = s.text || "";
      card.appendChild(meta);
      card.appendChild(pre);
      sourcesEl.appendChild(card);
    });
  }

  async function askNonStream(body) {
    const res = await fetch(`${apiBase()}/api/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(data.detail || res.statusText || "Request failed");
    }
    renderSources(data.sources);
    answerEl.hidden = false;
    answerEl.textContent = data.answer || "";
  }

  async function askStream(body) {
    answerEl.hidden = false;
    answerEl.textContent = "";
    const res = await fetch(`${apiBase()}/api/ask/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || res.statusText);
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop() || "";
      for (const block of parts) {
        const line = block.trim().split("\n").find((l) => l.startsWith("data:"));
        if (!line) continue;
        const json = line.replace(/^data:\s*/, "");
        let evt;
        try {
          evt = JSON.parse(json);
        } catch {
          continue;
        }
        if (evt.event === "sources") {
          renderSources(evt.sources);
        } else if (evt.event === "token") {
          answerEl.textContent += evt.content || "";
        } else if (evt.event === "done") {
          answerEl.textContent = evt.answer || answerEl.textContent;
        } else if (evt.event === "error") {
          throw new Error(evt.detail || "Stream error");
        }
      }
    }
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const question = document.getElementById("question").value.trim();
    const k = parseInt(document.getElementById("k").value, 10) || 5;
    const model = document.getElementById("model").value.trim() || "llama3.1:8b";
    const useStream = document.getElementById("stream").checked;
    const simRaw = document.getElementById("similarity_min").value.trim();
    let similarity_min = null;
    if (simRaw !== "") {
      const v = Number.parseFloat(simRaw);
      if (!Number.isFinite(v) || v < -1 || v > 1) {
        setStatus("Min similarity must be a number between -1 and 1.", true);
        return;
      }
      similarity_min = v;
    }
    const body = { question, k, model, similarity_min };

    submitBtn.disabled = true;
    setStatus("Working…");
    answerEl.hidden = true;
    sourcesEl.innerHTML = "";

    try {
      if (useStream) {
        await askStream(body);
      } else {
        await askNonStream(body);
      }
      setStatus("Done.");
    } catch (err) {
      setStatus(err.message || String(err), true);
    } finally {
      submitBtn.disabled = false;
    }
  });
})();
