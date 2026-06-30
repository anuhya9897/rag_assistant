(function () {
  const LS_KEY = "ab_kb_ui_model";

  const form = document.getElementById("ask-form");
  const statusEl = document.getElementById("status");
  const answerEl = document.getElementById("answer");
  const sourcesEl = document.getElementById("sources");
  const submitBtn = document.getElementById("submit");
  const modelSelect = document.getElementById("model");
  const refreshBtn = document.getElementById("refresh-models");

  let lastDefaultModel = "llama3.1:8b";
  /** @type {Set<string>} */
  let openaiModelTags = new Set();

  function isOpenAiModelTag(tag) {
    const t = (tag || "").trim();
    if (!t) return false;
    if (openaiModelTags.has(t)) return true;
    const lower = t.toLowerCase();
    return (
      lower.startsWith("gpt-") ||
      lower.startsWith("chatgpt-") ||
      lower.startsWith("o1") ||
      lower.startsWith("o3") ||
      lower.startsWith("o4") ||
      lower.startsWith("o5")
    );
  }

  function optionValuesInSelect() {
    return Array.from(modelSelect.options).map((o) => o.value).filter(Boolean);
  }

  function selectContainsValue(value) {
    return optionValuesInSelect().includes(value);
  }

  function apiBase() {
    const params = new URLSearchParams(window.location.search);
    return params.get("api") || "";
  }

  function setStatus(msg, isError) {
    statusEl.textContent = msg || "";
    statusEl.classList.toggle("error", !!isError);
  }

  function formatApiDetail(detail) {
    if (detail == null) return "";
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      return detail
        .map((x) => (typeof x === "object" && x.msg ? x.msg : JSON.stringify(x)))
        .join("\n");
    }
    if (typeof detail === "object") {
      const parts = [
        detail.message,
        detail.detail,
        detail.stderr_tail,
        detail.stdout_tail,
      ].filter(Boolean);
      if (parts.length) return parts.join("\n\n");
      return JSON.stringify(detail);
    }
    return String(detail);
  }

  function readSaved() {
    try {
      const o = JSON.parse(localStorage.getItem(LS_KEY) || "null");
      if (o && typeof o.tag === "string") return o.tag;
    } catch (_) {}
    return null;
  }

  function writeSaved() {
    if (modelSelect.value) {
      localStorage.setItem(LS_KEY, JSON.stringify({ tag: modelSelect.value }));
    }
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
      throw new Error(formatApiDetail(data.detail) || res.statusText || "Request failed");
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
      throw new Error(formatApiDetail(err.detail) || res.statusText);
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

  async function ensureModelTag(tag) {
    setStatus(`Installing ${tag}… this may take several minutes.`, false);
    const res = await fetch(`${apiBase()}/api/models/ensure`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: tag }),
    });
    const payload = await res.json().catch(() => ({}));
    if (!res.ok) {
      const detail = payload.detail;
      throw new Error(formatApiDetail(detail) || "model installation failed");
    }
    return tag;
  }

  function populateModelSelect(providers, flatModels) {
    modelSelect.innerHTML = "";
    if (providers && (providers.ollama?.length || providers.openai?.length)) {
      if (providers.ollama && providers.ollama.length) {
        const og = document.createElement("optgroup");
        og.label = "Ollama";
        for (const m of providers.ollama) {
          const o = document.createElement("option");
          o.value = m;
          o.textContent = m;
          og.appendChild(o);
        }
        modelSelect.appendChild(og);
      }
      if (providers.openai && providers.openai.length) {
        const og = document.createElement("optgroup");
        og.label = "OpenAI / GPT";
        for (const m of providers.openai) {
          const o = document.createElement("option");
          o.value = m;
          o.textContent = m;
          og.appendChild(o);
        }
        modelSelect.appendChild(og);
      }
    } else {
      for (const m of flatModels) {
        const o = document.createElement("option");
        o.value = m;
        o.textContent = m;
        modelSelect.appendChild(o);
      }
    }
  }

  async function loadModels(isRefresh, installTagOnRefresh) {
    refreshBtn.disabled = true;
    if (!isRefresh) {
      modelSelect.disabled = true;
      modelSelect.setAttribute("aria-busy", "true");
    }

    const keepSelection = isRefresh ? modelSelect.value : null;
    let ensuredTag = null;
    const attemptedInstall = isRefresh && !!installTagOnRefresh;

    if (attemptedInstall) {
      try {
        ensuredTag = await ensureModelTag(installTagOnRefresh);
      } catch (err) {
        modelSelect.disabled = false;
        modelSelect.setAttribute("aria-busy", "false");
        refreshBtn.disabled = false;
        setStatus(err.message || "model installation failed", true);
        return;
      }
    }

    let data;
    try {
      const res = await fetch(`${apiBase()}/api/models`);
      const payload = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(payload.detail || res.statusText || "Request failed");
      data = payload;
    } catch (err) {
      modelSelect.innerHTML = "";
      const o = document.createElement("option");
      o.value = lastDefaultModel;
      o.textContent = `${lastDefaultModel} (models list unavailable)`;
      modelSelect.appendChild(o);
      modelSelect.value = lastDefaultModel;
      modelSelect.disabled = false;
      modelSelect.setAttribute("aria-busy", "false");
      setStatus(
        (isRefresh ? "Refresh failed: " : "") +
          (err.message || String(err)) +
          " — start the API & Ollama, then click Refresh.",
        true
      );
      refreshBtn.disabled = false;
      return;
    }

    const models = data.models && data.models.length ? data.models : [data.default || lastDefaultModel];
    lastDefaultModel = data.default || models[0] || lastDefaultModel;

    openaiModelTags = new Set(
      data.providers && data.providers.openai ? data.providers.openai : []
    );

    populateModelSelect(data.providers, models);

    const pick =
      (ensuredTag && selectContainsValue(ensuredTag) && ensuredTag) ||
      (keepSelection && selectContainsValue(keepSelection) && keepSelection) ||
      (readSaved() && selectContainsValue(readSaved()) && readSaved()) ||
      (selectContainsValue(lastDefaultModel) && lastDefaultModel) ||
      optionValuesInSelect()[0] ||
      lastDefaultModel;

    modelSelect.value = pick;
    writeSaved();

    modelSelect.disabled = false;
    modelSelect.setAttribute("aria-busy", "false");
    refreshBtn.disabled = false;

    if (isRefresh) {
      if (attemptedInstall) {
        if (ensuredTag && selectContainsValue(ensuredTag)) {
          setStatus(`Model ${ensuredTag} installed. Models list updated.`, false);
        } else if (ensuredTag) {
          setStatus("Install failed — see message above.", true);
        }
      } else {
        setStatus("Models list updated.", false);
      }
    } else if (!data.ollama_ok && !data.openai_ok) {
      setStatus(
        "Note: Ollama and OpenAI health checks failed — see /api/health.",
        false
      );
    } else if (!data.ollama_ok) {
      setStatus("Note: Ollama health check failed — Ollama models may not work.", false);
    } else if (!data.openai_ok && data.providers?.openai?.length) {
      setStatus("GPT models are listed but Azure/OpenAI is not configured.", true);
    } else {
      setStatus("", false);
    }
  }

  modelSelect.addEventListener("change", writeSaved);

  refreshBtn.addEventListener("click", () => {
    const selected = (modelSelect.value || "").trim();
    const installTag = selected && !isOpenAiModelTag(selected) ? selected : null;
    void loadModels(true, installTag);
  });

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const question = document.getElementById("question").value.trim();
    const k = parseInt(document.getElementById("k").value, 10) || 5;
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

    const model = (modelSelect.value || "").trim() || lastDefaultModel;
    if (!model) {
      setStatus("Select a model from the list.", true);
      return;
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
      writeSaved();
      setStatus("Done.");
    } catch (err) {
      setStatus(err.message || String(err), true);
    } finally {
      submitBtn.disabled = false;
    }
  });

  const reindexForm = document.getElementById("reindex-form");
  const reindexPath = document.getElementById("reindex-path");
  const reindexStatus = document.getElementById("reindex-status");
  const reindexSubmit = document.getElementById("reindex-submit");

  function setReindexStatus(msg, isError) {
    if (!reindexStatus) return;
    reindexStatus.textContent = msg || "";
    reindexStatus.classList.toggle("error", !!isError);
  }

  if (reindexForm) {
    reindexForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const path = reindexPath ? reindexPath.value.trim() : "";
      const body = { source: "local", path: path || null };
      reindexSubmit.disabled = true;
      setReindexStatus("Rebuilding index — this may take several minutes…", false);
      try {
        const res = await fetch(`${apiBase()}/api/reindex`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
          throw new Error(formatApiDetail(data.detail) || res.statusText || "Reindex failed");
        }
        const tail = (data.stdout_tail || "").trim();
        setReindexStatus(
          "Index rebuild finished OK." + (tail ? "\n\n--- log (tail) ---\n" + tail.slice(-4000) : ""),
          false
        );
      } catch (err) {
        setReindexStatus(err.message || String(err), true);
      } finally {
        reindexSubmit.disabled = false;
      }
    });
  }

  try {
    const legacy = JSON.parse(localStorage.getItem(LS_KEY) || "null");
    if (legacy && legacy.custom && !legacy.tag) {
      localStorage.removeItem(LS_KEY);
    }
  } catch (_) {}

  void loadModels(false);
})();

