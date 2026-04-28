// ConsoleScreen.jsx — top-level layout. Orchestrates fake-job lifecycle for the demo.
const STORAGE_KEY = "console.demo.v1";

function uid(n = 8) {
  const s = "0123456789abcdef";
  let out = "";
  for (let i = 0; i < n; i++) out += s[Math.floor(Math.random() * s.length)];
  return out;
}

// Demo result fabrication — mirrors core-api shapes from the repo.
function fabricateResult(capability, payload) {
  if (capability === "MOCK") {
    return { result: { echo: payload.prompt, echoedAt: new Date().toISOString() } };
  }
  if (capability === "RAG") {
    const hits = [
      { doc: "anime-005#overview", score: 0.845, answer: "The Harbor Cats" },
      { doc: "anime-002#character",  score: 0.612, answer: "Old fisherman Hideo" },
      { doc: "anime-008#scene-12",   score: 0.471, answer: "Stray cats at dawn" },
      { doc: "anime-001#synopsis",   score: 0.355, answer: "Coastal village arc" },
      { doc: "anime-007#review",     score: 0.301, answer: "Critics praise the warmth" },
    ];
    return { result: { topHit: hits[0], hits, query: payload.question, topK: payload.topK } };
  }
  if (capability === "OCR") {
    const text = "INVOICE\nDate: 2025-03-14\nVendor: Harbor Supplies Co.\nLine items:\n  · Net mending kit  ₩42,000\n  · Tackle box (med) ₩18,500\nTotal: ₩60,500";
    return { result: { text, language: "ko-en", charCount: text.length, file: payload.file?.name } };
  }
  if (capability === "MULTIMODAL") {
    return {
      result: {
        answer: "문서 첫 줄에는 \"INVOICE\"라는 제목과 발행일(2025-03-14)이 있습니다.",
        evidence: [
          { span: "p1:line1", note: "이미지 상단 헤더" },
          { span: "p1:line2", note: "OCR 추출 메타" },
        ],
        question: payload.question,
        file: payload.file?.name,
      },
    };
  }
  return { result: {} };
}

function loadStore() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    return JSON.parse(raw);
  } catch { return null; }
}

function saveStore(store) {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(store)); } catch {}
}

const SEED_HISTORY = [
  {
    jobId: "7e1b4d", capability: "MULTIMODAL",
    preview: "demo_sample.pdf — What metrics are shown?",
    submittedAt: new Date(Date.now() - 14 * 60 * 1000).toISOString(),
    status: "SUCCEEDED", attempt: 1,
    events: { PENDING: 1, QUEUED: 120, RUNNING: 1500, TERMINAL: 4100, terminalKind: "SUCCEEDED" },
    result: { answer: "이 페이지는 매출(₩60,500)과 라인 아이템 2건을 보여줍니다.", evidence: [{ span: "p1:line5", note: "Total" }] },
  },
  {
    jobId: "c52a18", capability: "OCR",
    preview: "receipt-2025-03-14.png",
    submittedAt: new Date(Date.now() - 32 * 60 * 1000).toISOString(),
    status: "SUCCEEDED", attempt: 1,
    events: { PENDING: 1, QUEUED: 95, RUNNING: 800, TERMINAL: 2600, terminalKind: "SUCCEEDED" },
    result: { text: "Receipt — 2025-03-14\nTotal ₩60,500", language: "ko-en", charCount: 32 },
  },
  {
    jobId: "4490ee", capability: "RAG",
    preview: "best episode of The Harbor Cats?",
    submittedAt: new Date(Date.now() - 51 * 60 * 1000).toISOString(),
    status: "FAILED", attempt: 2,
    events: { PENDING: 1, QUEUED: 200, RUNNING: 900, TERMINAL: 3200, terminalKind: "FAILED" },
    error: "Vector store unreachable: connection refused",
  },
];

const ConsoleScreen = () => {
  const [dark, setDark] = React.useState(false);
  const [online, setOnline] = React.useState(null);
  const [baseUrl, setBaseUrl] = React.useState("http://localhost:8080");
  const [capability, setCapability] = React.useState("RAG");
  const [busy, setBusy] = React.useState(false);

  const [store, setStore] = React.useState(() => loadStore() || {
    history: SEED_HISTORY,
    activeJobId: SEED_HISTORY[0].jobId,
  });
  React.useEffect(() => saveStore(store), [store]);

  // Fake online ping.
  React.useEffect(() => {
    setOnline(null);
    const t = setTimeout(() => setOnline(true), 700);
    return () => clearTimeout(t);
  }, []);

  React.useEffect(() => {
    if (dark) document.documentElement.setAttribute("data-theme", "dark");
    else document.documentElement.removeAttribute("data-theme");
  }, [dark]);

  // Re-create lucide icons on every render — cheap + reliable for a demo.
  React.useEffect(() => {
    if (window.lucide) window.lucide.createIcons();
  });

  const activeJob = store.history.find((j) => j.jobId === store.activeJobId);
  const statusByJob = Object.fromEntries(store.history.map((j) => [j.jobId, j.status]));

  const setActive = (id) => setStore((s) => ({ ...s, activeJobId: id }));

  const removeJob = (id) => setStore((s) => {
    const next = s.history.filter((j) => j.jobId !== id);
    return { history: next, activeJobId: s.activeJobId === id ? (next[0]?.jobId || null) : s.activeJobId };
  });

  const clearAll = () => setStore({ history: [], activeJobId: null });

  const buildPreview = (capability, payload) => {
    if (capability === "MOCK") return payload.prompt?.slice(0, 80);
    if (capability === "RAG") return payload.question?.slice(0, 80);
    if (capability === "OCR") return payload.file?.name;
    if (capability === "MULTIMODAL") return `${payload.file?.name} — ${payload.question?.slice(0, 50)}`;
    return "";
  };

  const submitJob = (payload) => {
    const jobId = uid();
    const submittedAt = new Date().toISOString();
    const start = Date.now();
    const newJob = {
      jobId,
      capability: payload.capability,
      preview: buildPreview(payload.capability, payload),
      submittedAt,
      status: "PENDING",
      attempt: 1,
      events: { PENDING: 0 },
      payload,
    };
    setStore((s) => ({ history: [newJob, ...s.history], activeJobId: jobId }));
    setBusy(true);

    // Simulate the queue lifecycle.
    setTimeout(() => updateJob(jobId, { status: "QUEUED", events: { QUEUED: Date.now() - start } }), 180);
    setTimeout(() => updateJob(jobId, { status: "RUNNING", events: { RUNNING: Date.now() - start } }), 1100);
    const totalMs = 2400 + Math.random() * 1200;
    setTimeout(() => {
      const out = fabricateResult(payload.capability, payload);
      updateJob(jobId, {
        status: "SUCCEEDED",
        events: { TERMINAL: Date.now() - start, terminalKind: "SUCCEEDED" },
        result: out.result,
      });
      setBusy(false);
    }, totalMs);
  };

  const updateJob = (jobId, patch) => {
    setStore((s) => ({
      ...s,
      history: s.history.map((j) =>
        j.jobId !== jobId ? j : { ...j, ...patch, events: { ...(j.events || {}), ...(patch.events || {}) } }
      ),
    }));
  };

  return (
    <div className="ds-app">
      <Header online={online} dark={dark} onToggleTheme={() => setDark((d) => !d)} />
      <main className="ds-main">
        <JobSidebar
          history={store.history}
          statusByJob={statusByJob}
          activeJobId={store.activeJobId}
          onSelect={setActive}
          onRemove={removeJob}
          onClear={clearAll}
        />
        <div className="ds-stage">
          <SubmitForm
            capability={capability}
            setCapability={setCapability}
            onSubmit={submitJob}
            busy={busy}
            baseUrl={baseUrl}
            setBaseUrl={setBaseUrl}
          />
          <ResultViewer
            job={activeJob?.jobId}
            status={activeJob?.status}
            capability={activeJob?.capability}
            result={activeJob?.result}
            error={activeJob?.error}
            events={activeJob?.events || {}}
            attempt={activeJob?.attempt || 1}
          />
        </div>
      </main>
    </div>
  );
};

window.ConsoleScreen = ConsoleScreen;
