// ResultViewer.jsx — capability-aware result panel: header + tabs (결과/아티팩트/원본).
function copyText(s) {
  try { navigator.clipboard.writeText(s); } catch {}
}

const ScoreBar = ({ score }) => (
  <div className="ds-score">
    <div className="ds-score-track">
      <div className="ds-score-fill" style={{ width: `${Math.max(0, Math.min(1, score)) * 100}%` }}></div>
    </div>
    <div className="ds-score-num">{score.toFixed(3)}</div>
  </div>
);

const RagResult = ({ result }) => {
  const top = result.topHit;
  const hits = result.hits || [];
  return (
    <div className="ds-rv-body">
      {top && (
        <div className="ds-card-soft ds-rv-top">
          <div className="ds-rv-top-eyebrow">상위 히트</div>
          <div className="ds-rv-top-answer">{top.answer}</div>
          <div className="ds-rv-top-meta">
            <span className="ds-mono">{top.doc}</span>
            <ScoreBar score={top.score} />
          </div>
        </div>
      )}
      {hits.length > 0 && (
        <div className="ds-rv-hits">
          <div className="ds-rv-hits-head">
            <span>검색된 문서</span>
            <span className="ds-mono">{hits.length}건</span>
          </div>
          <ul>
            {hits.slice(0, 8).map((h, i) => (
              <li key={i} className="ds-rv-hit">
                <span className="ds-rv-hit-rank">{i + 1}</span>
                <span className="ds-mono ds-rv-hit-doc">{h.doc}</span>
                <ScoreBar score={h.score} />
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

const OcrResult = ({ result }) => (
  <div className="ds-rv-body">
    <div className="ds-card-soft">
      <div className="ds-rv-eyebrow">추출된 텍스트</div>
      <div className="ds-rv-ocr-text">{result.text}</div>
      <div className="ds-rv-ocr-meta">
        <span>language</span><span className="ds-mono">{result.language}</span>
        <span>chars</span><span className="ds-mono">{result.charCount ?? (result.text || "").length}</span>
      </div>
    </div>
  </div>
);

const MultimodalResult = ({ result }) => (
  <div className="ds-rv-body">
    <div className="ds-card-soft">
      <div className="ds-rv-eyebrow">답변</div>
      <div className="ds-rv-mm-answer">{result.answer}</div>
    </div>
    {result.evidence?.length > 0 && (
      <div className="ds-rv-evidence">
        <div className="ds-rv-eyebrow">근거</div>
        <ul>
          {result.evidence.map((e, i) => (
            <li key={i}><span className="ds-mono">{e.span}</span> · {e.note}</li>
          ))}
        </ul>
      </div>
    )}
  </div>
);

const MockResult = ({ result }) => (
  <div className="ds-rv-body">
    <div className="ds-card-soft">
      <div className="ds-rv-eyebrow">에코</div>
      <div className="ds-rv-mock">{result.echo}</div>
    </div>
  </div>
);

const ResultViewer = ({ job, status, capability, result, error, events, attempt }) => {
  const [tab, setTab] = React.useState("result");

  if (!job) {
    return (
      <section className="ds-card ds-rv-empty">
        <div className="ds-empty-icon"><i data-lucide="sparkles"></i></div>
        <div className="ds-empty-title">선택된 작업이 없습니다</div>
        <div className="ds-empty-body">왼쪽 폼에서 새 작업을 제출하거나 사이드바에서 기존 작업을 선택하세요.</div>
      </section>
    );
  }

  const terminal = ["SUCCEEDED", "FAILED", "CANCELED"].includes(status);
  const raw = JSON.stringify({ jobId: job, capability, status, attempt, events, result, error }, null, 2);

  return (
    <section className="ds-card ds-rv">
      <header className="ds-rv-head">
        <div className="ds-rv-head-l">
          <div className="ds-rv-jobid">
            <i data-lucide="hash"></i>
            <span className="ds-mono">{job}</span>
            <button className="ds-btn-mini" onClick={() => copyText(job)} aria-label="복사">
              <i data-lucide="copy"></i>
            </button>
          </div>
          <div className="ds-rv-cap">
            <span className={`ds-cap-pill ds-cap-pill-${capability?.toLowerCase()}`}>{capability}</span>
          </div>
        </div>
        <StatusBadge status={status} withDot />
      </header>

      <div className="ds-rv-timeline">
        <JobTimeline status={status} attempt={attempt} events={events} />
      </div>

      <nav className="ds-tabs">
        <button className={`ds-tab ${tab === "result" ? "is-active" : ""}`} onClick={() => setTab("result")}>결과</button>
        <button className={`ds-tab ${tab === "raw" ? "is-active" : ""}`} onClick={() => setTab("raw")}>원본</button>
      </nav>

      <div className="ds-rv-content">
        {tab === "result" && (
          <>
            {!terminal && (
              <div className="ds-rv-loading">
                <i data-lucide="loader-2" className="ds-icon-spin"></i>
                <span>워커가 작업을 처리하고 있습니다…</span>
              </div>
            )}
            {status === "FAILED" && (
              <div className="ds-rv-error">
                <i data-lucide="circle-alert"></i>
                <div>
                  <div className="ds-rv-error-title">작업 실패</div>
                  <div className="ds-rv-error-msg">{error || "워커가 예외를 반환했습니다."}</div>
                </div>
              </div>
            )}
            {status === "CANCELED" && (
              <div className="ds-rv-error ds-rv-error-muted">
                <i data-lucide="circle-slash"></i>
                <div>
                  <div className="ds-rv-error-title">작업 취소됨</div>
                  <div className="ds-rv-error-msg">사용자 또는 시스템에 의해 취소되었습니다.</div>
                </div>
              </div>
            )}
            {terminal && status === "SUCCEEDED" && capability === "RAG" && <RagResult result={result} />}
            {terminal && status === "SUCCEEDED" && capability === "OCR" && <OcrResult result={result} />}
            {terminal && status === "SUCCEEDED" && capability === "MULTIMODAL" && <MultimodalResult result={result} />}
            {terminal && status === "SUCCEEDED" && capability === "MOCK" && <MockResult result={result} />}
          </>
        )}

        {tab === "raw" && (
          <pre className="ds-rv-raw">{raw}</pre>
        )}
      </div>
    </section>
  );
};

window.ResultViewer = ResultViewer;
