// JobSidebar.jsx — local history list, bucketed, with selection + remove.
const CAP_DOT_CLASS = {
  MOCK: "ds-cap-mock",
  RAG: "ds-cap-rag",
  OCR: "ds-cap-ocr",
  MULTIMODAL: "ds-cap-multimodal",
};

function relTime(iso) {
  const ms = Date.now() - new Date(iso).getTime();
  if (ms < 60_000) return "방금";
  if (ms < 3600_000) return `${Math.floor(ms / 60_000)}분 전`;
  if (ms < 86400_000) return `${Math.floor(ms / 3600_000)}시간 전`;
  return new Date(iso).toLocaleDateString();
}

function shortId(id, n = 8) {
  return (id || "").slice(0, n);
}

const JobSidebar = ({ history, statusByJob, activeJobId, onSelect, onRemove, onClear }) => {
  return (
    <aside className="ds-sidebar">
      <div className="ds-sidebar-head">
        <div>
          <div className="ds-sidebar-title">최근</div>
          <div className="ds-sidebar-sub">로컬에 {history.length}건 저장됨</div>
        </div>
        {history.length > 0 && (
          <button className="ds-btn ds-btn-sm ds-btn-ghost" onClick={onClear}>
            <i data-lucide="trash-2"></i>
            비우기
          </button>
        )}
      </div>

      {history.length === 0 ? (
        <div className="ds-sidebar-empty">
          <div className="ds-empty-icon"><i data-lucide="inbox"></i></div>
          <div className="ds-empty-title">아직 작업이 없습니다</div>
          <div className="ds-empty-body">작업을 제출하면 여기에 표시됩니다.</div>
        </div>
      ) : (
        <div className="ds-sidebar-list">
          <div className="ds-bucket">Today</div>
          <ul className="ds-rows">
            {history.map((h) => {
              const status = statusByJob[h.jobId] || "PENDING";
              const active = h.jobId === activeJobId;
              return (
                <li key={h.jobId}>
                  <button
                    type="button"
                    onClick={() => onSelect(h.jobId)}
                    className={`ds-row ${active ? "is-active" : ""}`}
                  >
                    {active && <span className="ds-row-rail"></span>}
                    <div className="ds-row-head">
                      <span className="ds-row-cap">
                        <span className={`ds-row-cap-dot ${CAP_DOT_CLASS[h.capability]}`}></span>
                        {h.capability}
                      </span>
                      <StatusBadge status={status} size="sm" withDot />
                    </div>
                    {h.preview && <div className="ds-row-preview">{h.preview}</div>}
                    <div className="ds-row-meta">
                      <span>{shortId(h.jobId)}</span>
                      <span>{relTime(h.submittedAt)}</span>
                    </div>
                    <span
                      role="button"
                      className="ds-row-remove"
                      onClick={(e) => {
                        e.stopPropagation();
                        onRemove(h.jobId);
                      }}
                      aria-label="히스토리에서 제거"
                    >
                      <i data-lucide="trash-2"></i>
                    </span>
                  </button>
                </li>
              );
            })}
          </ul>
        </div>
      )}
    </aside>
  );
};

window.JobSidebar = JobSidebar;
