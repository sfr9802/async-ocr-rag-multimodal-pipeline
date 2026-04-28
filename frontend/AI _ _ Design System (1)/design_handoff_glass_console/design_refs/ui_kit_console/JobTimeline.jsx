// JobTimeline.jsx — pill-strip of lifecycle events with duration connectors.
function fmtDuration(ms) {
  if (ms == null) return null;
  if (ms < 1000) return `${Math.round(ms)}ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`;
  return `${(ms / 60_000).toFixed(1)}m`;
}

const TIMELINE_ORDER = ["PENDING", "QUEUED", "RUNNING", "TERMINAL"];

const JobTimeline = ({ status, attempt = 1, events = {} }) => {
  // events: { PENDING: ts, QUEUED: ts, RUNNING: ts, TERMINAL: ts, terminalKind: "SUCCEEDED"|"FAILED"|"CANCELED" }
  const reached = (k) => events[k] != null;
  const isCurrent = (k) => {
    if (k === "TERMINAL") return ["SUCCEEDED", "FAILED", "CANCELED"].includes(status);
    return status === k;
  };

  const renderDot = (k) => {
    if (k === "TERMINAL") {
      const tone = events.terminalKind === "SUCCEEDED" ? "success"
                : events.terminalKind === "FAILED" ? "destructive"
                : "muted";
      return <span className={`ds-tl-dot ds-tl-dot-${tone}`}></span>;
    }
    if (isCurrent(k)) {
      const tone = k === "RUNNING" ? "warning" : "muted";
      return <span className={`ds-tl-dot ds-tl-dot-${tone} ${k === "RUNNING" ? "animate-pulse-soft" : ""}`}></span>;
    }
    return <span className={`ds-tl-dot ds-tl-dot-${reached(k) ? "filled" : "empty"}`}></span>;
  };

  const stepLabel = (k) => {
    if (k === "PENDING") return "대기";
    if (k === "QUEUED") return "큐 대기";
    if (k === "RUNNING") return "실행 중";
    if (k === "TERMINAL") {
      if (events.terminalKind === "SUCCEEDED") return "성공";
      if (events.terminalKind === "FAILED") return "실패";
      if (events.terminalKind === "CANCELED") return "취소됨";
      return "종료";
    }
    return k;
  };

  const durations = TIMELINE_ORDER.slice(0, -1).map((from, i) => {
    const to = TIMELINE_ORDER[i + 1];
    if (events[from] && events[to]) {
      return events[to] - events[from];
    }
    return null;
  });

  return (
    <div className="ds-tl">
      {TIMELINE_ORDER.map((k, i) => (
        <React.Fragment key={k}>
          <span className={`ds-tl-event ${isCurrent(k) ? "is-current" : ""} ${reached(k) ? "is-reached" : ""}`}>
            {renderDot(k)}
            <span className="ds-tl-label">{stepLabel(k)}</span>
            {k === "QUEUED" && events.QUEUED != null && (
              <span className="ds-tl-aux" title="enqueueTimestamp">*</span>
            )}
          </span>
          {i < TIMELINE_ORDER.length - 1 && (
            <span className={`ds-tl-conn ${durations[i] == null ? "is-dashed" : ""}`}>
              <span className="ds-tl-conn-line"></span>
              {durations[i] != null && <span className="ds-tl-conn-dur">{fmtDuration(durations[i])}</span>}
              <span className="ds-tl-conn-line"></span>
            </span>
          )}
        </React.Fragment>
      ))}
      {attempt > 1 && (
        <span className="ds-tl-attempt">시도 #{attempt}</span>
      )}
    </div>
  );
};

window.JobTimeline = JobTimeline;
