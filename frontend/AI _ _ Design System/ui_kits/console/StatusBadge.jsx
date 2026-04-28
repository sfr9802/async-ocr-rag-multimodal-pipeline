// StatusBadge.jsx — pill with icon + KR label, derived from a status code.
const STATUS_VARIANTS = {
  PENDING:    { label: "대기",    tone: "muted",       icon: "circle-dashed" },
  QUEUED:     { label: "큐 대기", tone: "muted",       icon: "circle-dashed" },
  RUNNING:    { label: "실행 중", tone: "warning",     icon: "loader-2", spin: true },
  IN_PROGRESS:{ label: "실행 중", tone: "warning",     icon: "loader-2", spin: true },
  SUCCEEDED:  { label: "성공",    tone: "success",     icon: "check-circle-2" },
  COMPLETED:  { label: "성공",    tone: "success",     icon: "check-circle-2" },
  FAILED:     { label: "실패",    tone: "destructive", icon: "circle-alert" },
  CANCELED:   { label: "취소됨",  tone: "muted",       icon: "circle-slash" },
  CANCELLED:  { label: "취소됨",  tone: "muted",       icon: "circle-slash" },
};

const StatusBadge = ({ status, size = "md", withDot = false }) => {
  const key = (status || "PENDING").toUpperCase();
  const v = STATUS_VARIANTS[key] || STATUS_VARIANTS.PENDING;
  const className = `ds-pill ds-pill-${size} ds-pill-${v.tone}`;
  return (
    <span className={className}>
      {withDot ? (
        <span className={`ds-pill-dot ${v.tone === "warning" ? "animate-pulse-soft" : ""}`}></span>
      ) : (
        <i data-lucide={v.icon} className={v.spin ? "ds-icon-spin" : ""}></i>
      )}
      {v.label}
    </span>
  );
};

window.StatusBadge = StatusBadge;
window.STATUS_VARIANTS = STATUS_VARIANTS;
