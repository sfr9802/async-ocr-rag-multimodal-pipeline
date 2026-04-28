// Header.jsx — sticky top bar w/ logo, eyebrow, online chip, settings, theme toggle.
const Header = ({ online, dark, onToggleTheme }) => {
  return (
    <header className="ds-header">
      <div className="ds-header-inner">
        <div className="ds-brand">
          <img src="../../assets/logo.svg" width="28" height="28" alt="" />
          <div className="ds-brand-text">
            <div className="ds-brand-name">AI 파이프라인 콘솔</div>
            <div className="ds-brand-sub">async · ocr · rag · multimodal</div>
          </div>
        </div>

        <div className="ds-header-right">
          <span className={`ds-chip ${online === null ? "checking" : online ? "online" : "offline"}`}>
            <span className={`ds-chip-dot ${online === null ? "animate-pulse-soft" : ""}`}></span>
            <span className="ds-chip-label">{online === null ? "확인 중" : online ? "core-api 온라인" : "오프라인"}</span>
          </span>
          <button className="ds-btn ds-btn-icon ds-btn-ghost" aria-label="설정">
            <i data-lucide="settings-2"></i>
          </button>
          <button className="ds-btn ds-btn-icon ds-btn-ghost" aria-label="테마 전환" onClick={onToggleTheme}>
            <i data-lucide={dark ? "sun" : "moon"}></i>
          </button>
        </div>
      </div>
    </header>
  );
};

window.Header = Header;
