// FileDropzone.jsx — file picker w/ drop target + selected-file row.
const FileDropzone = ({ value, onChange, accept = "image/*,application/pdf", helper = "png · jpeg · pdf" }) => {
  const inputRef = React.useRef(null);
  const [hover, setHover] = React.useState(false);

  const onPick = (e) => {
    const f = e.target.files && e.target.files[0];
    if (f) onChange(f);
  };

  const onDrop = (e) => {
    e.preventDefault();
    setHover(false);
    const f = e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) onChange(f);
  };

  if (value) {
    return (
      <div className="ds-file">
        <div className="ds-file-l">
          <div className="ds-file-tile">
            <i data-lucide={value.type?.startsWith("image/") ? "image" : "file-text"}></i>
          </div>
          <div className="ds-file-meta">
            <div className="ds-file-name">{value.name}</div>
            <div className="ds-file-sub">
              {(value.size / 1024).toFixed(1)} KB · {value.type || "application/octet-stream"}
            </div>
          </div>
        </div>
        <button type="button" className="ds-btn ds-btn-icon ds-btn-ghost" onClick={() => onChange(null)} aria-label="파일 제거">
          <i data-lucide="x"></i>
        </button>
      </div>
    );
  }

  return (
    <div
      className={`ds-drop ${hover ? "is-hover" : ""}`}
      onClick={() => inputRef.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setHover(true); }}
      onDragLeave={() => setHover(false)}
      onDrop={onDrop}
      role="button"
      tabIndex={0}
    >
      <i data-lucide="upload"></i>
      <div className="ds-drop-title">파일을 드롭하거나 클릭하여 선택</div>
      <div className="ds-drop-sub">{helper}</div>
      <input ref={inputRef} type="file" accept={accept} onChange={onPick} hidden />
    </div>
  );
};

window.FileDropzone = FileDropzone;
