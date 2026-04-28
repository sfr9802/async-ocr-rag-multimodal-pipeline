// CapabilityPicker.jsx — segmented 4-up tile picker.
const CAPABILITIES = [
  { id: "MOCK", icon: "brain", desc: "에코" },
  { id: "RAG", icon: "file-text", desc: "검색" },
  { id: "OCR", icon: "image", desc: "추출" },
  { id: "MULTIMODAL", icon: "layers", desc: "비전 + 텍스트" },
];

const CAP_TONE_CLASS = {
  MOCK: "ds-cap-tile-mock",
  RAG: "ds-cap-tile-rag",
  OCR: "ds-cap-tile-ocr",
  MULTIMODAL: "ds-cap-tile-multimodal",
};

const CapabilityPicker = ({ value, onChange }) => {
  return (
    <div className="ds-cap-grid" role="radiogroup" aria-label="역량 선택">
      {CAPABILITIES.map((cap) => {
        const active = value === cap.id;
        return (
          <button
            key={cap.id}
            type="button"
            role="radio"
            aria-checked={active}
            onClick={() => onChange(cap.id)}
            className={`ds-cap-tile ${active ? `is-active ${CAP_TONE_CLASS[cap.id]}` : ""}`}
          >
            <div className="ds-cap-tile-top">
              <i data-lucide={cap.icon}></i>
              <span className="ds-cap-tile-tag">{cap.desc}</span>
            </div>
            <div className="ds-cap-tile-name">{cap.id === "MULTIMODAL" ? "멀티모달" : cap.id}</div>
          </button>
        );
      })}
    </div>
  );
};

window.CapabilityPicker = CapabilityPicker;
window.CAPABILITIES = CAPABILITIES;
