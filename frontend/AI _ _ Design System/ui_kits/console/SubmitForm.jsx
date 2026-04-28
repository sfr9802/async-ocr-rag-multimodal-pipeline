// SubmitForm.jsx — capability-aware composer for new jobs.
const SubmitForm = ({ capability, setCapability, onSubmit, busy, baseUrl, setBaseUrl }) => {
  const [prompt, setPrompt] = React.useState("새 콘솔에서 보내는 인사");
  const [ragQuestion, setRagQuestion] = React.useState("anime about an old fisherman feeding stray harbor cats");
  const [topK, setTopK] = React.useState(5);
  const [mmQuestion, setMmQuestion] = React.useState("이 문서의 첫 줄에는 무엇이 있나요?");
  const [file, setFile] = React.useState(null);

  const canSubmit = (() => {
    if (busy) return false;
    if (capability === "MOCK") return prompt.trim().length > 0;
    if (capability === "RAG") return ragQuestion.trim().length > 0;
    if (capability === "OCR") return !!file;
    if (capability === "MULTIMODAL") return !!file && mmQuestion.trim().length > 0;
    return false;
  })();

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!canSubmit) return;
    if (capability === "MOCK") onSubmit({ capability, prompt });
    if (capability === "RAG") onSubmit({ capability, question: ragQuestion, topK });
    if (capability === "OCR") onSubmit({ capability, file });
    if (capability === "MULTIMODAL") onSubmit({ capability, file, question: mmQuestion });
  };

  return (
    <form className="ds-card ds-form" onSubmit={handleSubmit}>
      <div className="ds-form-head">
        <div>
          <div className="ds-form-title">새 작업</div>
          <div className="ds-form-sub">역량을 고르고 입력을 채우세요. 큐가 처리합니다.</div>
        </div>
        <div className="ds-form-baseurl">
          <label className="ds-label">Base URL</label>
          <input className="ds-input ds-mono" type="text" value={baseUrl} onChange={(e) => setBaseUrl(e.target.value)} />
        </div>
      </div>

      <CapabilityPicker value={capability} onChange={setCapability} />

      <div className="ds-form-body">
        {capability === "MOCK" && (
          <div className="ds-field">
            <label className="ds-label">프롬프트</label>
            <textarea className="ds-textarea ds-mono" rows={3} value={prompt} onChange={(e) => setPrompt(e.target.value)} placeholder="워커에게 전달할 프롬프트…" />
            <div className="ds-help">MOCK 워커는 ~3초 후 prompt를 그대로 echo 합니다.</div>
          </div>
        )}

        {capability === "RAG" && (
          <div className="ds-grid-2">
            <div className="ds-field">
              <label className="ds-label">질문</label>
              <textarea className="ds-textarea" rows={3} value={ragQuestion} onChange={(e) => setRagQuestion(e.target.value)} />
            </div>
            <div className="ds-field">
              <label className="ds-label">Top-K</label>
              <input className="ds-input ds-mono" type="number" min="1" max="20" value={topK} onChange={(e) => setTopK(Number(e.target.value))} />
              <div className="ds-help">데모 인덱스: anime-001…anime-008.</div>
            </div>
          </div>
        )}

        {capability === "OCR" && (
          <div className="ds-field">
            <label className="ds-label">이미지 또는 PDF</label>
            <FileDropzone value={file} onChange={setFile} />
          </div>
        )}

        {capability === "MULTIMODAL" && (
          <div className="ds-grid-2">
            <div className="ds-field">
              <label className="ds-label">파일</label>
              <FileDropzone value={file} onChange={setFile} />
            </div>
            <div className="ds-field">
              <label className="ds-label">질문</label>
              <textarea className="ds-textarea" rows={5} value={mmQuestion} onChange={(e) => setMmQuestion(e.target.value)} />
            </div>
          </div>
        )}
      </div>

      <div className="ds-form-foot">
        <div className="ds-form-foot-l">
          <span className="ds-kbd"><kbd>⌘</kbd><kbd>Enter</kbd></span>
          <span>전송</span>
        </div>
        <button type="submit" className="ds-btn ds-btn-primary" disabled={!canSubmit}>
          {busy ? <i data-lucide="loader-2" className="ds-icon-spin"></i> : <i data-lucide="corner-down-left"></i>}
          작업 제출
        </button>
      </div>
    </form>
  );
};

window.SubmitForm = SubmitForm;
