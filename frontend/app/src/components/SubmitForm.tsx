import { useMemo, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { ApiError, submitFileJob, submitTextJob } from "@/lib/api";
import { CAPABILITIES, type Capability, type JobCreated } from "@/lib/types";
import { cn } from "@/lib/utils";
import {
  Brain,
  CornerDownLeft,
  FileText,
  ImageIcon,
  Layers,
  Loader2,
  Upload,
  X,
} from "lucide-react";

interface CapabilityMeta {
  key: Capability;
  label: string;
  shortDesc: string;
  longDesc: string;
  Icon: typeof Brain;
  needsFile: boolean;
  needsText: boolean;
  textOptional?: boolean;
  iconActiveClass: string;
  tagActiveClass: string;
  ringClass: string;
}

const META: Record<Capability, CapabilityMeta> = {
  MOCK: {
    key: "MOCK",
    label: "Mock",
    shortDesc: "에코",
    longDesc: "프롬프트를 JSON 아티팩트로 그대로 반환합니다. 파이프라인 전체 흐름을 점검할 때 유용합니다.",
    Icon: Brain,
    needsFile: false,
    needsText: true,
    iconActiveClass: "text-cap-mock",
    tagActiveClass: "text-cap-mock",
    ringClass: "ring-cap-mock/40",
  },
  RAG: {
    key: "RAG",
    label: "RAG",
    shortDesc: "검색",
    longDesc: "인덱싱된 코퍼스에 대한 질의응답 · bge-m3 + FAISS · Claude 생성은 선택 사항.",
    Icon: FileText,
    needsFile: false,
    needsText: true,
    iconActiveClass: "text-cap-rag",
    tagActiveClass: "text-cap-rag",
    ringClass: "ring-cap-rag/45",
  },
  OCR: {
    key: "OCR",
    label: "OCR",
    shortDesc: "추출",
    longDesc: "PNG / JPEG / PDF에서 텍스트 추출. 선택적 프롬프트는 워커가 무시합니다.",
    Icon: ImageIcon,
    needsFile: true,
    needsText: false,
    textOptional: true,
    iconActiveClass: "text-cap-ocr",
    tagActiveClass: "text-cap-ocr",
    ringClass: "ring-cap-ocr/45",
  },
  MULTIMODAL: {
    key: "MULTIMODAL",
    label: "멀티모달",
    shortDesc: "비전 + 텍스트",
    longDesc: "Claude Vision으로 이미지 + 질문을 결합합니다. PNG / JPEG / PDF + 질문 필요.",
    Icon: Layers,
    needsFile: true,
    needsText: true,
    iconActiveClass: "text-cap-multimodal",
    tagActiveClass: "text-cap-multimodal",
    ringClass: "ring-cap-multimodal/45",
  },
};

interface SubmitFormProps {
  onSubmitted: (job: JobCreated, preview: string) => void;
}

export function SubmitForm({ onSubmitted }: SubmitFormProps) {
  const [capability, setCapability] = useState<Capability>("MOCK");
  const [text, setText] = useState<string>("새 콘솔에서 보내는 인사");
  const [file, setFile] = useState<File | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const meta = META[capability];

  const canSubmit = useMemo(() => {
    if (submitting) return false;
    if (meta.needsFile && !file) return false;
    if (meta.needsText && !meta.textOptional && text.trim().length === 0) return false;
    return true;
  }, [submitting, meta, file, text]);

  async function handleSubmit() {
    setSubmitting(true);
    setError(null);
    try {
      let result: JobCreated;
      if (meta.needsFile) {
        if (!file) throw new Error("파일이 필요합니다");
        result = await submitFileJob(capability, file, text);
      } else {
        result = await submitTextJob(capability, text);
      }
      const preview = meta.needsFile
        ? `${file?.name ?? ""}${text.trim() ? ` — ${text}` : ""}`
        : text;
      onSubmitted(result, preview);
      setFile(null);
      if (fileInput.current) fileInput.current.value = "";
    } catch (e) {
      if (e instanceof ApiError) {
        setError(typeof e.body === "object" && e.body ? `${e.body.code}: ${e.body.message}` : e.message);
      } else {
        setError(e instanceof Error ? e.message : String(e));
      }
    } finally {
      setSubmitting(false);
    }
  }

  function onKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter" && canSubmit) {
      e.preventDefault();
      handleSubmit();
    }
  }

  function onDrop(e: React.DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setDragOver(false);
    if (!meta.needsFile) return;
    const f = e.dataTransfer.files[0];
    if (f) setFile(f);
  }

  return (
    <section className="overflow-hidden rounded-[22px] border border-hairline-2 bg-glass shadow-glass backdrop-blur-[18px] backdrop-saturate-150">
      <header className="flex flex-wrap items-end justify-between gap-4 px-6 pt-5">
        <div>
          <h2 className="text-[15px] font-semibold tracking-snug">새 작업</h2>
          <p className="mt-1 text-[12.5px] text-muted-foreground">
            기능을 고르고 입력을 채우세요.
          </p>
        </div>
      </header>

      <div className="space-y-5 px-6 py-5">
        <div>
          <Label className="text-[11px] font-medium uppercase tracking-[0.14em] text-muted-foreground">
            기능
          </Label>
          <div className="mt-2.5 grid grid-cols-2 gap-2.5 lg:grid-cols-4">
            {CAPABILITIES.map((c) => {
              const m = META[c];
              const active = c === capability;
              return (
                <button
                  key={c}
                  type="button"
                  onClick={() => setCapability(c)}
                  data-active={active}
                  data-cap={c}
                  className={cn(
                    "group flex flex-col items-start gap-2 rounded-[10px] border p-3 text-left transition-all",
                    active
                      ? cn("border-hairline-2 bg-glass-strong shadow-glass-pop ring-1", m.ringClass)
                      : "border-hairline-2 bg-glass-2 hover:-translate-y-px hover:bg-glass-3",
                  )}
                >
                  <div className="flex w-full items-center justify-between">
                    <m.Icon
                      className={cn(
                        "h-4 w-4 transition-colors",
                        active ? m.iconActiveClass : "text-muted-foreground",
                      )}
                    />
                    <span
                      className={cn(
                        "font-mono text-[9.5px] uppercase tracking-[0.18em]",
                        active ? m.tagActiveClass : "text-muted-foreground/70",
                      )}
                    >
                      {m.shortDesc}
                    </span>
                  </div>
                  <span
                    className={cn(
                      "text-[14px] font-semibold tracking-snug",
                      active ? "text-foreground" : "text-foreground/85",
                    )}
                  >
                    {m.label}
                  </span>
                </button>
              );
            })}
          </div>
          <p className="mt-2.5 text-[12px] leading-relaxed text-muted-foreground text-balance">
            {meta.longDesc}
          </p>
        </div>

        {meta.needsFile && (
          <div>
            <Label className="text-[11px] font-medium uppercase tracking-[0.14em] text-muted-foreground">
              파일 · PNG / JPEG / PDF
            </Label>
            <div className="mt-2">
              {file ? (
                <div className="flex items-center justify-between rounded-[10px] border border-hairline-2 bg-glass-3 px-3 py-2.5 text-sm">
                  <div className="flex min-w-0 items-center gap-2.5">
                    <div className="grid h-8 w-8 shrink-0 place-items-center rounded-lg border border-hairline-2 bg-glass-strong">
                      <FileText className="h-4 w-4 text-muted-foreground" />
                    </div>
                    <div className="min-w-0">
                      <div className="truncate text-[13px] font-medium">{file.name}</div>
                      <div className="font-mono text-[10.5px] text-muted-foreground">
                        {(file.size / 1024).toFixed(1)} KB · {file.type || "—"}
                      </div>
                    </div>
                  </div>
                  <Button
                    type="button"
                    variant="ghost"
                    size="icon"
                    className="h-7 w-7 hover:bg-glass-2"
                    onClick={() => {
                      setFile(null);
                      if (fileInput.current) fileInput.current.value = "";
                    }}
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </div>
              ) : (
                <div
                  onDragOver={(e) => {
                    e.preventDefault();
                    setDragOver(true);
                  }}
                  onDragLeave={() => setDragOver(false)}
                  onDrop={onDrop}
                  className={cn(
                    "rounded-[10px] border-[1.5px] border-dashed transition-colors",
                    dragOver
                      ? "border-ring/60 bg-glass-3"
                      : "border-hairline-strong bg-glass-2 hover:border-ring/50 hover:bg-glass-3",
                  )}
                >
                  <button
                    type="button"
                    onClick={() => fileInput.current?.click()}
                    className="flex w-full flex-col items-center justify-center gap-1.5 px-4 py-7 text-sm text-muted-foreground"
                  >
                    <Upload className="h-4 w-4" />
                    <span className="text-[13px] font-medium text-foreground/85">파일을 드롭하거나 클릭하여 선택</span>
                    <span className="font-mono text-[10.5px] uppercase tracking-[0.14em]">
                      png · jpeg · pdf
                    </span>
                  </button>
                </div>
              )}
              <input
                ref={fileInput}
                type="file"
                accept="image/png,image/jpeg,application/pdf"
                className="hidden"
                onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              />
            </div>
          </div>
        )}

        <div>
          <div className="flex items-baseline justify-between">
            <Label htmlFor="prompt" className="text-[11px] font-medium uppercase tracking-[0.14em] text-muted-foreground">
              {meta.needsFile ? "프롬프트" : "텍스트"}
              {meta.textOptional && <span className="ml-1 normal-case tracking-normal text-muted-foreground/70">(선택)</span>}
            </Label>
            <span className="font-mono text-[10.5px] tabular-nums text-muted-foreground">{text.length}자</span>
          </div>
          <Textarea
            id="prompt"
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder={
              meta.needsFile
                ? "파일과 함께 사용할 질문 (선택)…"
                : "워커에게 전달할 프롬프트…"
            }
            className="mt-2 min-h-[112px] resize-y rounded-lg border-hairline-2 bg-glass-3 font-mono text-[12.5px] leading-relaxed shadow-none focus-visible:bg-glass-strong focus-visible:ring-2 focus-visible:ring-ring/30"
          />
        </div>

        {error && (
          <div className="rounded-lg border border-destructive/40 bg-destructive/10 px-3 py-2.5 text-[12.5px] text-destructive">
            {error}
          </div>
        )}
      </div>

      <footer className="flex items-center justify-between gap-3 border-t border-dashed border-hairline-strong px-6 py-3.5">
        <span className="hidden items-center gap-1.5 font-mono text-[10.5px] uppercase tracking-[0.14em] text-muted-foreground sm:inline-flex">
          <kbd className="inline-grid h-[18px] min-w-[18px] place-items-center rounded border border-b-2 border-hairline-strong bg-glass-3 px-1 text-[10.5px] tracking-normal">⌘</kbd>
          <kbd className="inline-grid h-[18px] min-w-[18px] place-items-center rounded border border-b-2 border-hairline-strong bg-glass-3 px-1 text-[10.5px] tracking-normal">↵</kbd>
          <span>전송</span>
        </span>
        <Button
          onClick={handleSubmit}
          disabled={!canSubmit}
          className="h-9 gap-1.5 px-4 shadow-glass-button transition-all hover:-translate-y-px hover:bg-primary/90"
        >
          {submitting ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <CornerDownLeft className="h-3.5 w-3.5" />
          )}
          {submitting ? "전송 중…" : "작업 제출"}
        </Button>
      </footer>
    </section>
  );
}
