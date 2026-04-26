import { useMemo, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
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
  Sparkles,
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
  ringClass: string;
  iconClass: string;
}

const META: Record<Capability, CapabilityMeta> = {
  MOCK: {
    key: "MOCK",
    label: "Mock",
    shortDesc: "Echo",
    longDesc: "Echoes the prompt back as a JSON artifact. Useful for smoke-testing the pipeline end-to-end.",
    Icon: Brain,
    needsFile: false,
    needsText: true,
    ringClass: "data-[active=true]:ring-cap-mock/50 data-[active=true]:border-cap-mock/40",
    iconClass: "text-cap-mock",
  },
  RAG: {
    key: "RAG",
    label: "RAG",
    shortDesc: "Retrieval",
    longDesc: "Question answering over the indexed corpus · bge-m3 + FAISS · Claude generation optional.",
    Icon: FileText,
    needsFile: false,
    needsText: true,
    ringClass: "data-[active=true]:ring-cap-rag/50 data-[active=true]:border-cap-rag/40",
    iconClass: "text-cap-rag",
  },
  OCR: {
    key: "OCR",
    label: "OCR",
    shortDesc: "Extract",
    longDesc: "Text extraction from PNG / JPEG / PDF. Optional prompt is ignored by the worker.",
    Icon: ImageIcon,
    needsFile: true,
    needsText: false,
    textOptional: true,
    ringClass: "data-[active=true]:ring-cap-ocr/50 data-[active=true]:border-cap-ocr/40",
    iconClass: "text-cap-ocr",
  },
  MULTIMODAL: {
    key: "MULTIMODAL",
    label: "Multimodal",
    shortDesc: "Vision + Text",
    longDesc: "Image + question fusion via Claude Vision. PNG / JPEG / PDF + a question.",
    Icon: Layers,
    needsFile: true,
    needsText: true,
    ringClass: "data-[active=true]:ring-cap-multimodal/50 data-[active=true]:border-cap-multimodal/40",
    iconClass: "text-cap-multimodal",
  },
};

interface SubmitFormProps {
  onSubmitted: (job: JobCreated, preview: string) => void;
}

export function SubmitForm({ onSubmitted }: SubmitFormProps) {
  const [capability, setCapability] = useState<Capability>("MOCK");
  const [text, setText] = useState<string>("hello from the new console");
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
        if (!file) throw new Error("File required");
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
    <Card className="overflow-hidden border-border/80 shadow-soft">
      <CardHeader className="border-b border-border/60 bg-muted/30 px-5 py-3">
        <div className="flex items-center gap-2">
          <Sparkles className="h-3.5 w-3.5 text-accent" />
          <h2 className="text-[11px] font-semibold uppercase tracking-[0.16em] text-muted-foreground">
            New job
          </h2>
        </div>
      </CardHeader>
      <CardContent className="space-y-6 p-5 sm:p-6">
        <div>
          <Label className="text-[11px] font-medium uppercase tracking-[0.14em] text-muted-foreground">
            Capability
          </Label>
          <div className="mt-2.5 grid grid-cols-2 gap-2 lg:grid-cols-4">
            {CAPABILITIES.map((c) => {
              const m = META[c];
              const active = c === capability;
              return (
                <button
                  key={c}
                  type="button"
                  onClick={() => setCapability(c)}
                  data-active={active}
                  className={cn(
                    "group relative flex flex-col items-start gap-2 rounded-md border p-3 text-left transition-all",
                    "hover:border-foreground/25 hover:bg-foreground/[0.02]",
                    "data-[active=true]:bg-card data-[active=true]:shadow-soft data-[active=true]:ring-2",
                    m.ringClass,
                  )}
                >
                  <div className="flex w-full items-center justify-between">
                    <m.Icon
                      className={cn(
                        "h-4 w-4 transition-colors",
                        active ? m.iconClass : "text-muted-foreground",
                      )}
                    />
                    <span
                      className={cn(
                        "font-mono text-[9.5px] uppercase tracking-[0.18em]",
                        active ? m.iconClass : "text-muted-foreground/70",
                      )}
                    >
                      {m.shortDesc}
                    </span>
                  </div>
                  <span
                    className={cn(
                      "text-sm font-semibold tracking-snug",
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
              File · PNG / JPEG / PDF
            </Label>
            <div className="mt-2">
              {file ? (
                <div className="flex items-center justify-between rounded-md border border-border bg-secondary/40 px-3 py-2.5 text-sm">
                  <div className="flex min-w-0 items-center gap-2.5">
                    <div className="grid h-8 w-8 shrink-0 place-items-center rounded-md bg-background border">
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
                    className="h-7 w-7"
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
                    "rounded-md border-2 border-dashed transition-colors",
                    dragOver
                      ? "border-accent bg-accent/5"
                      : "border-border bg-muted/15 hover:border-foreground/25 hover:bg-muted/30",
                  )}
                >
                  <button
                    type="button"
                    onClick={() => fileInput.current?.click()}
                    className="flex w-full flex-col items-center justify-center gap-1.5 px-4 py-7 text-sm text-muted-foreground"
                  >
                    <Upload className="h-4 w-4" />
                    <span className="font-medium text-foreground/85">Drop a file or click to choose</span>
                    <span className="font-mono text-[10.5px] uppercase tracking-wider">
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
              {meta.needsFile ? "Prompt" : "Text"}
              {meta.textOptional && <span className="ml-1 normal-case tracking-normal text-muted-foreground/70">(optional)</span>}
            </Label>
            <span className="font-mono text-[10.5px] tabular-nums text-muted-foreground">{text.length} chars</span>
          </div>
          <Textarea
            id="prompt"
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder={
              meta.needsFile
                ? "Optional question to fuse with the file…"
                : "Your prompt for the worker…"
            }
            className="mt-2 min-h-[112px] resize-y rounded-md border-border bg-card font-mono text-[12.5px] leading-relaxed shadow-none focus-visible:ring-2 focus-visible:ring-ring/30"
          />
        </div>

        {error && (
          <div className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2.5 text-[12.5px] text-destructive">
            {error}
          </div>
        )}

        <div className="flex items-center justify-between gap-3">
          <span className="hidden items-center gap-1.5 font-mono text-[10.5px] uppercase tracking-wider text-muted-foreground sm:inline-flex">
            <kbd className="rounded border border-border bg-card px-1.5 py-0.5 text-[10px] tracking-normal">⌘</kbd>
            <kbd className="rounded border border-border bg-card px-1.5 py-0.5 text-[10px] tracking-normal">↵</kbd>
            <span>to submit</span>
          </span>
          <Button onClick={handleSubmit} disabled={!canSubmit} className="h-9 gap-1.5 px-4">
            {submitting ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <CornerDownLeft className="h-3.5 w-3.5" />
            )}
            {submitting ? "Submitting…" : "Submit"}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
