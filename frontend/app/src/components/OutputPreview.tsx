import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Check, Copy, FileWarning } from "lucide-react";
import { fetchArtifactText, isTextLike } from "@/lib/api";
import { tryPrettyJson } from "@/lib/format";
import type { ArtifactView } from "@/lib/types";
import { cn } from "@/lib/utils";

interface OutputPreviewProps {
  artifact: ArtifactView;
}

export function OutputPreview({ artifact }: OutputPreviewProps) {
  const [text, setText] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setText(null);
    setError(null);
    if (!isTextLike(artifact.contentType)) {
      setLoading(false);
      setError("binary");
      return;
    }
    fetchArtifactText(artifact.accessUrl)
      .then((t) => {
        if (!cancelled) setText(t);
      })
      .catch((e) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [artifact.accessUrl, artifact.contentType]);

  if (error === "binary") {
    return (
      <div className="flex items-center gap-2 rounded-md border border-dashed bg-muted/30 px-4 py-6 text-xs text-muted-foreground">
        <FileWarning className="h-4 w-4" />
        바이너리 출력 ({artifact.contentType}) — 아래 다운로드를 사용하세요.
      </div>
    );
  }

  if (loading) {
    return (
      <div className="space-y-2">
        <div className="h-3 w-1/3 animate-pulse-soft rounded bg-muted" />
        <div className="h-3 w-5/6 animate-pulse-soft rounded bg-muted" />
        <div className="h-3 w-4/6 animate-pulse-soft rounded bg-muted" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive">
        출력 로드 실패: {error}
      </div>
    );
  }

  const { pretty, isJson } = tryPrettyJson(text ?? "");

  return (
    <div className="group relative">
      <div className="absolute right-2 top-2 z-10 flex items-center gap-1 opacity-0 transition-opacity group-hover:opacity-100">
        {isJson && (
          <span className="rounded bg-background/80 px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wider text-muted-foreground backdrop-blur">
            json
          </span>
        )}
        <Button
          variant="ghost"
          size="icon"
          className="h-7 w-7 bg-background/80 backdrop-blur"
          onClick={() => {
            navigator.clipboard.writeText(pretty);
            setCopied(true);
            window.setTimeout(() => setCopied(false), 1500);
          }}
          aria-label="출력 복사"
        >
          {copied ? <Check className="h-3.5 w-3.5 text-success" /> : <Copy className="h-3.5 w-3.5" />}
        </Button>
      </div>
      <pre
        className={cn(
          "pretty-pre max-h-[420px] overflow-auto rounded-md border bg-card p-4 font-mono text-[12.5px] leading-relaxed text-foreground/90",
          "shadow-soft",
        )}
      >
        {pretty}
      </pre>
    </div>
  );
}
