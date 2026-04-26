import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { StatusBadge } from "@/components/StatusBadge";
import { bucketByDate, formatRelative, shortId } from "@/lib/format";
import type { HistoryEntry } from "@/lib/history";
import type { Capability } from "@/lib/types";
import { cn } from "@/lib/utils";
import { Inbox, Trash2 } from "lucide-react";

const CAP_DOT: Record<Capability, string> = {
  MOCK: "bg-cap-mock",
  RAG: "bg-cap-rag",
  OCR: "bg-cap-ocr",
  MULTIMODAL: "bg-cap-multimodal",
};

interface JobSidebarProps {
  history: HistoryEntry[];
  statusByJob: Record<string, string | undefined>;
  activeJobId: string | null;
  onSelect: (jobId: string) => void;
  onRemove: (jobId: string) => void;
  onClear: () => void;
}

export function JobSidebar({
  history,
  statusByJob,
  activeJobId,
  onSelect,
  onRemove,
  onClear,
}: JobSidebarProps) {
  const groups = bucketByDate(history);

  return (
    <aside className="flex h-full w-full flex-col border-border/80 bg-card/40 lg:w-72 lg:border-r">
      <div className="flex items-center justify-between border-b border-border/60 px-4 py-3">
        <div>
          <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-muted-foreground">
            Recent
          </div>
          <div className="mt-0.5 font-mono text-[10.5px] text-muted-foreground/80">
            {history.length} stored locally
          </div>
        </div>
        {history.length > 0 && (
          <Button
            variant="ghost"
            size="sm"
            className="h-7 gap-1.5 px-2 text-[11px] font-medium text-muted-foreground"
            onClick={onClear}
          >
            <Trash2 className="h-3 w-3" />
            Clear
          </Button>
        )}
      </div>

      {history.length === 0 ? (
        <div className="flex flex-1 flex-col items-center justify-center gap-3 px-6 py-10 text-center">
          <div className="grid h-11 w-11 place-items-center rounded-full bg-muted text-muted-foreground/80">
            <Inbox className="h-4 w-4" />
          </div>
          <div className="space-y-1">
            <div className="text-[13px] font-medium">No jobs yet</div>
            <div className="text-[11.5px] leading-relaxed text-muted-foreground">
              Submit a job and it'll appear here.
            </div>
          </div>
        </div>
      ) : (
        <ScrollArea className="flex-1">
          <div className="space-y-4 p-2 pb-6">
            {groups.map(([bucket, entries]) => (
              <div key={bucket}>
                <div className="px-2 py-1.5 font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground/70">
                  {bucket}
                </div>
                <ul className="space-y-0.5">
                  {entries.map((h) => {
                    const status = statusByJob[h.jobId] ?? "PENDING";
                    const active = h.jobId === activeJobId;
                    return (
                      <li key={h.jobId}>
                        <button
                          type="button"
                          onClick={() => onSelect(h.jobId)}
                          className={cn(
                            "group relative flex w-full flex-col gap-1.5 rounded-md border border-transparent px-3 py-2.5 text-left transition-colors",
                            active
                              ? "border-border/80 bg-card shadow-soft"
                              : "hover:bg-foreground/[0.03]",
                          )}
                        >
                          {active && (
                            <span className="absolute left-0 top-2.5 h-[calc(100%-1.25rem)] w-[2.5px] rounded-r bg-primary" />
                          )}
                          <div className="flex items-center justify-between gap-2">
                            <div className="flex items-center gap-1.5">
                              <span className={cn("h-1.5 w-1.5 rounded-full", CAP_DOT[h.capability])} />
                              <span className="font-mono text-[10.5px] uppercase tracking-[0.14em] text-muted-foreground">
                                {h.capability}
                              </span>
                            </div>
                            <StatusBadge status={status} size="sm" withDot />
                          </div>
                          {h.preview && (
                            <div className="line-clamp-2 text-[12.5px] leading-snug text-foreground/85">
                              {h.preview}
                            </div>
                          )}
                          <div className="flex items-center justify-between">
                            <span className="font-mono text-[10px] text-muted-foreground/80">
                              {shortId(h.jobId)}
                            </span>
                            <span className="font-mono text-[10px] text-muted-foreground/60">
                              {formatRelative(h.submittedAt)}
                            </span>
                          </div>
                          <span
                            role="button"
                            tabIndex={0}
                            onClick={(e) => {
                              e.stopPropagation();
                              onRemove(h.jobId);
                            }}
                            onKeyDown={(e) => {
                              if (e.key === "Enter" || e.key === " ") {
                                e.preventDefault();
                                e.stopPropagation();
                                onRemove(h.jobId);
                              }
                            }}
                            className="absolute right-2 top-2 cursor-pointer rounded p-1 text-muted-foreground opacity-0 transition-opacity hover:bg-muted hover:text-destructive group-hover:opacity-100 focus-visible:opacity-100"
                            aria-label="Remove from history"
                          >
                            <Trash2 className="h-3 w-3" />
                          </span>
                        </button>
                      </li>
                    );
                  })}
                </ul>
              </div>
            ))}
          </div>
        </ScrollArea>
      )}
    </aside>
  );
}
