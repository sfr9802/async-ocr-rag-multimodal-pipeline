import { Fragment, useEffect, useMemo, useState } from "react";
import { formatDuration } from "@/lib/format";
import { isTerminal, type JobEvent, type JobView } from "@/lib/types";
import { cn } from "@/lib/utils";

const TONE: Record<string, string> = {
  PENDING: "bg-muted-foreground/55",
  QUEUED: "bg-muted-foreground/55",
  RUNNING: "bg-warning",
  IN_PROGRESS: "bg-warning",
  SUCCEEDED: "bg-success",
  COMPLETED: "bg-success",
  FAILED: "bg-destructive",
  CANCELED: "bg-muted-foreground/45",
  CANCELLED: "bg-muted-foreground/45",
};

const LABEL: Record<string, string> = {
  PENDING: "대기",
  QUEUED: "큐 대기",
  RUNNING: "실행 중",
  IN_PROGRESS: "실행 중",
  SUCCEEDED: "성공",
  COMPLETED: "성공",
  FAILED: "실패",
  CANCELED: "취소됨",
  CANCELLED: "취소됨",
};

interface JobTimelineProps {
  view: JobView;
  className?: string;
}

export function JobTimeline({ view, className }: JobTimelineProps) {
  const snapshots = useStatusSnapshots(view);
  const events = useMemo(() => buildEvents(view, snapshots), [view, snapshots]);
  const live = !isTerminal(view.status);
  const showAttempt = view.attemptNo > 1;

  if (events.length === 0) return null;

  return (
    <div
      className={cn(
        "t-mono-tag flex flex-wrap items-center gap-x-2 gap-y-1.5",
        className,
      )}
      role="list"
      aria-label="작업 생애주기 타임라인"
    >
      {events.map((ev, i) => {
        const key = `${ev.status}@${ev.at}`;
        const isLast = i === events.length - 1;
        const prev = i > 0 ? events[i - 1] : null;
        return (
          <Fragment key={key}>
            {prev && <Connector fromIso={prev.at} toIso={ev.at} dashed={ev.source === "client"} />}
            <EventNode event={ev} active={isLast && live} />
          </Fragment>
        );
      })}
      {showAttempt && (
        <span
          className="ml-1 rounded-full border border-hairline-2 bg-glass-3 px-2 py-0.5 text-[9.5px] tracking-[0.18em]"
          aria-label={`시도 횟수 ${view.attemptNo}`}
        >
          시도 #{view.attemptNo}
        </span>
      )}
    </div>
  );
}

function EventNode({ event, active }: { event: JobEvent; active: boolean }) {
  const status = event.status.toUpperCase();
  const tone = TONE[status] ?? "bg-muted-foreground/55";
  const label = LABEL[status] ?? event.status;
  const derived = event.source === "client";
  const tip = `${label}${derived ? " (클라이언트 관찰)" : ""} · ${new Date(event.at).toLocaleString()}`;
  return (
    <span
      role="listitem"
      title={tip}
      className="inline-flex items-center gap-1.5"
    >
      <span
        aria-hidden="true"
        className={cn("h-1.5 w-1.5 rounded-full", tone, active && "animate-pulse-soft")}
      />
      <span className={cn("normal-case tracking-snug", active ? "text-foreground" : "text-muted-foreground")}>
        <span className="uppercase tracking-[0.16em]">{label}</span>
        {derived && <span aria-hidden="true" className="ml-0.5 text-muted-foreground/70">*</span>}
      </span>
    </span>
  );
}

function Connector({ fromIso, toIso, dashed }: { fromIso: string; toIso: string; dashed: boolean }) {
  const duration = formatDuration(fromIso, toIso);
  return (
    <span aria-hidden="true" className="inline-flex items-center gap-1.5">
      <span className={cn("h-px w-3", dashed ? "border-t border-dashed border-hairline-2" : "bg-hairline-2")} />
      <span className="tabular-nums tracking-normal text-muted-foreground/80">{duration}</span>
      <span className={cn("h-px w-3", dashed ? "border-t border-dashed border-hairline-2" : "bg-hairline-2")} />
    </span>
  );
}

function useStatusSnapshots(view: JobView | null): Map<string, string> {
  const jobId = view?.jobId ?? null;
  const status = view?.status?.toUpperCase() ?? null;
  const [snapshots, setSnapshots] = useState<Map<string, string>>(() => new Map());

  useEffect(() => {
    setSnapshots(new Map());
  }, [jobId]);

  useEffect(() => {
    if (!status) return;
    setSnapshots((prev) => {
      if (prev.has(status)) return prev;
      const next = new Map(prev);
      next.set(status, new Date().toISOString());
      return next;
    });
  }, [status]);

  return snapshots;
}

function buildEvents(view: JobView | null, snapshots: Map<string, string>): JobEvent[] {
  if (!view) return [];
  const events: JobEvent[] = [];
  const status = view.status.toUpperCase();

  events.push({ status: "PENDING", at: view.createdAt, source: "server" });

  const queuedAt = snapshots.get("QUEUED");
  if (queuedAt && status !== "PENDING") {
    events.push({ status: "QUEUED", at: queuedAt, source: "client" });
  }

  if (view.claimedAt) {
    events.push({ status: "RUNNING", at: view.claimedAt, source: "server" });
  } else if (snapshots.get("RUNNING")) {
    events.push({ status: "RUNNING", at: snapshots.get("RUNNING")!, source: "client" });
  }

  if (isTerminal(status)) {
    events.push({ status, at: view.updatedAt, source: "server" });
  }

  return dedupe(events);
}

function dedupe(events: JobEvent[]): JobEvent[] {
  const out: JobEvent[] = [];
  for (const ev of events) {
    const last = out[out.length - 1];
    if (last && last.status === ev.status) continue;
    out.push(ev);
  }
  return out;
}
