import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { StatusBadge } from "@/components/StatusBadge";
import { OutputPreview } from "@/components/OutputPreview";
import { ApiError, artifactUrl, getJob, getJobResult } from "@/lib/api";
import { formatBytes, formatDuration, formatRelative, shortId } from "@/lib/format";
import { isTerminal, type ArtifactView, type JobResult, type JobView } from "@/lib/types";
import { cn } from "@/lib/utils";
import {
  Activity,
  Check,
  CircleAlert,
  Copy,
  Download,
  ExternalLink,
  FileText,
  Hash,
  Inbox,
  Sparkles,
} from "lucide-react";

const POLL_INTERVAL_MS = 1500;

interface JobPanelProps {
  jobId: string | null;
  onStatusChange: (jobId: string, status: string) => void;
}

export function JobPanel({ jobId, onStatusChange }: JobPanelProps) {
  const [view, setView] = useState<JobView | null>(null);
  const [result, setResult] = useState<JobResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (!jobId) {
      setView(null);
      setResult(null);
      setError(null);
      return;
    }
    let cancelled = false;
    let timer: number | null = null;

    async function tick() {
      if (cancelled || !jobId) return;
      try {
        const v = await getJob(jobId);
        if (cancelled) return;
        setView(v);
        setError(null);
        onStatusChange(jobId, v.status);
        if (isTerminal(v.status)) {
          try {
            const r = await getJobResult(jobId);
            if (!cancelled) setResult(r);
          } catch (e) {
            if (!cancelled) setError(formatErr(e));
          }
        } else {
          setResult(null);
          timer = window.setTimeout(tick, POLL_INTERVAL_MS);
        }
      } catch (e) {
        if (!cancelled) setError(formatErr(e));
      }
    }

    setView(null);
    setResult(null);
    tick();

    return () => {
      cancelled = true;
      if (timer != null) window.clearTimeout(timer);
    };
  }, [jobId, onStatusChange]);

  if (!jobId) return <EmptyState />;

  const statusKey = (view?.status ?? "PENDING").toUpperCase();
  const isRunning = !isTerminal(statusKey);
  const primaryOutput = pickPrimaryOutput(result?.outputs);

  return (
    <Card className="overflow-hidden border-border/80 shadow-soft">
      <div className="flex flex-wrap items-start gap-x-6 gap-y-3 border-b border-border/60 bg-muted/25 px-5 py-4">
        <div className="flex items-center gap-3">
          <StatusBadge status={statusKey} withDot />
          {isRunning && (
            <span className="inline-flex items-center gap-1.5 font-mono text-[10.5px] uppercase tracking-[0.14em] text-muted-foreground">
              <Activity className="h-3 w-3 animate-pulse-soft" />
              polling
            </span>
          )}
        </div>

        <Stat label="Job">
          <button
            onClick={() => {
              navigator.clipboard.writeText(jobId);
              setCopied(true);
              window.setTimeout(() => setCopied(false), 1500);
            }}
            className="group inline-flex items-center gap-1.5 font-mono text-[12px] text-foreground hover:text-primary"
          >
            <Hash className="h-3 w-3 text-muted-foreground" />
            {shortId(jobId, 12)}
            {copied ? (
              <Check className="h-3 w-3 text-success" />
            ) : (
              <Copy className="h-3 w-3 opacity-0 group-hover:opacity-100" />
            )}
          </button>
        </Stat>

        {view && (
          <>
            <Stat label="Capability">
              <span className="font-mono text-[12px] uppercase tracking-[0.06em]">{view.capability}</span>
            </Stat>
            <Stat label="Attempt">
              <span className="font-mono tabular-nums text-[12px]">#{view.attemptNo}</span>
            </Stat>
            <Stat label={isTerminal(statusKey) ? "Duration" : "Updated"}>
              <span className="font-mono tabular-nums text-[12px]">
                {isTerminal(statusKey)
                  ? formatDuration(view.createdAt, view.updatedAt)
                  : formatRelative(view.updatedAt)}
              </span>
            </Stat>
          </>
        )}
      </div>

      {(error || view?.errorCode) && (
        <div className="border-b border-destructive/30 bg-destructive/8 px-5 py-3">
          <div className="flex items-start gap-2 text-destructive">
            <CircleAlert className="mt-0.5 h-4 w-4 shrink-0" />
            <div className="space-y-0.5">
              {view?.errorCode && (
                <div className="text-[13px] font-semibold tracking-snug">{view.errorCode}</div>
              )}
              <div className="font-mono text-[11.5px] leading-relaxed">
                {view?.errorMessage ?? error}
              </div>
            </div>
          </div>
        </div>
      )}

      <CardContent className="p-0">
        <Tabs defaultValue="result" className="w-full">
          <TabsList className="h-auto w-full justify-start gap-0 rounded-none border-b border-border/60 bg-transparent p-0 px-3">
            <TabTriggerSlim value="result">Result</TabTriggerSlim>
            <TabTriggerSlim value="artifacts">
              Artifacts
              <span className="ml-1.5 rounded-full bg-muted px-1.5 py-0.5 font-mono text-[9.5px] tabular-nums text-muted-foreground">
                {(result?.outputs?.length ?? 0) + (result?.inputs?.length ?? 0)}
              </span>
            </TabTriggerSlim>
            <TabTriggerSlim value="raw">Raw</TabTriggerSlim>
          </TabsList>

          <TabsContent value="result" className="m-0 p-5">
            {isRunning ? (
              <PendingState />
            ) : primaryOutput ? (
              <div className="space-y-3">
                <div className="flex items-center gap-2">
                  <Sparkles className="h-3.5 w-3.5 text-accent" />
                  <h3 className="text-[11px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">
                    Primary output · {primaryOutput.type}
                  </h3>
                </div>
                <OutputPreview artifact={primaryOutput} />
              </div>
            ) : (
              <div className="rounded-md border border-dashed border-border bg-muted/20 px-4 py-8 text-center text-[12.5px] text-muted-foreground">
                No outputs were produced.
              </div>
            )}
          </TabsContent>

          <TabsContent value="artifacts" className="m-0 space-y-6 p-5">
            <ArtifactSection title="Outputs" artifacts={result?.outputs ?? []} pending={isRunning} emphasize />
            <ArtifactSection title="Inputs" artifacts={result?.inputs ?? []} pending={false} />
          </TabsContent>

          <TabsContent value="raw" className="m-0 p-5">
            <pre className="pretty-pre max-h-[480px] overflow-auto rounded-md border bg-card p-4 font-mono text-[12px] leading-relaxed shadow-soft">
              {JSON.stringify({ view, result }, null, 2)}
            </pre>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}

function pickPrimaryOutput(outputs: ArtifactView[] | undefined): ArtifactView | null {
  if (!outputs || outputs.length === 0) return null;
  const text = outputs.find((a) => /^(text\/|application\/json)/.test((a.contentType ?? "").toLowerCase()));
  return text ?? outputs[0];
}

function Stat({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="font-mono text-[10px] uppercase tracking-[0.16em] text-muted-foreground/80">
        {label}
      </span>
      <span>{children}</span>
    </div>
  );
}

function TabTriggerSlim({ value, children }: { value: string; children: React.ReactNode }) {
  return (
    <TabsTrigger
      value={value}
      className="relative h-auto rounded-none border-b-2 border-transparent bg-transparent px-3.5 py-2.5 text-[11.5px] font-medium uppercase tracking-[0.14em] text-muted-foreground shadow-none data-[state=active]:border-primary data-[state=active]:bg-transparent data-[state=active]:text-foreground data-[state=active]:shadow-none"
    >
      {children}
    </TabsTrigger>
  );
}

function PendingState() {
  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <Activity className="h-3.5 w-3.5 animate-pulse-soft text-warning" />
        <h3 className="text-[11px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">
          Worker running…
        </h3>
      </div>
      <div className="space-y-2">
        <div className="h-3 w-1/3 animate-pulse-soft rounded bg-muted" />
        <div className="h-3 w-5/6 animate-pulse-soft rounded bg-muted" />
        <div className="h-3 w-4/6 animate-pulse-soft rounded bg-muted" />
      </div>
      <p className="font-mono text-[10.5px] text-muted-foreground/70">
        polling every {(POLL_INTERVAL_MS / 1000).toFixed(1)}s · output appears when the job reaches a terminal state.
      </p>
    </div>
  );
}

function ArtifactSection({
  title,
  artifacts,
  emphasize,
  pending,
}: {
  title: string;
  artifacts: ArtifactView[];
  emphasize?: boolean;
  pending: boolean;
}) {
  return (
    <div>
      <div className="mb-2.5 flex items-baseline gap-2">
        <h3
          className={cn(
            "text-[11px] font-semibold uppercase tracking-[0.14em]",
            emphasize ? "text-primary" : "text-muted-foreground",
          )}
        >
          {title}
        </h3>
        <span className="font-mono text-[10.5px] tabular-nums text-muted-foreground/80">
          {artifacts.length}
        </span>
      </div>
      {pending ? (
        <div className="rounded-md border border-dashed border-border bg-muted/20 px-4 py-6 text-center text-[12.5px] text-muted-foreground">
          Waiting for job to finish…
        </div>
      ) : artifacts.length === 0 ? (
        <div className="rounded-md border border-dashed border-border bg-muted/20 px-4 py-5 text-center text-[12.5px] text-muted-foreground">
          No {title.toLowerCase()}.
        </div>
      ) : (
        <ul className="space-y-1.5">
          {artifacts.map((a) => (
            <li
              key={a.id}
              className="group flex flex-wrap items-center gap-3 rounded-md border border-border/70 bg-card px-3 py-2.5 transition-colors hover:border-border"
            >
              <div className="grid h-8 w-8 shrink-0 place-items-center rounded-md bg-muted/50 border border-border/60">
                <FileText className="h-3.5 w-3.5 text-muted-foreground" />
              </div>
              <div className="min-w-0 flex-1">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-[12.5px] font-medium tracking-snug">{a.type}</span>
                  <span className="rounded-full border border-border/60 bg-muted/40 px-1.5 py-0.5 font-mono text-[9.5px] uppercase tracking-[0.12em] text-muted-foreground">
                    {a.role}
                  </span>
                </div>
                <div className="mt-0.5 flex flex-wrap items-center gap-x-3 font-mono text-[10.5px] text-muted-foreground">
                  <span>{a.contentType ?? "?"}</span>
                  <span>·</span>
                  <span>{formatBytes(a.sizeBytes)}</span>
                  {a.checksumSha256 && (
                    <>
                      <span>·</span>
                      <span>{a.checksumSha256.slice(0, 12)}</span>
                    </>
                  )}
                </div>
              </div>
              <div className="flex items-center gap-1">
                <Button asChild variant="ghost" size="sm" className="h-7 gap-1.5 text-[11px]">
                  <a href={artifactUrl(a.accessUrl)} target="_blank" rel="noreferrer">
                    <ExternalLink className="h-3 w-3" />
                    Open
                  </a>
                </Button>
                <Button asChild variant="ghost" size="sm" className="h-7 gap-1.5 text-[11px]">
                  <a href={artifactUrl(a.accessUrl)} download>
                    <Download className="h-3 w-3" />
                    Download
                  </a>
                </Button>
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function EmptyState() {
  return (
    <Card className="overflow-hidden border-dashed border-border/60 bg-card/40 shadow-none">
      <CardContent className="grid place-items-center gap-3 px-6 py-16 text-center">
        <div className="grid h-12 w-12 place-items-center rounded-full border border-border/80 bg-card text-muted-foreground/80 shadow-soft">
          <Inbox className="h-5 w-5" />
        </div>
        <div className="space-y-1">
          <div className="text-[14px] font-semibold tracking-snug">No job selected</div>
          <p className="text-[12.5px] leading-relaxed text-muted-foreground">
            Submit a job above, or pick one from the recent list.
          </p>
        </div>
      </CardContent>
    </Card>
  );
}

function formatErr(e: unknown): string {
  if (e instanceof ApiError) {
    return typeof e.body === "object" && e.body ? `${e.body.code}: ${e.body.message}` : e.message;
  }
  return e instanceof Error ? e.message : String(e);
}
