import { cn } from "@/lib/utils";
import { CheckCircle2, CircleAlert, CircleDashed, CircleSlash, Loader2 } from "lucide-react";

const VARIANTS: Record<
  string,
  { label: string; tone: "muted" | "warning" | "success" | "destructive"; Icon: typeof CheckCircle2; spin?: boolean }
> = {
  PENDING: { label: "대기", tone: "muted", Icon: CircleDashed },
  QUEUED: { label: "큐 대기", tone: "muted", Icon: CircleDashed },
  RUNNING: { label: "실행 중", tone: "warning", Icon: Loader2, spin: true },
  IN_PROGRESS: { label: "실행 중", tone: "warning", Icon: Loader2, spin: true },
  SUCCEEDED: { label: "성공", tone: "success", Icon: CheckCircle2 },
  COMPLETED: { label: "성공", tone: "success", Icon: CheckCircle2 },
  FAILED: { label: "실패", tone: "destructive", Icon: CircleAlert },
  CANCELED: { label: "취소됨", tone: "muted", Icon: CircleSlash },
  CANCELLED: { label: "취소됨", tone: "muted", Icon: CircleSlash },
};

const TONE_CLASS: Record<string, string> = {
  muted: "bg-muted text-muted-foreground border-border",
  warning: "bg-warning/12 text-warning border-warning/35",
  success: "bg-success/12 text-success border-success/35",
  destructive: "bg-destructive/12 text-destructive border-destructive/35",
};

interface StatusBadgeProps {
  status: string;
  size?: "sm" | "md";
  className?: string;
  withDot?: boolean;
}

export function StatusBadge({ status, size = "md", className, withDot = false }: StatusBadgeProps) {
  const key = status.toUpperCase();
  const v = VARIANTS[key] ?? { label: status, tone: "muted" as const, Icon: CircleDashed };
  const Icon = v.Icon;
  const tone = TONE_CLASS[v.tone];
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border font-medium tabular-nums tracking-tight",
        size === "sm" ? "px-2 py-0.5 text-[11px]" : "px-2.5 py-1 text-xs",
        tone,
        className,
      )}
    >
      {withDot ? (
        <span
          className={cn(
            "inline-block rounded-full",
            size === "sm" ? "h-1.5 w-1.5" : "h-2 w-2",
            v.tone === "success" && "bg-success",
            v.tone === "warning" && "bg-warning animate-pulse-soft",
            v.tone === "destructive" && "bg-destructive",
            v.tone === "muted" && "bg-muted-foreground/60",
          )}
        />
      ) : (
        <Icon className={cn(size === "sm" ? "h-3 w-3" : "h-3.5 w-3.5", v.spin && "animate-spin")} />
      )}
      {v.label}
    </span>
  );
}
