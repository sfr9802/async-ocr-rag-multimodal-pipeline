import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Label } from "@/components/ui/label";
import { Moon, Settings2, Sun } from "lucide-react";
import { getApiBase, pingApi, setApiBase } from "@/lib/api";
import { cn } from "@/lib/utils";

function Logo() {
  return (
    <svg
      viewBox="0 0 32 32"
      width="28"
      height="28"
      aria-hidden="true"
      className="text-primary"
    >
      <rect x="0.75" y="0.75" width="30.5" height="30.5" rx="7.25" fill="currentColor" opacity="0.12" />
      <rect x="0.75" y="0.75" width="30.5" height="30.5" rx="7.25" fill="none" stroke="currentColor" strokeOpacity="0.35" strokeWidth="0.75" />
      <path d="M9 22 L13 10 L19 10 L23 22" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" fill="none" />
      <path d="M11 18 L21 18" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
      <circle cx="23" cy="22" r="2.4" fill="hsl(var(--accent))" />
    </svg>
  );
}

export function Header() {
  const [dark, setDark] = useState<boolean>(() =>
    typeof window !== "undefined" && document.documentElement.classList.contains("dark"),
  );
  const [base, setBase] = useState<string>(getApiBase());
  const [draftBase, setDraftBase] = useState<string>(base);
  const [online, setOnline] = useState<boolean | null>(null);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
  }, [dark]);

  useEffect(() => {
    let cancelled = false;
    const check = async () => {
      const ok = await pingApi();
      if (!cancelled) setOnline(ok);
    };
    check();
    const id = window.setInterval(check, 8000);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [base]);

  function applyBase() {
    setApiBase(draftBase);
    setBase(draftBase);
  }

  return (
    <header className="sticky top-0 z-20 border-b border-border/80 bg-background/85 backdrop-blur supports-[backdrop-filter]:bg-background/65">
      <div className="mx-auto flex max-w-[1380px] items-center gap-3 px-5 py-2.5">
        <div className="flex items-center gap-2.5">
          <Logo />
          <div className="leading-tight">
            <div className="text-[13.5px] font-semibold tracking-snug">AI 파이프라인 콘솔</div>
            <div className="font-mono text-[10.5px] uppercase tracking-[0.14em] text-muted-foreground">
              async · ocr · rag · multimodal
            </div>
          </div>
        </div>

        <div className="ml-auto flex items-center gap-2">
          <div
            className={cn(
              "hidden items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] font-medium sm:inline-flex",
              online === null
                ? "border-border text-muted-foreground"
                : online
                  ? "border-success/35 bg-success/10 text-success"
                  : "border-destructive/40 bg-destructive/10 text-destructive",
            )}
            title={base}
          >
            <span
              className={cn(
                "h-1.5 w-1.5 rounded-full",
                online === null
                  ? "bg-muted-foreground/60 animate-pulse-soft"
                  : online
                    ? "bg-success"
                    : "bg-destructive",
              )}
            />
            <span className="font-mono uppercase tracking-[0.12em]">
              {online === null ? "확인 중" : online ? "core-api 온라인" : "오프라인"}
            </span>
          </div>

          <Popover>
            <PopoverTrigger asChild>
              <Button variant="ghost" size="icon" className="h-8 w-8" aria-label="설정">
                <Settings2 className="h-4 w-4" />
              </Button>
            </PopoverTrigger>
            <PopoverContent align="end" className="w-80">
              <div className="space-y-3">
                <div className="space-y-1">
                  <h3 className="text-sm font-semibold tracking-snug">API 엔드포인트</h3>
                  <p className="text-xs leading-relaxed text-muted-foreground">
                    비워두면 같은 origin의 <span className="font-mono">/api</span>를 사용합니다
                    (compose 리버스 프록시 및 <span className="font-mono">pnpm dev</span>에서 작동).
                    별도 origin 백엔드를 가리키려면 절대 URL을 입력하세요.
                  </p>
                </div>
                <div className="space-y-1.5">
                  <Label htmlFor="api-base" className="text-[11px] uppercase tracking-wider text-muted-foreground">
                    Base URL
                  </Label>
                  <Input
                    id="api-base"
                    value={draftBase}
                    onChange={(e) => setDraftBase(e.target.value)}
                    placeholder="(같은 origin)"
                    className="font-mono text-xs"
                  />
                </div>
                <Button size="sm" className="w-full" onClick={applyBase}>
                  저장
                </Button>
              </div>
            </PopoverContent>
          </Popover>

          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8"
            aria-label="테마 전환"
            onClick={() => setDark((d) => !d)}
          >
            {dark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
          </Button>
        </div>
      </div>
    </header>
  );
}
