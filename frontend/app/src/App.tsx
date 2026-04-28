import { useCallback, useEffect, useState } from "react";
import { Header } from "@/components/Header";
import { JobPanel } from "@/components/JobPanel";
import { JobSidebar } from "@/components/JobSidebar";
import { SubmitForm } from "@/components/SubmitForm";
import {
  addHistoryEntry,
  type HistoryEntry,
  loadHistory,
  removeHistoryEntry,
  saveHistory,
} from "@/lib/history";
import type { JobCreated } from "@/lib/types";

function App() {
  const [history, setHistory] = useState<HistoryEntry[]>(() => loadHistory());
  const [activeJobId, setActiveJobId] = useState<string | null>(() => loadHistory()[0]?.jobId ?? null);
  const [statusByJob, setStatusByJob] = useState<Record<string, string | undefined>>({});

  useEffect(() => {
    saveHistory(history);
  }, [history]);

  function handleSubmitted(job: JobCreated, preview: string) {
    const entry: HistoryEntry = {
      jobId: job.jobId,
      capability: job.capability,
      submittedAt: new Date().toISOString(),
      preview: preview.slice(0, 140),
    };
    setHistory((cur) => addHistoryEntry(entry, cur));
    setActiveJobId(job.jobId);
    setStatusByJob((s) => ({ ...s, [job.jobId]: job.status }));
  }

  const handleStatusChange = useCallback((jobId: string, status: string) => {
    setStatusByJob((s) => (s[jobId] === status ? s : { ...s, [jobId]: status }));
  }, []);

  function handleSelect(jobId: string) {
    setActiveJobId(jobId);
  }

  function handleRemove(jobId: string) {
    setHistory((cur) => {
      const next = removeHistoryEntry(jobId, cur);
      if (activeJobId === jobId) {
        setActiveJobId(next[0]?.jobId ?? null);
      }
      return next;
    });
  }

  function handleClear() {
    setHistory([]);
    setActiveJobId(null);
    setStatusByJob({});
  }

  return (
    <div className="canvas-bg relative flex min-h-screen flex-col overflow-hidden">
      <div className="glass-aurora" aria-hidden="true" />
      <div className="relative z-10 flex min-h-screen flex-col">
        <Header />
        <div className="flex-1">
          <div className="mx-auto grid w-full max-w-[1440px] grid-cols-1 gap-5 px-5 py-5 lg:grid-cols-[20rem_1fr] lg:gap-6 lg:px-7 lg:py-6">
            <JobSidebar
              history={history}
              statusByJob={statusByJob}
              activeJobId={activeJobId}
              onSelect={handleSelect}
              onRemove={handleRemove}
              onClear={handleClear}
            />
            <main className="flex min-w-0 flex-col gap-5 lg:gap-6">
              <SubmitForm onSubmitted={handleSubmitted} />
              <JobPanel jobId={activeJobId} onStatusChange={handleStatusChange} />
            </main>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
