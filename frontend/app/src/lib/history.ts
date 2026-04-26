import type { Capability } from "./types";

const KEY = "ai-pipeline.jobHistory";
const MAX = 25;

export interface HistoryEntry {
  jobId: string;
  capability: Capability;
  submittedAt: string;
  preview: string;
}

export function loadHistory(): HistoryEntry[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? (parsed as HistoryEntry[]) : [];
  } catch {
    return [];
  }
}

export function saveHistory(entries: HistoryEntry[]) {
  window.localStorage.setItem(KEY, JSON.stringify(entries.slice(0, MAX)));
}

export function addHistoryEntry(entry: HistoryEntry, current: HistoryEntry[]): HistoryEntry[] {
  const next = [entry, ...current.filter((e) => e.jobId !== entry.jobId)];
  return next.slice(0, MAX);
}

export function removeHistoryEntry(jobId: string, current: HistoryEntry[]): HistoryEntry[] {
  return current.filter((e) => e.jobId !== jobId);
}
