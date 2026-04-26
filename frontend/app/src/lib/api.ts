import type { Capability, ErrorBody, JobCreated, JobResult, JobView } from "./types";

const STORAGE_KEY = "ai-pipeline.apiBase";
// Empty string = same-origin: requests go to /api/... on whatever host is
// serving the page. Works behind the docker-compose reverse proxy and with
// Vite's dev-server proxy. Override via the Settings popover when running
// the bundled HTML directly against a separate-origin backend.
const DEFAULT_BASE = "";

export function getApiBase(): string {
  if (typeof window === "undefined") return DEFAULT_BASE;
  const stored = window.localStorage.getItem(STORAGE_KEY);
  return stored == null ? DEFAULT_BASE : stored;
}

export function setApiBase(base: string) {
  window.localStorage.setItem(STORAGE_KEY, base.trim().replace(/\/$/, ""));
}

export class ApiError extends Error {
  constructor(public status: number, public body: ErrorBody | string | null) {
    super(typeof body === "object" && body ? `${body.code}: ${body.message}` : `HTTP ${status}`);
  }
}

async function parseError(r: Response): Promise<ErrorBody | string | null> {
  const text = await r.text().catch(() => "");
  if (!text) return null;
  try {
    return JSON.parse(text) as ErrorBody;
  } catch {
    return text;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(`${getApiBase()}${path}`, init);
  if (!r.ok) {
    throw new ApiError(r.status, await parseError(r));
  }
  return (await r.json()) as T;
}

export function submitTextJob(capability: Capability, text: string): Promise<JobCreated> {
  return request<JobCreated>("/api/v1/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ capability, text }),
  });
}

export function submitFileJob(capability: Capability, file: File, text: string): Promise<JobCreated> {
  const fd = new FormData();
  fd.append("capability", capability);
  fd.append("file", file);
  if (text.trim().length > 0) fd.append("text", text);
  return request<JobCreated>("/api/v1/jobs", { method: "POST", body: fd });
}

export function getJob(jobId: string): Promise<JobView> {
  return request<JobView>(`/api/v1/jobs/${encodeURIComponent(jobId)}`);
}

export function getJobResult(jobId: string): Promise<JobResult> {
  return request<JobResult>(`/api/v1/jobs/${encodeURIComponent(jobId)}/result`);
}

export function artifactUrl(accessUrl: string): string {
  return `${getApiBase()}${accessUrl}`;
}

const TEXT_LIKE = /^(text\/|application\/(json|x-ndjson|xml|yaml|x-yaml))/;

export function isTextLike(contentType: string | null | undefined): boolean {
  if (!contentType) return false;
  return TEXT_LIKE.test(contentType.toLowerCase());
}

export async function fetchArtifactText(accessUrl: string, maxBytes = 256_000): Promise<string> {
  const r = await fetch(`${getApiBase()}${accessUrl}`);
  if (!r.ok) throw new ApiError(r.status, await parseError(r));
  const blob = await r.blob();
  const slice = blob.size > maxBytes ? blob.slice(0, maxBytes) : blob;
  const text = await slice.text();
  return blob.size > maxBytes ? `${text}\n\n… (truncated, ${(blob.size / 1024).toFixed(1)} KB total)` : text;
}

export async function pingApi(): Promise<boolean> {
  try {
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), 2500);
    const r = await fetch(`${getApiBase()}/api/v1/jobs/__ping__`, { signal: ctrl.signal });
    clearTimeout(t);
    return r.status === 404 || r.ok;
  } catch {
    return false;
  }
}
