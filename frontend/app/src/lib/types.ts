export type Capability = "MOCK" | "OCR" | "RAG" | "MULTIMODAL";

export const CAPABILITIES: Capability[] = ["MOCK", "OCR", "RAG", "MULTIMODAL"];

export interface ArtifactView {
  id: string;
  role: "INPUT" | "OUTPUT" | string;
  type: string;
  contentType: string | null;
  sizeBytes: number | null;
  checksumSha256: string | null;
  accessUrl: string;
}

export interface JobCreated {
  jobId: string;
  status: string;
  capability: Capability;
  inputs: ArtifactView[];
}

export interface JobView {
  jobId: string;
  capability: Capability;
  status: string;
  attemptNo: number;
  errorCode: string | null;
  errorMessage: string | null;
  createdAt: string;
  claimedAt: string | null;
  updatedAt: string;
}

export interface JobEvent {
  status: string;
  at: string;
  source: "server" | "client";
}

export interface JobResult {
  jobId: string;
  status: string;
  inputs: ArtifactView[];
  outputs: ArtifactView[];
  errorCode: string | null;
  errorMessage: string | null;
}

export interface ErrorBody {
  code: string;
  message: string;
}

export const TERMINAL_STATUSES = new Set(["SUCCEEDED", "FAILED", "CANCELED", "CANCELLED"]);

export function isTerminal(status: string | undefined | null): boolean {
  if (!status) return false;
  return TERMINAL_STATUSES.has(status.toUpperCase());
}
