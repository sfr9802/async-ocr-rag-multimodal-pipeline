export function formatBytes(bytes: number | null | undefined): string {
  if (bytes == null) return "—";
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB"];
  let v = bytes / 1024;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v.toFixed(v < 10 ? 1 : 0)} ${units[i]}`;
}

export function formatRelative(iso: string | undefined | null): string {
  if (!iso) return "—";
  const t = new Date(iso).getTime();
  if (Number.isNaN(t)) return iso ?? "—";
  const diff = Math.floor((Date.now() - t) / 1000);
  if (diff < 5) return "방금";
  if (diff < 60) return `${diff}초 전`;
  if (diff < 3600) return `${Math.floor(diff / 60)}분 전`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}시간 전`;
  return `${Math.floor(diff / 86400)}일 전`;
}

export function formatDuration(fromIso: string | undefined | null, toIso: string | undefined | null): string {
  if (!fromIso || !toIso) return "—";
  const from = new Date(fromIso).getTime();
  const to = new Date(toIso).getTime();
  if (Number.isNaN(from) || Number.isNaN(to)) return "—";
  const ms = Math.max(0, to - from);
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(ms < 10_000 ? 2 : 1)}s`;
  const mins = Math.floor(ms / 60_000);
  const secs = Math.floor((ms % 60_000) / 1000);
  return `${mins}m ${secs}s`;
}

export function shortId(id: string, head = 8): string {
  if (!id) return "";
  if (id.length <= head + 4) return id;
  return `${id.slice(0, head)}…${id.slice(-4)}`;
}

export type DateBucket = "오늘" | "어제" | "이전";

export function bucketByDate<T extends { submittedAt: string }>(entries: T[]): Array<[DateBucket, T[]]> {
  const buckets: Record<DateBucket, T[]> = { 오늘: [], 어제: [], 이전: [] };
  const now = new Date();
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  const yesterday = today - 86_400_000;
  for (const e of entries) {
    const t = new Date(e.submittedAt).getTime();
    if (Number.isNaN(t)) {
      buckets.이전.push(e);
      continue;
    }
    if (t >= today) buckets.오늘.push(e);
    else if (t >= yesterday) buckets.어제.push(e);
    else buckets.이전.push(e);
  }
  const out: Array<[DateBucket, T[]]> = [];
  (Object.keys(buckets) as DateBucket[]).forEach((k) => {
    if (buckets[k].length > 0) out.push([k, buckets[k]]);
  });
  return out;
}

export function tryPrettyJson(text: string): { pretty: string; isJson: boolean } {
  try {
    const parsed = JSON.parse(text);
    return { pretty: JSON.stringify(parsed, null, 2), isJson: true };
  } catch {
    return { pretty: text, isJson: false };
  }
}
