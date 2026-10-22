/**
 * Typed client for the FastAPI backend.
 *
 * Same-origin only: every call goes to "/api/..." and is forwarded to the backend by the
 * rewrite in next.config.ts. The backend origin is never sent to the browser.
 *
 * Types mirror backend/api/schemas.py, backend/services/orchestrator.py and
 * backend/services/intervention.py. Error text from the server is never surfaced to the user:
 * an AuthError carries no message of its own, and ApiError carries the status code only.
 */

export class AuthError extends Error {
  constructor() {
    super("unauthorized");
    this.name = "AuthError";
  }
}

export class ApiError extends Error {
  readonly status: number;

  constructor(status: number) {
    super(`request failed with status ${status}`);
    this.name = "ApiError";
    this.status = status;
  }
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type LoginResponse = {
  ok: boolean;
  user_id: number;
  username: string | null;
};

export type LogoutResponse = {
  ok: boolean;
};

export type Me = {
  user_id: number;
  username: string | null;
  timezone: string;
  quiet_mode: boolean;
};

export type SessionNewResponse = {
  session_id: number;
};

export type HistoryMessage = {
  role: "user" | "assistant";
  content: string;
};

export type ChatRequest = {
  text: string;
  session_id: number;
  history: HistoryMessage[];
  source: "text" | "voice";
};

/** `confidence` is the relevance judge's 0-100 score, not a probability. */
export type MemoryUsedItem = {
  record_id: number;
  content: string;
  kind: string;
  certainty: string;
  source: string;
  created_at: string | null;
  confidence: number;
};

export type TraceInfo = {
  timings: Record<string, number>;
  retrieval_ran: boolean;
  candidates_considered: number;
  retrieval_timed_out: boolean;
  retrieval_embedding_failed: boolean;
  errors: string[];
  skipped: boolean;
};

export type ChatResponse = {
  turn_id: string;
  session_id: number;
  intent: string | null;
  reply: string;
  memory_used: MemoryUsedItem[];
  trace: TraceInfo;
};

export type PendingRecord = {
  id: number;
  content: string;
  kind: string;
  certainty: string;
  source: string;
  status: string;
};

export type PendingTaskOut = {
  index: number;
  title: string;
  due_at: string | null;
  origin: string;
};

export type ConflictRef = {
  kind: "task" | "goal" | "record";
  id: number;
};

export type ActionDirective = {
  type: "mark_done" | "snooze" | "reschedule";
  task_id: number;
  new_due_at: string | null;
};

export type InterventionDecision = {
  action: "none" | "remind" | "surface" | "suggest" | "act";
  message: string | null;
  contradiction: boolean;
  contradiction_detail: string | null;
  conflicts_with: ConflictRef | null;
  clarifying_question: string | null;
  target_task_id: number | null;
  surfaced_record_id: number | null;
  act: ActionDirective | null;
  requires_confirmation: boolean;
  reason: string;
};

export type FollowupResponse = {
  status: "pending" | "ready";
  pending_records: PendingRecord[];
  pending_tasks: PendingTaskOut[];
  intervention: InterventionDecision;
  extraction_failed: boolean;
  trace: TraceInfo;
};

export type RecordActionResponse = {
  id: number;
  status: string;
};

export type TaskActionResponse = {
  status: "confirmed" | "discarded";
  index: number;
  task_id: number | null;
};

// ---------------------------------------------------------------------------
// Transport
// ---------------------------------------------------------------------------

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, {
    ...init,
    credentials: "same-origin",
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });

  if (response.status === 401) {
    throw new AuthError();
  }
  if (!response.ok) {
    throw new ApiError(response.status);
  }
  if (response.status === 204) {
    return undefined as T;
  }
  return (await response.json()) as T;
}

// ---------------------------------------------------------------------------
// Endpoints
// ---------------------------------------------------------------------------

export function login(username: string, password: string): Promise<LoginResponse> {
  return request<LoginResponse>("/auth/login", {
    method: "POST",
    body: JSON.stringify({ username, password }),
  });
}

export function logout(): Promise<LogoutResponse> {
  return request<LogoutResponse>("/auth/logout", { method: "POST" });
}

export function getMe(): Promise<Me> {
  return request<Me>("/me");
}

export function newSession(): Promise<SessionNewResponse> {
  return request<SessionNewResponse>("/session/new", { method: "POST" });
}

export function chat(payload: ChatRequest): Promise<ChatResponse> {
  return request<ChatResponse>("/chat", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function getFollowup(turnId: string): Promise<FollowupResponse> {
  return request<FollowupResponse>(`/chat/${encodeURIComponent(turnId)}/followup`);
}

export function confirmRecord(recordId: number): Promise<RecordActionResponse> {
  return request<RecordActionResponse>(`/memory/${recordId}/confirm`, { method: "POST" });
}

export function rejectRecord(recordId: number): Promise<RecordActionResponse> {
  return request<RecordActionResponse>(`/memory/${recordId}/reject`, { method: "POST" });
}

export function confirmTask(turnId: string, index: number): Promise<TaskActionResponse> {
  return request<TaskActionResponse>(
    `/chat/${encodeURIComponent(turnId)}/tasks/${index}/confirm`,
    { method: "POST" },
  );
}

export function discardTask(turnId: string, index: number): Promise<TaskActionResponse> {
  return request<TaskActionResponse>(
    `/chat/${encodeURIComponent(turnId)}/tasks/${index}/discard`,
    { method: "POST" },
  );
}
