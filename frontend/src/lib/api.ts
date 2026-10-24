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
  rationale: string | null;
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

/** Mirrors backend/api/schemas.py TimelineItem. */
export type TimelineItem = {
  id: number;
  content: string;
  kind: string;
  certainty: string;
  source: string;
  source_ref: string | null;
  status: string;
  created_at: string;
  rationale: string | null;
};

export type TimelineDay = {
  date: string;
  items: TimelineItem[];
};

export type TaskActionResponse = {
  status: "confirmed" | "discarded";
  index: number;
  task_id: number | null;
};

/** Mirrors backend/api/schemas.py TaskOut. The tasks table has no created_at/completed_at. */
export type TaskOut = {
  id: number;
  title: string;
  due_at: string | null;
  status: string;
  origin: string;
};

export type OnboardingQuestion = {
  id: string;
  text: string;
};

export type OnboardingPrompt = {
  import_prompt: string;
  questions: OnboardingQuestion[];
};

/** A staged onboarding candidate. Mirrors services/onboarding.py CandidateOut. */
export type OnboardingCandidate = {
  record_id: number;
  content: string;
  kind: string;
  certainty: string;
};

export type OnboardingDropped = {
  count: number;
  invalid: number;
  secret: number;
  over_cap: number;
};

/** Mirrors services/onboarding.py OnboardingResult. */
export type OnboardingResult = {
  groups: Record<string, OnboardingCandidate[]>;
  pending_record_ids: number[];
  duplicates_skipped: number;
  dropped: OnboardingDropped;
  suggested_timezone: string | null;
  errors: string[];
  timings: Record<string, number>;
};

export type BatchConfirmResponse = {
  confirmed: number[];
  ignored: number[];
};

export type BatchRejectResponse = {
  rejected: number[];
  ignored: number[];
};

// ---------------------------------------------------------------------------
// Conversations (server-side chat history)
// Mirrors backend/api/schemas.py. Field caps are shared by the UI and the client.
// ---------------------------------------------------------------------------

export const CONVERSATION_TITLE_MAX = 120;
export const CONVERSATION_FIRST_MESSAGE_MAX = 8000;
export const CONVERSATION_MESSAGE_MAX = 8000;

/** List item. Only id, title and updated_at are exposed by the backend. */
export type ConversationOut = {
  id: number;
  title: string;
  updated_at: string;
};

/** One stored turn. `content` is untrusted text, rendered as plain text only. */
export type ChatMessageOut = {
  id: number;
  role: "user" | "assistant";
  content: string;
  created_at: string;
};

export type ConversationDetail = {
  id: number;
  title: string;
  created_at: string;
  updated_at: string;
  messages: ChatMessageOut[];
};

export type MessagePairResponse = {
  conversation_id: number;
  messages: ChatMessageOut[];
};

export type ConversationActionResponse = {
  id: number;
  status: string;
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

export function signup(
  username: string,
  password: string,
  inviteCode: string,
): Promise<LoginResponse> {
  return request<LoginResponse>("/auth/signup", {
    method: "POST",
    body: JSON.stringify({ username, password, invite_code: inviteCode }),
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

/**
 * GET /memory/timeline. Default status is "all" (everything except rejected), but this
 * app renders one status view at a time, so it always passes "confirmed" or "pending".
 */
export function getTimeline(
  status: "confirmed" | "pending",
  limit = 200,
): Promise<TimelineDay[]> {
  const params = new URLSearchParams({ status, limit: String(limit) });
  return request<TimelineDay[]>(`/memory/timeline?${params.toString()}`);
}

export function editMemory(recordId: number, content: string): Promise<RecordActionResponse> {
  return request<RecordActionResponse>(`/memory/${recordId}`, {
    method: "PATCH",
    body: JSON.stringify({ content }),
  });
}

export function deleteMemory(recordId: number): Promise<RecordActionResponse> {
  return request<RecordActionResponse>(`/memory/${recordId}?confirm=true`, {
    method: "DELETE",
  });
}

/** GET /tasks?status=open|done|all (default all). Ownership-scoped by the backend. */
export function getTasks(status: "open" | "done" | "all" = "all"): Promise<TaskOut[]> {
  const params = new URLSearchParams({ status });
  return request<TaskOut[]>(`/tasks?${params.toString()}`);
}

/** POST /tasks/{id}/done. Already-done tasks return their current state without error. */
export function markTaskDone(taskId: number): Promise<TaskOut> {
  return request<TaskOut>(`/tasks/${taskId}/done`, { method: "POST" });
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

// ---------------------------------------------------------------------------
// Conversations (server-side chat history)
// ---------------------------------------------------------------------------

/** GET /conversations. Newest first, ownership-scoped by the backend. */
export function listConversations(): Promise<ConversationOut[]> {
  return request<ConversationOut[]>("/conversations");
}

/** POST /conversations. The title is derived from the first message by the backend. */
export function createConversation(firstMessage: string): Promise<ConversationOut> {
  return request<ConversationOut>("/conversations", {
    method: "POST",
    body: JSON.stringify({ first_message: firstMessage }),
  });
}

/** GET /conversations/{id}. Returns the stored turns plus conversation metadata. */
export function getConversation(conversationId: number): Promise<ConversationDetail> {
  return request<ConversationDetail>(`/conversations/${conversationId}`);
}

/** POST /conversations/{id}/messages. Appends one user/assistant pair in one transaction. */
export function appendMessages(
  conversationId: number,
  user: string,
  assistant: string,
): Promise<MessagePairResponse> {
  return request<MessagePairResponse>(`/conversations/${conversationId}/messages`, {
    method: "POST",
    body: JSON.stringify({ user, assistant }),
  });
}

/** PATCH /conversations/{id}. Renames a conversation (title max 120 chars). */
export function renameConversation(
  conversationId: number,
  title: string,
): Promise<ConversationOut> {
  return request<ConversationOut>(`/conversations/${conversationId}`, {
    method: "PATCH",
    body: JSON.stringify({ title }),
  });
}

/**
 * DELETE /conversations/{id}. Destructive: confirm=true is sent only after the UI's own
 * confirmation step has been completed by the user.
 */
export function deleteConversation(conversationId: number): Promise<ConversationActionResponse> {
  return request<ConversationActionResponse>(`/conversations/${conversationId}?confirm=true`, {
    method: "DELETE",
  });
}

// ---------------------------------------------------------------------------
// Onboarding + batch memory transitions
// ---------------------------------------------------------------------------

export function getOnboardingPrompt(): Promise<OnboardingPrompt> {
  return request<OnboardingPrompt>("/onboarding/prompt");
}

/** Import pasted text. The backend rejects payloads over ONBOARDING_MAX_CHARS with a 413. */
export function importOnboarding(text: string): Promise<OnboardingResult> {
  return request<OnboardingResult>("/onboarding/import", {
    method: "POST",
    body: JSON.stringify({ text }),
  });
}

export function confirmBatch(recordIds: number[]): Promise<BatchConfirmResponse> {
  return request<BatchConfirmResponse>("/memory/confirm-batch", {
    method: "POST",
    body: JSON.stringify({ record_ids: recordIds }),
  });
}

export function rejectBatch(recordIds: number[]): Promise<BatchRejectResponse> {
  return request<BatchRejectResponse>("/memory/reject-batch", {
    method: "POST",
    body: JSON.stringify({ record_ids: recordIds }),
  });
}
