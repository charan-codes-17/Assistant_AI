"use client";

import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type FormEvent,
  type KeyboardEvent as ReactKeyboardEvent,
} from "react";
import { useRouter } from "next/navigation";

import { MessageExtras, type RecordState, type TaskState } from "@/components/message-extras";
import { ChatSidebar } from "@/components/chat-sidebar";
import {
  ApiError,
  AuthError,
  appendMessages,
  chat,
  confirmRecord,
  confirmTask,
  createConversation,
  deleteConversation,
  discardTask,
  getConversation,
  getFollowup,
  getMe,
  listConversations,
  logout,
  newSession,
  rejectRecord,
  renameConversation,
  CONVERSATION_FIRST_MESSAGE_MAX,
  CONVERSATION_MESSAGE_MAX,
  CONVERSATION_TITLE_MAX,
  type ConversationOut,
  type HistoryMessage,
  type Me,
  type MemoryUsedItem,
  type TraceInfo,
} from "@/lib/api";

const BANNER = "Demo only: please don't send real personal data.";
const GENERIC_ERROR = "Something went wrong. Please try again.";
const HISTORY_MAX_ITEMS = 10;
const HISTORY_CONTENT_MAX = 2000;
const POLL_INTERVAL_MS = 1000;
const POLL_TIMEOUT_MS = 30000;

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  turnId: string | null;
  memoryUsed: MemoryUsedItem[];
  trace: TraceInfo | null;
  followup: import("@/lib/api").FollowupResponse | null;
  recordState: Record<number, RecordState>;
  taskState: Record<number, TaskState>;
  // Server-side history save state for this exchange. Only the assistant message carries it.
  saveStatus: "pending" | "saved" | "failed";
  // True only for messages appended live in this session, so restored history does not animate in.
  live: boolean;
};

export default function ChatPage() {
  const router = useRouter();
  const [me, setMe] = useState<Me | null>(null);
  const [sessionId, setSessionId] = useState<number | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [busyItem, setBusyItem] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Sidebar / server-side history.
  const [conversations, setConversations] = useState<ConversationOut[]>([]);
  const [conversationsLoading, setConversationsLoading] = useState(true);
  const [conversationsError, setConversationsError] = useState(false);
  const [activeConversationId, setActiveConversationId] = useState<number | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [collapsed, setCollapsed] = useState(false);

  // Bumped on unmount-less resets (new session / logout): any poller from an older generation
  // stops as soon as it notices the change.
  const generationRef = useRef(0);
  const timersRef = useRef<Set<ReturnType<typeof setTimeout>>>(new Set());
  const idRef = useRef(0);
  const endRef = useRef<HTMLDivElement>(null);
  // The single composer: one textarea + one form shared by the empty and chat states.
  const composerRef = useRef<HTMLTextAreaElement>(null);
  const formRef = useRef<HTMLFormElement>(null);
  // Per-message copy feedback, keyed by message id; cleared on a timer.
  const [copyState, setCopyState] = useState<Record<string, "copied" | "failed">>({});
  const copyTimersRef = useRef<Set<ReturnType<typeof setTimeout>>>(new Set());
  // Latest active conversation id, readable from background saves without stale closures.
  const activeConversationIdRef = useRef<number | null>(null);

  const activateConversation = useCallback((id: number | null) => {
    activeConversationIdRef.current = id;
    setActiveConversationId(id);
  }, []);

  const refreshConversations = useCallback(
    async (showLoading: boolean) => {
      if (showLoading) {
        setConversationsLoading(true);
      }
      try {
        const list = await listConversations();
        setConversations(list);
        setConversationsError(false);
      } catch (err) {
        if (err instanceof AuthError) {
          router.replace("/login");
          return;
        }
        setConversationsError(true);
      } finally {
        if (showLoading) {
          setConversationsLoading(false);
        }
      }
    },
    [router],
  );

  // ---------------------------------------------------------------------------
  // Auth bootstrap
  // ---------------------------------------------------------------------------

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const profile = await getMe();
        if (cancelled) {
          return;
        }
        setMe(profile);
        const session = await newSession();
        if (cancelled) {
          return;
        }
        setSessionId(session.session_id);
      } catch (err) {
        if (cancelled) {
          return;
        }
        if (err instanceof AuthError) {
          router.replace("/login");
          return;
        }
        setError(GENERIC_ERROR);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [router]);

  // Load the stored chat list once on mount. An unauthenticated call redirects to /login.
  useEffect(() => {
    void refreshConversations(true);
  }, [refreshConversations]);

  // ---------------------------------------------------------------------------
  // Followup polling
  // ---------------------------------------------------------------------------

  const stopPolling = useCallback(() => {
    timersRef.current.forEach((timer) => clearTimeout(timer));
    timersRef.current.clear();
  }, []);

  useEffect(() => {
    const timers = timersRef.current;
    return () => {
      timers.forEach((timer) => clearTimeout(timer));
      timers.clear();
    };
  }, []);

  const startPolling = useCallback(
    (turnId: string) => {
      const generation = generationRef.current;
      const deadline = Date.now() + POLL_TIMEOUT_MS;

      const tick = async () => {
        if (generation !== generationRef.current) {
          return;
        }
        if (Date.now() >= deadline) {
          // Timed out: show nothing at all, not an error.
          return;
        }
        try {
          const followup = await getFollowup(turnId);
          if (generation !== generationRef.current) {
            return;
          }
          if (followup.status === "ready") {
            setMessages((prev) =>
              prev.map((message) => (message.turnId === turnId ? { ...message, followup } : message)),
            );
            return;
          }
        } catch (err) {
          if (generation !== generationRef.current) {
            return;
          }
          if (err instanceof AuthError) {
            router.replace("/login");
            return;
          }
          // Any other failure: keep polling quietly until the deadline.
        }
        if (generation !== generationRef.current) {
          return;
        }
        const timer = setTimeout(tick, POLL_INTERVAL_MS);
        timersRef.current.add(timer);
      };

      const timer = setTimeout(tick, POLL_INTERVAL_MS);
      timersRef.current.add(timer);
    },
    [router],
  );

  useEffect(() => {
    endRef.current?.scrollIntoView({ block: "end" });
  }, [messages]);

  // Auto-grow the composer up to about six lines, then let it scroll internally.
  useEffect(() => {
    const el = composerRef.current;
    if (!el) {
      return;
    }
    const LINE_HEIGHT = 24;
    const MAX_HEIGHT = LINE_HEIGHT * 6 + 8;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, MAX_HEIGHT)}px`;
  }, [input]);

  // Focus the composer on mount and again every time a reply finishes (sending -> false).
  useEffect(() => {
    if (!sending) {
      composerRef.current?.focus();
    }
  }, [sending]);

  // Global shortcuts: paste into the composer when nothing editable is focused; recover focus
  // with Enter when the page body is focused. Buttons and links keep their native Enter.
  useEffect(() => {
    function isEditable(target: EventTarget | null): boolean {
      if (!(target instanceof HTMLElement)) {
        return false;
      }
      const tag = target.tagName;
      return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || target.isContentEditable;
    }

    function onPaste(event: ClipboardEvent) {
      if (isEditable(event.target)) {
        return;
      }
      event.preventDefault();
      const pasted = event.clipboardData?.getData("text") ?? "";
      if (pasted) {
        setInput((prev) => prev + pasted);
      }
      composerRef.current?.focus();
    }

    function onKeyDown(event: KeyboardEvent) {
      if (event.key !== "Enter") {
        return;
      }
      if (event.target !== document.body) {
        return;
      }
      composerRef.current?.focus();
    }

    document.addEventListener("paste", onPaste);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("paste", onPaste);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, []);

  // Clear any pending copy-feedback timers on unmount.
  useEffect(() => {
    const timers = copyTimersRef.current;
    return () => {
      timers.forEach((timer) => clearTimeout(timer));
      timers.clear();
    };
  }, []);

  // ---------------------------------------------------------------------------
  // Actions
  // ---------------------------------------------------------------------------

  /**
   * Persist one completed exchange. Runs in the background after the reply is already on screen:
   * a failure never blocks or breaks chat, it only marks the assistant message as "Not saved"
   * after one retry. Content is truncated to the backend caps before it is sent.
   */
  const saveExchange = useCallback(
    async (
      userText: string,
      assistantText: string,
      assistantMessageId: string,
      generation: number,
    ) => {
      const user = userText.slice(0, CONVERSATION_MESSAGE_MAX);
      const assistant = assistantText.slice(0, CONVERSATION_MESSAGE_MAX);
      let conversationId = activeConversationIdRef.current;

      const run = async () => {
        if (conversationId === null) {
          const created = await createConversation(
            user.slice(0, CONVERSATION_FIRST_MESSAGE_MAX),
          );
          conversationId = created.id;
          // Only claim the active slot if the user has not started a new chat meanwhile.
          if (generation === generationRef.current) {
            activateConversation(created.id);
          }
        }
        await appendMessages(conversationId, user, assistant);
      };

      const mark = (status: "saved" | "failed") => {
        setMessages((prev) =>
          prev.map((message) =>
            message.id === assistantMessageId ? { ...message, saveStatus: status } : message,
          ),
        );
      };

      try {
        await run();
        mark("saved");
        void refreshConversations(false);
      } catch (err) {
        if (err instanceof AuthError) {
          router.replace("/login");
          return;
        }
        try {
          await run();
          mark("saved");
          void refreshConversations(false);
        } catch (retryErr) {
          if (retryErr instanceof AuthError) {
            router.replace("/login");
            return;
          }
          mark("failed");
        }
      }
    },
    [activateConversation, refreshConversations, router],
  );

  const handleSend = useCallback(
    async (event: FormEvent<HTMLFormElement>) => {
      event.preventDefault();
      const text = input.trim();
      if (!text || sending || sessionId === null) {
        return;
      }

      const history: HistoryMessage[] = messages.slice(-HISTORY_MAX_ITEMS).map((message) => ({
        role: message.role,
        content: message.content.slice(0, HISTORY_CONTENT_MAX),
      }));
      const generation = generationRef.current;

      idRef.current += 1;
      const userMessage: Message = {
        id: `local-${idRef.current}`,
        role: "user",
        content: text,
        turnId: null,
        memoryUsed: [],
        trace: null,
        followup: null,
        recordState: {},
        taskState: {},
        saveStatus: "saved",
        live: true,
      };

      setMessages((prev) => [...prev, userMessage]);
      setInput("");
      setSending(true);
      setError(null);

      try {
        const response = await chat({
          text,
          session_id: sessionId,
          history,
          source: "text",
        });
        setMessages((prev) => [
          ...prev,
          {
            id: response.turn_id,
            role: "assistant",
            content: response.reply,
            turnId: response.turn_id,
            memoryUsed: response.memory_used,
            trace: response.trace,
            followup: null,
            recordState: {},
            taskState: {},
            saveStatus: "pending",
            live: true,
          },
        ]);
        startPolling(response.turn_id);
        void saveExchange(text, response.reply, response.turn_id, generation);
      } catch (err) {
        if (err instanceof AuthError) {
          router.replace("/login");
          return;
        }
        setError(GENERIC_ERROR);
      } finally {
        setSending(false);
      }
    },
    [input, sending, sessionId, messages, router, startPolling, saveExchange],
  );

  const handleNewSession = useCallback(async () => {
    stopPolling();
    generationRef.current += 1;
    setMessages([]);
    setInput("");
    setError(null);
    activateConversation(null);
    composerRef.current?.focus();
    try {
      const session = await newSession();
      setSessionId(session.session_id);
    } catch (err) {
      if (err instanceof AuthError) {
        router.replace("/login");
        return;
      }
      setError(GENERIC_ERROR);
    }
  }, [activateConversation, router, stopPolling]);

  // "New chat": clear the visible messages and detach the active conversation so the next message
  // starts a new one. It does not touch any stored chat or memory (and keeps the session).
  const handleNewChat = useCallback(() => {
    stopPolling();
    generationRef.current += 1;
    setMessages([]);
    setInput("");
    setError(null);
    activateConversation(null);
    composerRef.current?.focus();
  }, [activateConversation, stopPolling]);

  const handleSelectConversation = useCallback(
    async (id: number) => {
      stopPolling();
      generationRef.current += 1;
      setError(null);
      try {
        const detail = await getConversation(id);
        setMessages(
          detail.messages.map((message) => ({
            id: `stored-${message.id}`,
            role: message.role,
            content: message.content,
            turnId: null,
            memoryUsed: [],
            trace: null,
            followup: null,
            recordState: {},
            taskState: {},
            saveStatus: "saved" as const,
            live: false,
          })),
        );
        activateConversation(id);
        composerRef.current?.focus();
      } catch (err) {
        if (err instanceof AuthError) {
          router.replace("/login");
          return;
        }
        setError(GENERIC_ERROR);
      }
    },
    [activateConversation, router, stopPolling],
  );

  const handleRenameConversation = useCallback(
    async (id: number, title: string) => {
      try {
        const updated = await renameConversation(id, title.slice(0, CONVERSATION_TITLE_MAX));
        setConversations((prev) => prev.map((item) => (item.id === id ? updated : item)));
      } catch (err) {
        if (err instanceof AuthError) {
          router.replace("/login");
        }
        throw err;
      }
    },
    [router],
  );

  const handleDeleteConversation = useCallback(
    async (id: number) => {
      try {
        await deleteConversation(id);
        setConversations((prev) => prev.filter((item) => item.id !== id));
        if (activeConversationIdRef.current === id) {
          stopPolling();
          generationRef.current += 1;
          setMessages([]);
          setInput("");
          activateConversation(null);
        }
      } catch (err) {
        if (err instanceof AuthError) {
          router.replace("/login");
        }
        throw err;
      }
    },
    [activateConversation, router, stopPolling],
  );

  const handleLogout = useCallback(async () => {
    stopPolling();
    generationRef.current += 1;
    try {
      await logout();
    } catch (err) {
      // The cookie is cleared server-side when possible; any failure still leaves this page.
      void err;
    }
    router.replace("/login");
  }, [router, stopPolling]);

  const handleRecord = useCallback(
    async (turnId: string, recordId: number, action: "save" | "discard") => {
      setBusyItem(true);
      try {
        if (action === "save") {
          await confirmRecord(recordId);
        } else {
          await rejectRecord(recordId);
        }
        const state: RecordState = action === "save" ? "saved" : "discarded";
        setMessages((prev) =>
          prev.map((message) =>
            message.turnId === turnId
              ? { ...message, recordState: { ...message.recordState, [recordId]: state } }
              : message,
          ),
        );
      } catch (err) {
        if (err instanceof AuthError) {
          router.replace("/login");
          return;
        }
        setError(GENERIC_ERROR);
      } finally {
        setBusyItem(false);
      }
    },
    [router],
  );

  const handleTask = useCallback(
    async (turnId: string, index: number, action: "save" | "discard") => {
      setBusyItem(true);
      try {
        if (action === "save") {
          await confirmTask(turnId, index);
        } else {
          await discardTask(turnId, index);
        }
        const state: TaskState = action === "save" ? "confirmed" : "discarded";
        setMessages((prev) =>
          prev.map((message) =>
            message.turnId === turnId
              ? { ...message, taskState: { ...message.taskState, [index]: state } }
              : message,
          ),
        );
      } catch (err) {
        if (err instanceof AuthError) {
          router.replace("/login");
          return;
        }
        if (err instanceof ApiError) {
          setError(GENERIC_ERROR);
          return;
        }
        setError(GENERIC_ERROR);
      } finally {
        setBusyItem(false);
      }
    },
    [router],
  );

  const disabled = sending || sessionId === null;

  // Enter sends (through the existing handler); Shift+Enter adds a newline; empty, busy and
  // IME composition keys are left to the textarea.
  const handleComposerKeyDown = useCallback(
    (event: ReactKeyboardEvent<HTMLTextAreaElement>) => {
      if (event.key !== "Enter" || event.shiftKey || event.nativeEvent.isComposing) {
        return;
      }
      if (disabled || input.trim().length === 0) {
        return;
      }
      event.preventDefault();
      formRef.current?.requestSubmit();
    },
    [disabled, input],
  );

  const handleCopy = useCallback(async (id: string, text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopyState((prev) => ({ ...prev, [id]: "copied" }));
    } catch {
      setCopyState((prev) => ({ ...prev, [id]: "failed" }));
    }
    const timer = setTimeout(() => {
      setCopyState((prev) => {
        const next = { ...prev };
        delete next[id];
        return next;
      });
    }, 1500);
    copyTimersRef.current.add(timer);
  }, []);

  return (
    <div className="flex min-h-screen bg-black text-neutral-200">
      <ChatSidebar
        conversations={conversations}
        activeId={activeConversationId}
        loading={conversationsLoading}
        error={conversationsError}
        open={sidebarOpen}
        onOpenChange={setSidebarOpen}
        onRetry={() => void refreshConversations(true)}
        onSelect={(id) => void handleSelectConversation(id)}
        onNewChat={handleNewSession}
        onRename={handleRenameConversation}
        onDelete={handleDeleteConversation}
        username={me?.username ?? null}
        onLogout={handleLogout}
        collapsed={collapsed}
        onCollapse={() => setCollapsed(true)}
      />

      <div className="flex min-h-screen flex-1 flex-col bg-black">
        <div className="border-b border-amber-800/60 bg-amber-950/40 px-4 py-2 text-center text-xs text-amber-300">
          {BANNER}
        </div>

        <div className="flex items-center border-b border-neutral-800 px-3 py-2 md:hidden">
          <button
            type="button"
            onClick={() => setSidebarOpen(true)}
            aria-label="Open chats"
            className="rounded-lg border border-neutral-800 bg-neutral-900 px-3 py-1.5 text-xs font-medium text-neutral-300 transition-[background-color,border-color,transform] duration-[var(--dur-hover)] ease-[var(--ease-out)] hover:bg-neutral-800 hover:border-neutral-700 active:scale-[0.97] active:duration-[var(--dur-press)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 focus-visible:ring-offset-1 focus-visible:ring-offset-black"
          >
            Menu
          </button>
        </div>

        {collapsed ? (
          <div className="hidden border-b border-neutral-800 px-3 py-2 md:block">
            <button
              type="button"
              onClick={() => setCollapsed(false)}
              aria-label="Open sidebar"
              className="rounded-lg border border-neutral-800 bg-neutral-900 px-2.5 py-1.5 text-xs font-medium text-neutral-300 transition-[background-color,border-color,transform] duration-[var(--dur-hover)] ease-[var(--ease-out)] hover:bg-neutral-800 hover:border-neutral-700 active:scale-[0.97] active:duration-[var(--dur-press)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 focus-visible:ring-offset-1 focus-visible:ring-offset-black"
            >
              &#8250;
            </button>
          </div>
        ) : null}

        <main
          className={`flex flex-1 flex-col gap-4 px-4 py-6 ${
            messages.length === 0
              ? "items-center justify-center"
              : "mx-auto w-full max-w-3xl"
          }`}
        >
          {messages.length === 0 ? (
            <div className="flex flex-col items-center gap-4">
              <h1 className="text-3xl font-semibold text-neutral-100">
                {me ? `Hello, ${me?.username ?? ""}` : "Hello"}
              </h1>
            </div>
          ) : (
            <div className="flex flex-1 flex-col gap-4">
              {messages.map((message) =>
                message.role === "user" ? (
                  <div
                    key={message.id}
                    className={`flex justify-end ${message.live ? "animate-fade-rise" : ""}`}
                  >
                    <div className="max-w-[85%] whitespace-pre-wrap rounded-xl border border-neutral-800 bg-neutral-900 px-4 py-2 text-sm text-neutral-100">
                      {message.content}
                    </div>
                  </div>
                ) : (
                  <div
                    key={message.id}
                    className={`flex justify-start ${message.live ? "animate-fade-rise" : ""}`}
                  >
                    <div className="w-full max-w-[92%]">
                      <p className="whitespace-pre-wrap text-sm leading-relaxed text-neutral-100">
                        {message.content}
                      </p>
                      <MessageExtras
                        actions={
                          <button
                            type="button"
                            onClick={() => void handleCopy(message.id, message.content)}
                            aria-label="Copy reply text"
                            className="mt-1 inline-flex items-center gap-1 rounded-md px-1.5 py-1 text-xs text-neutral-500 transition-[background-color,border-color,transform] duration-[var(--dur-hover)] ease-[var(--ease-out)] hover:bg-neutral-900 hover:text-neutral-300 active:scale-[0.97] active:duration-[var(--dur-press)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 focus-visible:ring-offset-1 focus-visible:ring-offset-black"
                          >
                            <svg
                              viewBox="0 0 24 24"
                              className="h-3.5 w-3.5"
                              fill="none"
                              stroke="currentColor"
                              strokeWidth="2"
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              aria-hidden="true"
                            >
                              <rect x="9" y="9" width="11" height="11" rx="2" />
                              <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
                            </svg>
                            {copyState[message.id] === "copied"
                              ? "Copied"
                              : copyState[message.id] === "failed"
                                ? "Copy failed"
                                : "Copy"}
                          </button>
                        }
                        memoryUsed={message.memoryUsed}
                        trace={message.trace}
                        followup={message.followup}
                        recordState={message.recordState}
                        taskState={message.taskState}
                        busy={busyItem}
                        onRecord={(recordId, action) =>
                          message.turnId
                            ? handleRecord(message.turnId, recordId, action)
                            : undefined
                        }
                        onTask={(index, action) =>
                          message.turnId ? handleTask(message.turnId, index, action) : undefined
                        }
                      />
                      {message.saveStatus === "failed" ? (
                        <p className="mt-2 text-xs text-rose-400">Not saved</p>
                      ) : null}
                    </div>
                  </div>
                ),
              )}
              {sending ? (
                <div className="flex justify-start">
                  <div
                    role="status"
                    aria-label="Assistant is thinking"
                    className="flex items-center gap-1 py-1"
                  >
                    <span className="h-1.5 w-1.5 animate-thinking-dot rounded-full bg-neutral-500" />
                    <span className="h-1.5 w-1.5 animate-thinking-dot rounded-full bg-neutral-500 [animation-delay:0.15s]" />
                    <span className="h-1.5 w-1.5 animate-thinking-dot rounded-full bg-neutral-500 [animation-delay:0.3s]" />
                  </div>
                </div>
              ) : null}
              <div ref={endRef} />
            </div>
          )}

          {error ? (
            <p role="alert" className="text-xs text-rose-400">
              {error}
            </p>
          ) : null}

          <div className={`w-full ${messages.length === 0 ? "max-w-2xl" : ""}`}>
            <form
              ref={formRef}
              onSubmit={handleSend}
              className="flex w-full items-end gap-2 rounded-2xl border border-neutral-800 bg-neutral-900 px-4 py-2 focus-within:border-neutral-600"
            >
              <textarea
                ref={composerRef}
                value={input}
                onChange={(event) => setInput(event.target.value)}
                onKeyDown={handleComposerKeyDown}
                rows={1}
                placeholder="Ask anything"
                aria-label="Message"
                className="flex-1 resize-none bg-transparent py-1 text-sm leading-6 text-neutral-100 placeholder:text-neutral-500 outline-none"
              />
              <button
                type="submit"
                disabled={disabled || input.trim().length === 0}
                aria-label="Send"
                className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full transition-[background-color,border-color,transform] duration-[var(--dur-hover)] ease-[var(--ease-out)] active:scale-[0.97] active:duration-[var(--dur-press)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 focus-visible:ring-offset-1 focus-visible:ring-offset-black disabled:cursor-not-allowed ${
                  disabled || input.trim().length === 0
                    ? "bg-neutral-800 text-neutral-600"
                    : "bg-teal-500 text-black hover:bg-teal-400"
                }`}
              >
                {sending ? (
                  <svg
                    viewBox="0 0 24 24"
                    className="h-4 w-4 animate-spin"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    aria-hidden="true"
                  >
                    <circle cx="12" cy="12" r="9" className="opacity-25" />
                    <path d="M21 12a9 9 0 0 0-9-9" />
                  </svg>
                ) : (
                  <svg
                    viewBox="0 0 24 24"
                    className="h-4 w-4"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    aria-hidden="true"
                  >
                    <path d="M22 2 11 13" />
                    <path d="M22 2 15 22l-4-9-9-4Z" />
                  </svg>
                )}
              </button>
            </form>
          </div>
        </main>
      </div>
    </div>
  );
}
