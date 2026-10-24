"use client";

import { useCallback, useEffect, useRef, useState, type FormEvent } from "react";
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
          })),
        );
        activateConversation(id);
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
            className="rounded-md border border-neutral-800 bg-neutral-900 px-3 py-1.5 text-xs font-medium text-neutral-300 hover:border-neutral-700"
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
              className="rounded-md border border-neutral-800 bg-neutral-900 px-2.5 py-1.5 text-xs font-medium text-neutral-300 hover:border-neutral-700"
            >
              &#8250;
            </button>
          </div>
        ) : null}

        <main className="mx-auto flex w-full max-w-3xl flex-1 flex-col gap-4 px-4 py-6">
          {messages.length === 0 ? (
            <div className="flex flex-1 flex-col items-center justify-center gap-6">
              <h1 className="text-3xl font-semibold text-neutral-100">
                {me ? `Hello, ${me?.username ?? ""}` : "Hello"}
              </h1>

              {error ? (
                <p role="alert" className="text-xs text-rose-400">
                  {error}
                </p>
              ) : null}

              <div className="w-full max-w-2xl">
                <form
                  onSubmit={handleSend}
                  className="flex w-full items-center gap-2 rounded-full border border-neutral-800 bg-neutral-900 px-4 py-2 focus-within:border-neutral-600"
                >
                  <input
                    type="text"
                    value={input}
                    onChange={(event) => setInput(event.target.value)}
                    placeholder="Ask anything"
                    aria-label="Message"
                    className="flex-1 bg-transparent text-sm text-neutral-100 placeholder:text-neutral-500 outline-none"
                  />
                  <button
                    type="submit"
                    disabled={disabled || input.trim().length === 0}
                    aria-label="Send"
                    className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full ${
                      disabled || input.trim().length === 0
                        ? "bg-neutral-800 text-neutral-500"
                        : "bg-teal-500 text-black"
                    }`}
                  >
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
                  </button>
                </form>
              </div>
            </div>
          ) : (
            <>
              <div className="flex flex-1 flex-col gap-4">
                {messages.map((message) =>
                  message.role === "user" ? (
                    <div key={message.id} className="flex justify-end">
                      <div className="max-w-[85%] whitespace-pre-wrap rounded-xl border border-neutral-800 bg-neutral-900 px-4 py-2 text-sm text-neutral-100">
                        {message.content}
                      </div>
                    </div>
                  ) : (
                    <div key={message.id} className="flex justify-start">
                      <div className="w-full max-w-[92%] rounded-xl border border-neutral-800 bg-neutral-950 px-4 py-3">
                        <p className="whitespace-pre-wrap text-sm leading-relaxed text-neutral-100">
                          {message.content}
                        </p>
                        <MessageExtras
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
                <div ref={endRef} />
              </div>

              {error ? (
                <p role="alert" className="text-xs text-rose-400">
                  {error}
                </p>
              ) : null}

              <form
                onSubmit={handleSend}
                className="flex w-full items-center gap-2 rounded-full border border-neutral-800 bg-neutral-900 px-4 py-2 focus-within:border-neutral-600"
              >
                <input
                  type="text"
                  value={input}
                  onChange={(event) => setInput(event.target.value)}
                  placeholder="Ask anything"
                  aria-label="Message"
                  className="flex-1 bg-transparent text-sm text-neutral-100 placeholder:text-neutral-500 outline-none"
                />
                <button
                  type="submit"
                  disabled={disabled || input.trim().length === 0}
                  aria-label="Send"
                  className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full ${
                    disabled || input.trim().length === 0
                      ? "bg-neutral-800 text-neutral-500"
                      : "bg-teal-500 text-black"
                  }`}
                >
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
                </button>
              </form>
            </>
          )}
        </main>
      </div>
    </div>
  );
}
