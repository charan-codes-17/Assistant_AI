"use client";

import { useCallback, useEffect, useRef, useState, type FormEvent } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";

import { MessageExtras, type RecordState, type TaskState } from "@/components/message-extras";
import {
  ApiError,
  AuthError,
  chat,
  confirmRecord,
  confirmTask,
  discardTask,
  getFollowup,
  getMe,
  logout,
  newSession,
  rejectRecord,
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

  // Bumped on unmount-less resets (new session / logout): any poller from an older generation
  // stops as soon as it notices the change.
  const generationRef = useRef(0);
  const timersRef = useRef<Set<ReturnType<typeof setTimeout>>>(new Set());
  const idRef = useRef(0);
  const endRef = useRef<HTMLDivElement>(null);

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
          },
        ]);
        startPolling(response.turn_id);
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
    [input, sending, sessionId, messages, router, startPolling],
  );

  const handleNewSession = useCallback(async () => {
    stopPolling();
    generationRef.current += 1;
    setMessages([]);
    setInput("");
    setError(null);
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
  }, [router, stopPolling]);

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
    <div className="flex min-h-screen flex-col bg-slate-950 text-slate-100">
      <div className="border-b border-amber-800/60 bg-amber-950/40 px-4 py-2 text-center text-xs text-amber-300">
        {BANNER}
      </div>

      <header className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 px-4 py-3">
        <div className="flex items-center gap-3">
          <div className="h-2.5 w-2.5 rounded-full bg-emerald-500" />
          <span className="text-sm font-semibold tracking-tight text-white">
            Personal Memory Assistant
          </span>
          {me ? <span className="text-xs text-slate-500">{me.username ?? "signed in"}</span> : null}
        </div>
        <div className="flex gap-2">
          <Link
            href="/onboarding"
            className="rounded-md border border-slate-700 bg-slate-900 px-3 py-1.5 text-xs font-medium text-slate-300 hover:border-slate-600"
          >
            Import knowledge
          </Link>
          <button
            type="button"
            onClick={handleNewSession}
            className="rounded-md border border-slate-700 bg-slate-900 px-3 py-1.5 text-xs font-medium text-slate-300 hover:border-slate-600"
          >
            New session
          </button>
          <button
            type="button"
            onClick={handleLogout}
            className="rounded-md border border-slate-700 bg-slate-900 px-3 py-1.5 text-xs font-medium text-slate-300 hover:border-slate-600"
          >
            Log out
          </button>
        </div>
      </header>

      <main className="mx-auto flex w-full max-w-3xl flex-1 flex-col gap-4 px-4 py-6">
        <div className="flex flex-1 flex-col gap-4">
          {messages.length === 0 ? (
            <p className="text-sm text-slate-500">
              Ask something, or tell me something worth remembering.
            </p>
          ) : null}

          {messages.map((message) =>
            message.role === "user" ? (
              <div key={message.id} className="flex justify-end">
                <div className="max-w-[85%] whitespace-pre-wrap rounded-xl border border-slate-700 bg-slate-800 px-4 py-2 text-sm text-slate-100">
                  {message.content}
                </div>
              </div>
            ) : (
              <div key={message.id} className="flex justify-start">
                <div className="w-full max-w-[92%] rounded-xl border border-slate-800 bg-slate-900 px-4 py-3">
                  <p className="whitespace-pre-wrap text-sm leading-relaxed text-slate-100">
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
                      message.turnId ? handleRecord(message.turnId, recordId, action) : undefined
                    }
                    onTask={(index, action) =>
                      message.turnId ? handleTask(message.turnId, index, action) : undefined
                    }
                  />
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

        <form onSubmit={handleSend} className="flex items-center gap-2">
          <input
            type="text"
            value={input}
            onChange={(event) => setInput(event.target.value)}
            placeholder="Type a message…"
            aria-label="Message"
            className="flex-1 rounded-md border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-slate-100 outline-none placeholder:text-slate-600 focus:border-emerald-700"
          />
          <button
            type="submit"
            disabled={disabled || input.trim().length === 0}
            className="rounded-md border border-emerald-700 bg-emerald-900/60 px-4 py-2 text-sm font-semibold text-emerald-100 hover:bg-emerald-900 disabled:opacity-50"
          >
            {sending ? "Sending…" : "Send"}
          </button>
        </form>
      </main>
    </div>
  );
}
