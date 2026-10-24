"use client";

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";

import { AppShell } from "@/components/app-shell";
import {
  AuthError,
  getMe,
  getTasks,
  markTaskDone,
  type TaskOut,
} from "@/lib/api";

const GENERIC_ERROR = "Something went wrong. Please try again.";

type Tab = "open" | "done";

/** Backend timestamps are UTC; a missing offset means UTC. */
function parseTimestamp(value: string): Date {
  const hasZone = /[zZ]$|[+-]\d{2}:?\d{2}$/.test(value);
  return new Date(hasZone ? value : `${value}Z`);
}

function formatDateTime(value: string, timeZone: string | null): string {
  const date = parseTimestamp(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  try {
    return new Intl.DateTimeFormat(undefined, {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
      timeZone: timeZone ?? undefined,
    }).format(date);
  } catch {
    return new Intl.DateTimeFormat(undefined, {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    }).format(date);
  }
}

function originLabel(origin: string): string {
  if (origin === "promise") {
    return "Promise";
  }
  return origin.charAt(0).toUpperCase() + origin.slice(1);
}

/** Open: nearest due first, no due date last. */
function sortOpen(a: TaskOut, b: TaskOut): number {
  const at = a.due_at ? parseTimestamp(a.due_at).getTime() : Number.POSITIVE_INFINITY;
  const bt = b.due_at ? parseTimestamp(b.due_at).getTime() : Number.POSITIVE_INFINITY;
  if (at !== bt) {
    return at - bt;
  }
  return a.id - b.id;
}

/** Completed: by due date descending (no completion column exists), no due date last. */
function sortDone(a: TaskOut, b: TaskOut): number {
  const at = a.due_at ? parseTimestamp(a.due_at).getTime() : Number.NEGATIVE_INFINITY;
  const bt = b.due_at ? parseTimestamp(b.due_at).getTime() : Number.NEGATIVE_INFINITY;
  if (at !== bt) {
    return bt - at;
  }
  return b.id - a.id;
}

export default function TasksPage() {
  const router = useRouter();

  const [ready, setReady] = useState(false);
  const [timezone, setTimezone] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("open");
  const [tasks, setTasks] = useState<TaskOut[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [busyId, setBusyId] = useState<number | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const me = await getMe();
        if (!cancelled) {
          setTimezone(me.timezone ?? null);
          setReady(true);
        }
      } catch (err) {
        if (cancelled) {
          return;
        }
        if (err instanceof AuthError) {
          router.replace("/login");
          return;
        }
        // Fall back to the browser timezone.
        setReady(true);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [router]);

  const load = useCallback(
    async (which: Tab) => {
      setLoading(true);
      setLoadError(null);
      try {
        const rows = await getTasks(which);
        setTasks(rows);
      } catch (err) {
        if (err instanceof AuthError) {
          router.replace("/login");
          return;
        }
        setTasks(null);
        setLoadError(GENERIC_ERROR);
      } finally {
        setLoading(false);
      }
    },
    [router],
  );

  useEffect(() => {
    if (ready) {
      void load(tab);
    }
  }, [ready, tab, load]);

  async function handleMarkDone(taskId: number) {
    setBusyId(taskId);
    setActionError(null);
    try {
      await markTaskDone(taskId);
      setTasks((prev) => (prev ? prev.filter((item) => item.id !== taskId) : prev));
    } catch (err) {
      if (err instanceof AuthError) {
        router.replace("/login");
        return;
      }
      setActionError(GENERIC_ERROR);
    } finally {
      setBusyId(null);
    }
  }

  if (!ready) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-black p-6 text-sm text-neutral-500">
        Loading…
      </div>
    );
  }

  const now = Date.now();
  const sorted = tasks
    ? tasks.slice().sort(tab === "open" ? sortOpen : sortDone)
    : [];
  const busy = busyId !== null;

  return (
    <AppShell activeHref="/tasks">
      <div className="mx-auto flex w-full max-w-3xl flex-1 flex-col gap-4 px-6 py-8">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h1 className="text-xl font-bold tracking-tight text-neutral-100">My tasks</h1>
          <button
            type="button"
            onClick={() => void load(tab)}
            disabled={loading}
            className="rounded-lg border border-neutral-800 bg-neutral-900 px-3 py-1.5 text-xs font-medium text-neutral-300 transition-[background-color,border-color,transform] duration-[var(--dur-hover)] ease-[var(--ease-out)] hover:bg-neutral-800 active:duration-[var(--dur-press)] active:scale-[0.98] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 focus-visible:ring-offset-1 focus-visible:ring-offset-black disabled:opacity-50"
          >
            Refresh
          </button>
        </div>

        <div className="flex gap-2">
          <button
            type="button"
            onClick={() => setTab("open")}
            aria-pressed={tab === "open"}
            className={
              tab === "open"
                ? "rounded-lg border border-neutral-700 bg-neutral-800 px-3 py-1.5 text-xs font-semibold text-neutral-100"
                : "rounded-lg border border-neutral-800 bg-neutral-900 px-3 py-1.5 text-xs font-medium text-neutral-400 transition-[background-color,border-color,transform] duration-[var(--dur-hover)] ease-[var(--ease-out)] hover:bg-neutral-800 active:duration-[var(--dur-press)] active:scale-[0.98] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 focus-visible:ring-offset-1 focus-visible:ring-offset-black"
            }
          >
            Not completed
          </button>
          <button
            type="button"
            onClick={() => setTab("done")}
            aria-pressed={tab === "done"}
            className={
              tab === "done"
                ? "rounded-lg border border-neutral-700 bg-neutral-800 px-3 py-1.5 text-xs font-semibold text-neutral-100"
                : "rounded-lg border border-neutral-800 bg-neutral-900 px-3 py-1.5 text-xs font-medium text-neutral-400 transition-[background-color,border-color,transform] duration-[var(--dur-hover)] ease-[var(--ease-out)] hover:bg-neutral-800 active:duration-[var(--dur-press)] active:scale-[0.98] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 focus-visible:ring-offset-1 focus-visible:ring-offset-black"
            }
          >
            Completed
          </button>
        </div>

        <p className="text-xs text-neutral-500">
          Reminders are sent for tasks that have a due time.
        </p>

        {actionError ? (
          <p role="alert" className="rounded-lg border border-rose-800/60 bg-rose-950/30 p-3 text-xs text-rose-300">
            {actionError}
          </p>
        ) : null}

        {loading ? <p className="text-sm text-neutral-500">Loading…</p> : null}

        {!loading && loadError ? (
          <div className="flex flex-col items-start gap-3 rounded-lg border border-rose-800/60 bg-rose-950/30 p-4">
            <p role="alert" className="text-sm text-rose-300">
              {loadError}
            </p>
            <button
              type="button"
              onClick={() => void load(tab)}
              className="rounded-lg border border-rose-700 bg-rose-900/50 px-3 py-1.5 text-xs font-medium text-rose-100 transition-[background-color,border-color,transform] duration-[var(--dur-hover)] ease-[var(--ease-out)] hover:bg-rose-900/70 active:duration-[var(--dur-press)] active:scale-[0.98] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-rose-500 focus-visible:ring-offset-1 focus-visible:ring-offset-black"
            >
              Try again
            </button>
          </div>
        ) : null}

        {!loading && !loadError && tasks !== null && sorted.length === 0 ? (
          <p className="text-sm text-neutral-500">
            {tab === "open" ? "No open tasks" : "No completed tasks"}
          </p>
        ) : null}

        {!loading && !loadError && sorted.length > 0 ? (
          <ul className="flex flex-col gap-3">
            {sorted.map((item) => {
              const isOpen = item.status === "open";
              const dueAt = item.due_at ? parseTimestamp(item.due_at).getTime() : null;
              const overdue = isOpen && dueAt !== null && dueAt < now;
              const isBusy = busyId === item.id;
              return (
                <li key={item.id} className="rounded-xl border border-neutral-800 bg-neutral-950 p-4">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="rounded border border-neutral-800 bg-neutral-900 px-2 py-0.5 text-xs font-medium text-neutral-300">
                      Task
                    </span>
                    <span className="rounded border border-neutral-800 bg-neutral-900 px-2 py-0.5 text-xs font-medium text-neutral-300">
                      {originLabel(item.origin)}
                    </span>
                    {overdue ? (
                      <span className="rounded border border-rose-700 bg-rose-900/40 px-2 py-0.5 text-xs font-medium text-rose-200">
                        Overdue
                      </span>
                    ) : null}
                  </div>

                  <p className="mt-2 whitespace-pre-wrap text-sm text-neutral-100">{item.title}</p>

                  <p className="mt-1 text-xs text-neutral-400">
                    {item.due_at
                      ? `Due: ${formatDateTime(item.due_at, timezone)}`
                      : "No due date"}
                  </p>

                  {isOpen ? (
                    <div className="mt-3">
                      <button
                        type="button"
                        onClick={() => void handleMarkDone(item.id)}
                        disabled={busy}
                        className="rounded-lg border border-teal-700 bg-neutral-900 px-3 py-1.5 text-xs font-semibold text-teal-300 transition-[background-color,border-color,transform] duration-[var(--dur-hover)] ease-[var(--ease-out)] hover:bg-neutral-800 active:duration-[var(--dur-press)] active:scale-[0.98] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 focus-visible:ring-offset-1 focus-visible:ring-offset-black disabled:opacity-50"
                      >
                        {isBusy ? "Marking…" : "Mark done"}
                      </button>
                    </div>
                  ) : null}
                </li>
              );
            })}
          </ul>
        ) : null}
      </div>
    </AppShell>
  );
}
