"use client";

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";

import { AppShell } from "@/components/app-shell";
import {
  ApiError,
  AuthError,
  confirmRecord,
  deleteMemory,
  editMemory,
  getMe,
  getTimeline,
  type TimelineItem,
} from "@/lib/api";

/** Mirrors backend MEMORY_EDIT_MAX: PATCH /memory/{id} rejects longer content. */
const MEMORY_EDIT_MAX = 500;

const GENERIC_ERROR = "Something went wrong. Please try again.";

const KIND_ORDER = ["knowledge", "decision", "preference", "glossary"] as const;
const KIND_LABELS: Record<string, string> = {
  knowledge: "Knowledge",
  decision: "Decisions",
  preference: "Preferences",
  glossary: "Glossary",
};

type StatusView = "confirmed" | "pending";

type Group = { key: string; label: string; items: TimelineItem[] };

/** Backend timestamps are naive UTC; treat a missing offset as UTC before localising. */
function parseTimestamp(value: string): Date {
  const hasZone = /[zZ]$|[+-]\d{2}:?\d{2}$/.test(value);
  return new Date(hasZone ? value : `${value}Z`);
}

function formatDateTime(value: string): string {
  const date = parseTimestamp(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return new Intl.DateTimeFormat(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

function formatDate(value: string): string {
  const date = parseTimestamp(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return new Intl.DateTimeFormat(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  }).format(date);
}

function titleCase(value: string): string {
  if (!value) {
    return value;
  }
  return value.charAt(0).toUpperCase() + value.slice(1);
}

function groupByKind(items: TimelineItem[]): Group[] {
  const buckets = new Map<string, TimelineItem[]>();
  for (const item of items) {
    const key = (KIND_ORDER as readonly string[]).includes(item.kind) ? item.kind : "other";
    const bucket = buckets.get(key);
    if (bucket) {
      bucket.push(item);
    } else {
      buckets.set(key, [item]);
    }
  }
  const order = [...KIND_ORDER, "other"];
  return order
    .map((key) => ({ key, items: buckets.get(key) ?? [] }))
    .filter((group) => group.items.length > 0)
    .map((group) => ({
      key: group.key,
      label: group.key === "other" ? "Other" : KIND_LABELS[group.key],
      items: group.items
        .slice()
        .sort((a, b) => parseTimestamp(b.created_at).getTime() - parseTimestamp(a.created_at).getTime()),
    }));
}

export default function MemoriesPage() {
  const router = useRouter();

  const [ready, setReady] = useState(false);
  const [view, setView] = useState<StatusView>("confirmed");
  const [items, setItems] = useState<TimelineItem[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [busyId, setBusyId] = useState<number | null>(null);

  const [editingId, setEditingId] = useState<number | null>(null);
  const [editText, setEditText] = useState("");
  const [confirmingDeleteId, setConfirmingDeleteId] = useState<number | null>(null);
  const [expandedId, setExpandedId] = useState<number | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        await getMe();
        if (!cancelled) {
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
        setReady(true);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [router]);

  const load = useCallback(
    async (status: StatusView) => {
      setLoading(true);
      setLoadError(null);
      try {
        const days = await getTimeline(status);
        setItems(days.flatMap((day) => day.items));
      } catch (err) {
        if (err instanceof AuthError) {
          router.replace("/login");
          return;
        }
        setItems(null);
        setLoadError(GENERIC_ERROR);
      } finally {
        setLoading(false);
      }
    },
    [router],
  );

  useEffect(() => {
    if (ready) {
      void load(view);
    }
  }, [ready, view, load]);

  async function handleSaveRecord(recordId: number) {
    setBusyId(recordId);
    setActionError(null);
    try {
      await confirmRecord(recordId);
      setItems((prev) => (prev ? prev.filter((item) => item.id !== recordId) : prev));
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

  async function handleEditSave(recordId: number) {
    const next = editText.trim();
    if (!next) {
      setActionError("Memory text cannot be empty.");
      return;
    }
    setBusyId(recordId);
    setActionError(null);
    try {
      await editMemory(recordId, next);
      setItems((prev) =>
        prev ? prev.map((item) => (item.id === recordId ? { ...item, content: next } : item)) : prev,
      );
      setEditingId(null);
      setEditText("");
    } catch (err) {
      if (err instanceof AuthError) {
        router.replace("/login");
        return;
      }
      setActionError(
        err instanceof ApiError && err.status === 422
          ? "That edit was rejected. Keep it under the limit and try again."
          : GENERIC_ERROR,
      );
    } finally {
      setBusyId(null);
    }
  }

  async function handleDelete(recordId: number) {
    setBusyId(recordId);
    setActionError(null);
    try {
      await deleteMemory(recordId);
      setItems((prev) => (prev ? prev.filter((item) => item.id !== recordId) : prev));
      setConfirmingDeleteId(null);
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

  const groups = items ? groupByKind(items) : [];

  return (
    <AppShell activeHref="/memories">
      <div className="mx-auto flex w-full max-w-3xl flex-1 flex-col gap-4 px-6 py-8">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h1 className="text-xl font-bold tracking-tight text-neutral-100">My memories</h1>
          <button
            type="button"
            onClick={() => void load(view)}
            disabled={loading}
            className="rounded-lg border border-neutral-800 bg-neutral-900 px-3 py-1.5 text-xs font-medium text-neutral-300 transition-[background-color,border-color,transform] duration-[var(--dur-hover)] ease-[var(--ease-out)] hover:bg-neutral-800 active:duration-[var(--dur-press)] active:scale-[0.98] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 focus-visible:ring-offset-1 focus-visible:ring-offset-black disabled:opacity-50"
          >
            Refresh
          </button>
        </div>

        <div className="flex gap-2">
          <button
            type="button"
            onClick={() => setView("confirmed")}
            aria-pressed={view === "confirmed"}
            className={
              view === "confirmed"
                ? "rounded-lg border border-neutral-700 bg-neutral-800 px-3 py-1.5 text-xs font-semibold text-neutral-100"
                : "rounded-lg border border-neutral-800 bg-neutral-900 px-3 py-1.5 text-xs font-medium text-neutral-400 transition-[background-color,border-color,transform] duration-[var(--dur-hover)] ease-[var(--ease-out)] hover:bg-neutral-800 active:duration-[var(--dur-press)] active:scale-[0.98] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 focus-visible:ring-offset-1 focus-visible:ring-offset-black"
            }
          >
            Confirmed
          </button>
          <button
            type="button"
            onClick={() => setView("pending")}
            aria-pressed={view === "pending"}
            className={
              view === "pending"
                ? "rounded-lg border border-amber-700 bg-amber-900/50 px-3 py-1.5 text-xs font-semibold text-amber-100"
                : "rounded-lg border border-neutral-800 bg-neutral-900 px-3 py-1.5 text-xs font-medium text-neutral-400 transition-[background-color,border-color,transform] duration-[var(--dur-hover)] ease-[var(--ease-out)] hover:bg-neutral-800 active:duration-[var(--dur-press)] active:scale-[0.98] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 focus-visible:ring-offset-1 focus-visible:ring-offset-black"
            }
          >
            Pending
          </button>
        </div>

        <p className="text-xs text-neutral-500">
          {view === "confirmed"
            ? "Confirmed: these memories are used when you chat."
            : "Pending: saved as candidates. Each one is used only after you press Save."}
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
              onClick={() => void load(view)}
              className="rounded-lg border border-rose-700 bg-rose-900/50 px-3 py-1.5 text-xs font-medium text-rose-100 transition-[background-color,border-color,transform] duration-[var(--dur-hover)] ease-[var(--ease-out)] hover:bg-rose-900/70 active:duration-[var(--dur-press)] active:scale-[0.98] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-rose-500 focus-visible:ring-offset-1 focus-visible:ring-offset-black"
            >
              Try again
            </button>
          </div>
        ) : null}

        {!loading && !loadError && items !== null && items.length === 0 ? (
          <p className="text-sm text-neutral-500">No memories here yet</p>
        ) : null}

        {!loading && !loadError && groups.length > 0 ? (
          <div className="flex flex-col gap-6">
            {groups.map((group) => (
              <section key={group.key} className="flex flex-col gap-3">
                <h2 className="text-xs font-semibold uppercase tracking-wider text-neutral-500">
                  {group.label}
                </h2>
                <ul className="flex flex-col gap-3">
                  {group.items.map((item) => {
                    const isPending = item.status !== "confirmed";
                    const isEditing = editingId === item.id;
                    const isBusy = busyId === item.id;
                    return (
                      <li
                        key={item.id}
                        className="rounded-xl border border-neutral-800 bg-neutral-950 p-4"
                      >
                        {isPending ? (
                          <p className="mb-2 inline-block rounded border border-amber-700 bg-amber-900/40 px-2 py-0.5 text-xs font-medium text-amber-200">
                            Pending: not used until you save it
                          </p>
                        ) : null}

                        {isEditing ? (
                          <div className="flex flex-col gap-2">
                            <textarea
                              value={editText}
                              onChange={(event) => setEditText(event.target.value)}
                              maxLength={MEMORY_EDIT_MAX}
                              aria-label="Edit memory text"
                              className="h-28 w-full resize-y rounded-lg border border-neutral-800 bg-neutral-900 p-2 text-sm text-neutral-100 outline-none focus:border-teal-600"
                            />
                            <div className="flex items-center justify-between text-xs text-neutral-500">
                              <span>Edits are saved as you confirm.</span>
                              <span>
                                {editText.length} / {MEMORY_EDIT_MAX}
                              </span>
                            </div>
                            <div className="flex gap-2">
                              <button
                                type="button"
                                onClick={() => void handleEditSave(item.id)}
                                disabled={isBusy}
                                className="rounded-lg border border-teal-700 bg-neutral-900 px-3 py-1.5 text-xs font-semibold text-teal-300 transition-[background-color,border-color,transform] duration-[var(--dur-hover)] ease-[var(--ease-out)] hover:bg-neutral-800 active:duration-[var(--dur-press)] active:scale-[0.98] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 focus-visible:ring-offset-1 focus-visible:ring-offset-black disabled:opacity-50"
                              >
                                {isBusy ? "Saving…" : "Save"}
                              </button>
                              <button
                                type="button"
                                onClick={() => {
                                  setEditingId(null);
                                  setEditText("");
                                }}
                                disabled={isBusy}
                                className="rounded-lg border border-neutral-800 bg-neutral-900 px-3 py-1.5 text-xs font-medium text-neutral-300 transition-[background-color,border-color,transform] duration-[var(--dur-hover)] ease-[var(--ease-out)] hover:bg-neutral-800 active:duration-[var(--dur-press)] active:scale-[0.98] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 focus-visible:ring-offset-1 focus-visible:ring-offset-black disabled:opacity-50"
                              >
                                Cancel
                              </button>
                            </div>
                          </div>
                        ) : (
                          <>
                            <p className="whitespace-pre-wrap text-sm text-neutral-100">
                              {item.content}
                            </p>

                            <p className="mt-2 text-xs text-neutral-400">
                              {item.kind === "other" ? "Other" : KIND_LABELS[item.kind] ?? titleCase(item.kind)}
                              {" · "}
                              {titleCase(item.certainty)}
                              {" · "}
                              {titleCase(item.source)}
                              {" · "}
                              {formatDateTime(item.created_at)}
                            </p>

                            {item.kind === "decision" && item.rationale ? (
                              <p className="mt-1 text-xs text-neutral-400">
                                Reason: {item.rationale}
                              </p>
                            ) : null}

                            {expandedId === item.id ? (
                              <dl className="mt-2 flex flex-col gap-1 rounded-lg border border-neutral-800 bg-neutral-900 p-2 text-xs text-neutral-400">
                                <div>
                                  <dt className="inline font-medium text-neutral-500">Source: </dt>
                                  <dd className="inline">{titleCase(item.source)}</dd>
                                </div>
                                {item.source_ref ? (
                                  <div>
                                    <dt className="inline font-medium text-neutral-500">
                                      Source reference:{" "}
                                    </dt>
                                    <dd className="inline whitespace-pre-wrap">{item.source_ref}</dd>
                                  </div>
                                ) : null}
                                <div>
                                  <dt className="inline font-medium text-neutral-500">Certainty: </dt>
                                  <dd className="inline">{titleCase(item.certainty)}</dd>
                                </div>
                                <div>
                                  <dt className="inline font-medium text-neutral-500">Created: </dt>
                                  <dd className="inline">{formatDate(item.created_at)}</dd>
                                </div>
                              </dl>
                            ) : null}

                            <div className="mt-3 flex flex-wrap gap-2">
                              {isPending ? (
                                <button
                                  type="button"
                                  onClick={() => void handleSaveRecord(item.id)}
                                  disabled={isBusy}
                                  className="rounded-lg border border-teal-700 bg-neutral-900 px-3 py-1.5 text-xs font-semibold text-teal-300 transition-[background-color,border-color,transform] duration-[var(--dur-hover)] ease-[var(--ease-out)] hover:bg-neutral-800 active:duration-[var(--dur-press)] active:scale-[0.98] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 focus-visible:ring-offset-1 focus-visible:ring-offset-black disabled:opacity-50"
                                >
                                  {isBusy ? "Saving…" : "Save"}
                                </button>
                              ) : null}
                              <button
                                type="button"
                                onClick={() => {
                                  setEditingId(item.id);
                                  setEditText(item.content);
                                  setConfirmingDeleteId(null);
                                }}
                                disabled={isBusy}
                                className="rounded-lg border border-neutral-800 bg-neutral-900 px-3 py-1.5 text-xs font-medium text-neutral-300 transition-[background-color,border-color,transform] duration-[var(--dur-hover)] ease-[var(--ease-out)] hover:bg-neutral-800 active:duration-[var(--dur-press)] active:scale-[0.98] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 focus-visible:ring-offset-1 focus-visible:ring-offset-black disabled:opacity-50"
                              >
                                Edit
                              </button>
                              <button
                                type="button"
                                onClick={() => setExpandedId(expandedId === item.id ? null : item.id)}
                                aria-expanded={expandedId === item.id}
                                className="rounded-lg border border-neutral-800 bg-neutral-900 px-3 py-1.5 text-xs font-medium text-neutral-300 transition-[background-color,border-color,transform] duration-[var(--dur-hover)] ease-[var(--ease-out)] hover:bg-neutral-800 active:duration-[var(--dur-press)] active:scale-[0.98] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 focus-visible:ring-offset-1 focus-visible:ring-offset-black"
                              >
                                {expandedId === item.id ? "Hide source" : "Show source"}
                              </button>
                              {confirmingDeleteId === item.id ? (
                                <>
                                  <span className="self-center text-xs text-rose-300">
                                    Delete this memory?
                                  </span>
                                  <button
                                    type="button"
                                    onClick={() => void handleDelete(item.id)}
                                    disabled={isBusy}
                                    className="rounded-lg border border-rose-700 bg-rose-900/50 px-3 py-1.5 text-xs font-semibold text-rose-100 transition-[background-color,border-color,transform] duration-[var(--dur-hover)] ease-[var(--ease-out)] hover:bg-rose-900/70 active:duration-[var(--dur-press)] active:scale-[0.98] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-rose-500 focus-visible:ring-offset-1 focus-visible:ring-offset-black disabled:opacity-50"
                                  >
                                    {isBusy ? "Deleting…" : "Confirm delete"}
                                  </button>
                                  <button
                                    type="button"
                                    onClick={() => setConfirmingDeleteId(null)}
                                    disabled={isBusy}
                                    className="rounded-lg border border-neutral-800 bg-neutral-900 px-3 py-1.5 text-xs font-medium text-neutral-300 transition-[background-color,border-color,transform] duration-[var(--dur-hover)] ease-[var(--ease-out)] hover:bg-neutral-800 active:duration-[var(--dur-press)] active:scale-[0.98] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 focus-visible:ring-offset-1 focus-visible:ring-offset-black disabled:opacity-50"
                                  >
                                    Cancel
                                  </button>
                                </>
                              ) : (
                                <button
                                  type="button"
                                  onClick={() => {
                                    setConfirmingDeleteId(item.id);
                                    setEditingId(null);
                                  }}
                                  disabled={isBusy}
                                  className="rounded-lg border border-neutral-800 bg-neutral-900 px-3 py-1.5 text-xs font-medium text-rose-300 transition-[background-color,border-color,transform] duration-[var(--dur-hover)] ease-[var(--ease-out)] hover:border-rose-700 hover:bg-neutral-800 active:duration-[var(--dur-press)] active:scale-[0.98] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-rose-500 focus-visible:ring-offset-1 focus-visible:ring-offset-black disabled:opacity-50"
                                >
                                  Delete
                                </button>
                              )}
                            </div>
                          </>
                        )}
                      </li>
                    );
                  })}
                </ul>
              </section>
            ))}
          </div>
        ) : null}
      </div>
    </AppShell>
  );
}
