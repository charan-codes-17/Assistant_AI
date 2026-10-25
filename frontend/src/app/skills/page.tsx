"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";

import { AppShell } from "@/components/app-shell";
import { MessageExtras, type TaskState } from "@/components/message-extras";
import {
  AuthError,
  confirmTask,
  deleteSkill,
  discardTask,
  getFollowup,
  getMe,
  getSkills,
  runSkill,
  type FollowupResponse,
  type SkillOut,
  type SkillRunResponse,
} from "@/lib/api";

const GENERIC_ERROR = "Something went wrong. Please try again.";

type Tab = "all" | "builtin" | "user";

const TABS: { key: Tab; label: string }[] = [
  { key: "all", label: "All" },
  { key: "builtin", label: "Built-in" },
  { key: "user", label: "User-defined" },
];

/**
 * Per-card state, keyed by skill name. A run never touches the database, so everything a card
 * shows after a run (reply, trace, pending-task candidates) lives here and disappears on reload.
 */
type CardState = {
  inputOpen: boolean;
  input: string;
  running: boolean;
  error: string | null;
  result: SkillRunResponse | null;
  followup: FollowupResponse | null;
  taskState: Record<number, TaskState>;
};

function emptyCard(): CardState {
  return {
    inputOpen: false,
    input: "",
    running: false,
    error: null,
    result: null,
    followup: null,
    taskState: {},
  };
}

const BUTTON_PRIMARY =
  "rounded-lg border border-teal-700 bg-neutral-900 px-3 py-1.5 text-xs font-semibold text-teal-300 transition-[background-color,border-color,transform] duration-[var(--dur-hover)] ease-[var(--ease-out)] hover:bg-neutral-800 active:duration-[var(--dur-press)] active:scale-[0.98] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 focus-visible:ring-offset-1 focus-visible:ring-offset-black disabled:opacity-50";

const BUTTON_NEUTRAL =
  "rounded-lg border border-neutral-800 bg-neutral-900 px-3 py-1.5 text-xs font-medium text-neutral-300 transition-[background-color,border-color,transform] duration-[var(--dur-hover)] ease-[var(--ease-out)] hover:bg-neutral-800 active:duration-[var(--dur-press)] active:scale-[0.98] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 focus-visible:ring-offset-1 focus-visible:ring-offset-black disabled:opacity-50";

export default function SkillsPage() {
  const router = useRouter();

  const [ready, setReady] = useState(false);
  const [skills, setSkills] = useState<SkillOut[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

  const [tab, setTab] = useState<Tab>("all");
  const [search, setSearch] = useState("");

  const [cards, setCards] = useState<Record<string, CardState>>({});
  const [taskBusy, setTaskBusy] = useState(false);

  const [deleting, setDeleting] = useState<string | null>(null);
  const [busyDelete, setBusyDelete] = useState<string | null>(null);

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

  const load = useCallback(async () => {
    setLoading(true);
    setLoadError(null);
    try {
      const rows = await getSkills();
      setSkills(rows);
    } catch (err) {
      if (err instanceof AuthError) {
        router.replace("/login");
        return;
      }
      setSkills(null);
      setLoadError(GENERIC_ERROR);
    } finally {
      setLoading(false);
    }
  }, [router]);

  useEffect(() => {
    if (ready) {
      void load();
    }
  }, [ready, load]);

  const patchCard = useCallback((name: string, patch: Partial<CardState>) => {
    setCards((prev) => ({ ...prev, [name]: { ...(prev[name] ?? emptyCard()), ...patch } }));
  }, []);

  async function runNow(skill: SkillOut, input: string) {
    patchCard(skill.name, { running: true, error: null, result: null, followup: null });
    try {
      const result = await runSkill(skill.name, input);
      let followup: FollowupResponse | null = null;
      if (result.pending_tasks_count > 0) {
        try {
          const fetched = await getFollowup(result.turn_id);
          followup = fetched.status === "ready" ? fetched : null;
        } catch {
          // The reply still renders; the candidate cards simply stay hidden.
          followup = null;
        }
      }
      patchCard(skill.name, { running: false, result, followup, taskState: {} });
    } catch (err) {
      if (err instanceof AuthError) {
        router.replace("/login");
        return;
      }
      patchCard(skill.name, { running: false, error: GENERIC_ERROR });
    }
  }

  async function handleTask(
    name: string,
    turnId: string,
    index: number,
    action: "save" | "discard",
  ) {
    setTaskBusy(true);
    setActionError(null);
    try {
      if (action === "save") {
        await confirmTask(turnId, index);
      } else {
        await discardTask(turnId, index);
      }
      const state: TaskState = action === "save" ? "confirmed" : "discarded";
      setCards((prev) => {
        const card = prev[name];
        if (!card) {
          return prev;
        }
        return { ...prev, [name]: { ...card, taskState: { ...card.taskState, [index]: state } } };
      });
    } catch (err) {
      if (err instanceof AuthError) {
        router.replace("/login");
        return;
      }
      setActionError(GENERIC_ERROR);
    } finally {
      setTaskBusy(false);
    }
  }

  async function handleDelete(skill: SkillOut) {
    if (skill.id === null) {
      return;
    }
    setBusyDelete(skill.name);
    setActionError(null);
    try {
      await deleteSkill(skill.id);
      setSkills((prev) => (prev ? prev.filter((item) => item.id !== skill.id) : prev));
      setDeleting(null);
    } catch (err) {
      if (err instanceof AuthError) {
        router.replace("/login");
        return;
      }
      setActionError(GENERIC_ERROR);
    } finally {
      setBusyDelete(null);
    }
  }

  if (!ready) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-black p-6 text-sm text-neutral-500">
        Loading…
      </div>
    );
  }

  const all = skills ?? [];
  const query = search.trim().toLowerCase();
  const tabScoped =
    tab === "all"
      ? all
      : all.filter((skill) => (tab === "builtin" ? skill.builtin : !skill.builtin));
  const visible = query
    ? tabScoped.filter(
        (skill) =>
          skill.name.toLowerCase().includes(query) ||
          skill.description.toLowerCase().includes(query),
      )
    : tabScoped;

  return (
    <AppShell activeHref="/skills">
      <div className="mx-auto flex h-full min-h-0 w-full max-w-3xl flex-1 flex-col gap-4 px-6 py-8">
        <div className="flex shrink-0 flex-wrap items-center justify-between gap-3">
          <h1 className="text-xl font-bold tracking-tight text-neutral-100">Skills</h1>
          <Link
            href="/skills/create"
            className="rounded-lg border border-teal-700 bg-neutral-900 px-4 py-2 text-sm font-semibold text-teal-300 transition-[background-color,border-color,transform] duration-[var(--dur-hover)] ease-[var(--ease-out)] hover:bg-neutral-800 active:duration-[var(--dur-press)] active:scale-[0.98] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 focus-visible:ring-offset-1 focus-visible:ring-offset-black"
          >
            Create Skill
          </Link>
        </div>

        <div className="flex shrink-0 gap-2">
          {TABS.map((entry) => (
            <button
              key={entry.key}
              type="button"
              onClick={() => setTab(entry.key)}
              aria-pressed={tab === entry.key}
              className={
                tab === entry.key
                  ? "rounded-lg border border-neutral-700 bg-neutral-800 px-3 py-1.5 text-xs font-semibold text-neutral-100"
                  : "rounded-lg border border-neutral-800 bg-neutral-900 px-3 py-1.5 text-xs font-medium text-neutral-400 transition-[background-color,border-color,transform] duration-[var(--dur-hover)] ease-[var(--ease-out)] hover:bg-neutral-800 active:duration-[var(--dur-press)] active:scale-[0.98] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 focus-visible:ring-offset-1 focus-visible:ring-offset-black"
              }
            >
              {entry.label}
            </button>
          ))}
        </div>

        <p className="shrink-0 text-xs text-neutral-500">
          Skills draft a reply from your own memory. Nothing is saved until you confirm.
        </p>

        <div className="relative shrink-0">
          <input
            type="text"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="Search skills"
            aria-label="Search skills"
            className="w-full rounded-lg border border-neutral-800 bg-neutral-900 px-3 py-1.5 pr-8 text-xs text-neutral-100 placeholder:text-neutral-500 outline-none focus:border-neutral-600 focus-visible:ring-2 focus-visible:ring-teal-500 focus-visible:ring-offset-1 focus-visible:ring-offset-black"
          />
          {search ? (
            <button
              type="button"
              onClick={() => setSearch("")}
              aria-label="Clear search"
              className="absolute right-2 top-1/2 -translate-y-1/2 rounded p-0.5 text-sm leading-none text-neutral-500 transition-[color,transform] duration-[var(--dur-hover)] ease-[var(--ease-out)] hover:text-neutral-200 active:duration-[var(--dur-press)] active:scale-[0.97] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500"
            >
              ×
            </button>
          ) : null}
        </div>

        {actionError ? (
          <p role="alert" className="rounded-lg border border-rose-800/60 bg-rose-950/30 p-3 text-xs text-rose-300">
            {actionError}
          </p>
        ) : null}

        <div className="chat-scroll min-h-0 flex-1 overflow-y-auto">
          {loading ? <p className="text-sm text-neutral-500">Loading…</p> : null}

          {!loading && loadError ? (
            <div className="flex flex-col items-start gap-3 rounded-lg border border-rose-800/60 bg-rose-950/30 p-4">
              <p role="alert" className="text-sm text-rose-300">
                {loadError}
              </p>
              <button
                type="button"
                onClick={() => void load()}
                className="rounded-lg border border-rose-700 bg-rose-900/50 px-3 py-1.5 text-xs font-medium text-rose-100 transition-[background-color,border-color,transform] duration-[var(--dur-hover)] ease-[var(--ease-out)] hover:bg-rose-900/70 active:duration-[var(--dur-press)] active:scale-[0.98] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-rose-500 focus-visible:ring-offset-1 focus-visible:ring-offset-black"
              >
                Try again
              </button>
            </div>
          ) : null}

          {!loading && !loadError && skills !== null && all.length === 0 ? (
            <p className="text-sm text-neutral-500">No skills yet</p>
          ) : null}

          {!loading && !loadError && all.length > 0 && visible.length === 0 ? (
            <div className="flex flex-col items-start gap-3 rounded-lg border border-neutral-800 bg-neutral-950 p-4">
              <p className="text-sm text-neutral-400">No skills match your search</p>
              <button
                type="button"
                onClick={() => {
                  setTab("all");
                  setSearch("");
                }}
                className={BUTTON_NEUTRAL}
              >
                Clear filters
              </button>
            </div>
          ) : null}

          {!loading && !loadError && visible.length > 0 ? (
            <ul className="flex flex-col gap-3 pb-2">
              {visible.map((skill) => {
                const card = cards[skill.name] ?? emptyCard();
                const runTurnId = card.result ? card.result.turn_id : null;
                const needsInput = skill.input !== "none";
                const requireInput = skill.input === "required";
                const isDeleting = deleting === skill.name;
                const isBusyDelete = busyDelete === skill.name;
                return (
                  <li key={skill.name} className="rounded-xl border border-neutral-800 bg-neutral-950 p-4">
                    <div className="flex flex-wrap items-start justify-between gap-2">
                      <div className="min-w-0">
                        <p className="text-sm font-semibold text-neutral-100">{skill.name}</p>
                        <p className="mt-1 text-xs text-neutral-400">{skill.description}</p>
                      </div>
                      <span className="rounded border border-neutral-800 bg-neutral-900 px-2 py-0.5 text-xs font-medium text-neutral-300">
                        {skill.builtin ? "Built-in" : "User-defined"}
                      </span>
                    </div>

                    <div className="mt-3 flex flex-wrap items-center gap-2">
                      {needsInput && card.inputOpen ? (
                        <>
                          <input
                            type="text"
                            value={card.input}
                            onChange={(event) =>
                              patchCard(skill.name, { input: event.target.value })
                            }
                            placeholder={requireInput ? "Input required" : "Optional input"}
                            aria-label={`Input for ${skill.name}`}
                            onKeyDown={(event) => {
                              if (event.key !== "Enter") {
                                return;
                              }
                              event.preventDefault();
                              if (card.running || (requireInput && card.input.trim().length === 0)) {
                                return;
                              }
                              void runNow(skill, card.input);
                            }}
                            className="min-w-48 flex-1 rounded-lg border border-neutral-800 bg-neutral-900 px-3 py-1.5 text-xs text-neutral-100 placeholder:text-neutral-500 outline-none focus:border-neutral-600 focus-visible:ring-2 focus-visible:ring-teal-500 focus-visible:ring-offset-1 focus-visible:ring-offset-black"
                          />
                          <button
                            type="button"
                            onClick={() => void runNow(skill, card.input)}
                            disabled={card.running || (requireInput && card.input.trim().length === 0)}
                            className={BUTTON_PRIMARY}
                          >
                            {card.running ? "Running…" : "Run"}
                          </button>
                        </>
                      ) : (
                        <button
                          type="button"
                          onClick={() => {
                            if (needsInput) {
                              patchCard(skill.name, { inputOpen: true });
                            } else {
                              void runNow(skill, "");
                            }
                          }}
                          disabled={card.running}
                          className={BUTTON_PRIMARY}
                        >
                          {card.running ? "Running…" : "Run"}
                        </button>
                      )}

                      {!skill.builtin ? (
                        isDeleting ? (
                          <>
                            <span className="self-center text-xs text-rose-300">
                              Delete this skill?
                            </span>
                            <button
                              type="button"
                              onClick={() => void handleDelete(skill)}
                              disabled={isBusyDelete}
                              className="rounded-lg border border-rose-700 bg-rose-900/50 px-3 py-1.5 text-xs font-semibold text-rose-100 transition-[background-color,border-color,transform] duration-[var(--dur-hover)] ease-[var(--ease-out)] hover:bg-rose-900/70 active:duration-[var(--dur-press)] active:scale-[0.98] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-rose-500 focus-visible:ring-offset-1 focus-visible:ring-offset-black disabled:opacity-50"
                            >
                              {isBusyDelete ? "Deleting…" : "Confirm delete"}
                            </button>
                            <button
                              type="button"
                              onClick={() => setDeleting(null)}
                              disabled={isBusyDelete}
                              className={BUTTON_NEUTRAL}
                            >
                              Cancel
                            </button>
                          </>
                        ) : (
                          <button
                            type="button"
                            onClick={() => setDeleting(skill.name)}
                            className="rounded-lg border border-neutral-800 bg-neutral-900 px-3 py-1.5 text-xs font-medium text-rose-300 transition-[background-color,border-color,transform] duration-[var(--dur-hover)] ease-[var(--ease-out)] hover:border-rose-700 hover:bg-neutral-800 active:duration-[var(--dur-press)] active:scale-[0.98] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-rose-500 focus-visible:ring-offset-1 focus-visible:ring-offset-black"
                          >
                            Delete
                          </button>
                        )
                      ) : null}
                    </div>

                    {card.error ? (
                      <p role="alert" className="mt-2 text-xs text-rose-400">
                        {card.error}
                      </p>
                    ) : null}

                    {card.result ? (
                      <div className="mt-3 rounded-lg border border-neutral-800 bg-neutral-900 p-3">
                        <div className="flex items-start justify-between gap-2">
                          <p className="min-w-0 flex-1 whitespace-pre-wrap text-sm leading-relaxed text-neutral-100">
                            {card.result.reply}
                          </p>
                          <button
                            type="button"
                            onClick={() =>
                              patchCard(skill.name, {
                                result: null,
                                followup: null,
                                taskState: {},
                                error: null,
                              })
                            }
                            aria-label={`Close ${skill.name} result`}
                            className="shrink-0 rounded-md border border-neutral-800 bg-neutral-900 px-2 py-0.5 text-xs leading-none text-neutral-400 transition-[background-color,border-color,color,transform] duration-[var(--dur-hover)] ease-[var(--ease-out)] hover:border-neutral-700 hover:text-neutral-200 active:duration-[var(--dur-press)] active:scale-[0.97] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 focus-visible:ring-offset-1 focus-visible:ring-offset-black"
                          >
                            ✕
                          </button>
                        </div>
                        <MessageExtras
                          memoryUsed={card.result.memory_used}
                          trace={card.result.trace}
                          followup={card.followup}
                          recordState={{}}
                          taskState={card.taskState}
                          busy={taskBusy}
                          onRecord={() => undefined}
                          onTask={(index, action) => {
                            if (runTurnId) {
                              void handleTask(skill.name, runTurnId, index, action);
                            }
                          }}
                        />
                      </div>
                    ) : null}
                  </li>
                );
              })}
            </ul>
          ) : null}
        </div>
      </div>
    </AppShell>
  );
}
