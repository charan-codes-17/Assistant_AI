"use client";

import { useMemo, useState } from "react";

import { CONVERSATION_TITLE_MAX, type ConversationOut } from "@/lib/api";

/**
 * Left-hand chat history.
 *
 * Presentational + local UI state only: the conversation list itself is owned by the page and
 * passed in. Rename/Delete call the parent's async handlers and only leave their editing state
 * once the backend has confirmed success.
 *
 * Always visible on desktop; on mobile it is a slide-in drawer that is closed by default. Titles
 * are untrusted text and are rendered as plain text (never HTML).
 */

type ChatSidebarProps = {
  conversations: ConversationOut[];
  activeId: number | null;
  loading: boolean;
  error: boolean;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onRetry: () => void;
  onSelect: (id: number) => void;
  onNewChat: () => void;
  onRename: (id: number, title: string) => Promise<void>;
  onDelete: (id: number) => Promise<void>;
};

type ConversationGroup = {
  label: string;
  items: ConversationOut[];
};

const DAY_MS = 24 * 60 * 60 * 1000;

function startOfLocalDay(value: Date): number {
  return new Date(value.getFullYear(), value.getMonth(), value.getDate()).getTime();
}

/** Today / Previous 7 days / Older, using updated_at in the viewer's local time. */
function groupConversations(items: ConversationOut[], now: Date): ConversationGroup[] {
  const todayStart = startOfLocalDay(now);
  const weekStart = todayStart - 7 * DAY_MS;
  const today: ConversationOut[] = [];
  const week: ConversationOut[] = [];
  const older: ConversationOut[] = [];

  for (const item of items) {
    const parsed = new Date(item.updated_at);
    const time = Number.isNaN(parsed.getTime()) ? 0 : parsed.getTime();
    if (time >= todayStart) {
      today.push(item);
    } else if (time >= weekStart) {
      week.push(item);
    } else {
      older.push(item);
    }
  }

  const groups: ConversationGroup[] = [];
  if (today.length > 0) {
    groups.push({ label: "Today", items: today });
  }
  if (week.length > 0) {
    groups.push({ label: "Previous 7 days", items: week });
  }
  if (older.length > 0) {
    groups.push({ label: "Older", items: older });
  }
  return groups;
}

export function ChatSidebar({
  conversations,
  activeId,
  loading,
  error,
  open,
  onOpenChange,
  onRetry,
  onSelect,
  onNewChat,
  onRename,
  onDelete,
}: ChatSidebarProps) {
  const [query, setQuery] = useState("");
  const [editingId, setEditingId] = useState<number | null>(null);
  const [editValue, setEditValue] = useState("");
  const [confirmingId, setConfirmingId] = useState<number | null>(null);
  const [busyId, setBusyId] = useState<number | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

  const now = useMemo(() => new Date(), []);
  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) {
      return conversations;
    }
    return conversations.filter((item) => item.title.toLowerCase().includes(needle));
  }, [conversations, query]);
  const groups = useMemo(() => groupConversations(filtered, now), [filtered, now]);

  const startRename = (item: ConversationOut) => {
    setConfirmingId(null);
    setActionError(null);
    setEditingId(item.id);
    setEditValue(item.title.slice(0, CONVERSATION_TITLE_MAX));
  };

  const cancelRename = () => {
    setEditingId(null);
    setEditValue("");
  };

  const submitRename = async (id: number) => {
    const title = editValue.trim();
    if (title.length === 0) {
      return;
    }
    setBusyId(id);
    setActionError(null);
    try {
      await onRename(id, title.slice(0, CONVERSATION_TITLE_MAX));
      cancelRename();
    } catch {
      setActionError("Could not rename. Try again.");
    } finally {
      setBusyId(null);
    }
  };

  const submitDelete = async (id: number) => {
    setBusyId(id);
    setActionError(null);
    try {
      await onDelete(id);
      setConfirmingId(null);
    } catch {
      setActionError("Could not delete. Try again.");
    } finally {
      setBusyId(null);
    }
  };

  const handleSelect = (id: number) => {
    onSelect(id);
    onOpenChange(false);
  };

  const panel = (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex items-center justify-between gap-2 border-b border-slate-800 px-4 py-3">
        <h2 className="text-sm font-semibold text-white">Chats</h2>
        <button
          type="button"
          onClick={() => {
            onNewChat();
            onOpenChange(false);
          }}
          className="rounded-md border border-emerald-700 bg-emerald-900/60 px-3 py-1.5 text-xs font-semibold text-emerald-100 hover:bg-emerald-900"
        >
          New chat
        </button>
      </div>

      <div className="px-4 py-2">
        <input
          type="text"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search chats"
          aria-label="Search chats by title"
          className="w-full rounded-md border border-slate-700 bg-slate-900 px-3 py-1.5 text-sm text-slate-100 outline-none placeholder:text-slate-600 focus:border-emerald-700"
        />
      </div>

      {actionError ? (
        <p role="alert" className="px-4 pb-1 text-xs text-rose-400">
          {actionError}
        </p>
      ) : null}

      <div className="min-h-0 flex-1 overflow-y-auto px-2 pb-2">
        {loading ? (
          <p className="px-2 py-3 text-xs text-slate-500">Loading…</p>
        ) : error ? (
          <div className="px-2 py-3 text-xs text-slate-400">
            <p>Could not load chats.</p>
            <button
              type="button"
              onClick={onRetry}
              className="mt-2 rounded-md border border-slate-700 bg-slate-900 px-2.5 py-1 text-xs font-medium text-slate-300 hover:border-slate-600"
            >
              Retry
            </button>
          </div>
        ) : conversations.length === 0 ? (
          <p className="px-2 py-3 text-xs text-slate-500">No chats yet</p>
        ) : filtered.length === 0 ? (
          <p className="px-2 py-3 text-xs text-slate-500">No chats match your search</p>
        ) : (
          groups.map((group) => (
            <div key={group.label} className="mt-1">
              <p className="px-2 py-1 text-[11px] font-medium uppercase tracking-wide text-slate-500">
                {group.label}
              </p>
              <ul className="space-y-0.5">
                {group.items.map((item) => {
                  const active = item.id === activeId;

                  if (editingId === item.id) {
                    return (
                      <li key={item.id}>
                        <form
                          onSubmit={(event) => {
                            event.preventDefault();
                            void submitRename(item.id);
                          }}
                          className="flex items-center gap-1 px-1 py-1"
                        >
                          <input
                            autoFocus
                            type="text"
                            value={editValue}
                            maxLength={CONVERSATION_TITLE_MAX}
                            onChange={(event) => setEditValue(event.target.value)}
                            aria-label="New chat title"
                            className="min-w-0 flex-1 rounded-md border border-slate-700 bg-slate-900 px-2 py-1 text-sm text-slate-100 outline-none focus:border-emerald-700"
                          />
                          <button
                            type="submit"
                            disabled={busyId === item.id || editValue.trim().length === 0}
                            className="rounded-md border border-emerald-800/60 bg-emerald-950/40 px-2 py-1 text-xs font-medium text-emerald-300 hover:border-emerald-700 disabled:opacity-50"
                          >
                            Save
                          </button>
                          <button
                            type="button"
                            onClick={cancelRename}
                            className="rounded-md border border-slate-700 bg-slate-900 px-2 py-1 text-xs font-medium text-slate-300 hover:border-slate-600"
                          >
                            Cancel
                          </button>
                        </form>
                      </li>
                    );
                  }

                  if (confirmingId === item.id) {
                    return (
                      <li key={item.id} className="px-2 py-1.5">
                        <p className="text-xs text-slate-300">
                          Delete this chat? This cannot be undone.
                        </p>
                        <div className="mt-2 flex gap-2">
                          <button
                            type="button"
                            disabled={busyId === item.id}
                            onClick={() => void submitDelete(item.id)}
                            className="rounded-md border border-rose-800/60 bg-rose-950/40 px-2.5 py-1 text-xs font-medium text-rose-300 hover:border-rose-700 disabled:opacity-50"
                          >
                            Delete
                          </button>
                          <button
                            type="button"
                            onClick={() => setConfirmingId(null)}
                            className="rounded-md border border-slate-700 bg-slate-900 px-2.5 py-1 text-xs font-medium text-slate-300 hover:border-slate-600"
                          >
                            Cancel
                          </button>
                        </div>
                      </li>
                    );
                  }

                  return (
                    <li key={item.id}>
                      <div
                        className={`flex items-center gap-1 rounded-md ${
                          active ? "bg-slate-800" : "hover:bg-slate-900"
                        }`}
                      >
                        <button
                          type="button"
                          onClick={() => handleSelect(item.id)}
                          aria-current={active ? "true" : undefined}
                          className="flex min-w-0 flex-1 items-center gap-2 px-2 py-1.5 text-left text-sm"
                        >
                          <span
                            aria-hidden="true"
                            className={active ? "text-emerald-400" : "text-transparent"}
                          >
                            ▍
                          </span>
                          <span
                            className={`truncate ${
                              active ? "font-semibold text-white" : "text-slate-300"
                            }`}
                          >
                            {item.title}
                          </span>
                        </button>
                        <button
                          type="button"
                          onClick={() => startRename(item)}
                          aria-label={`Rename ${item.title}`}
                          className="shrink-0 rounded-md px-1.5 py-1 text-xs text-slate-400 hover:text-slate-200"
                        >
                          Rename
                        </button>
                        <button
                          type="button"
                          onClick={() => {
                            setEditingId(null);
                            setActionError(null);
                            setConfirmingId(item.id);
                          }}
                          aria-label={`Delete ${item.title}`}
                          className="shrink-0 rounded-md px-1.5 py-1 text-xs text-slate-400 hover:text-rose-300"
                        >
                          Delete
                        </button>
                      </div>
                    </li>
                  );
                })}
              </ul>
            </div>
          ))
        )}
      </div>

      <p className="border-t border-slate-800 px-4 py-3 text-xs text-slate-500">
        Chats are saved to your account. Delete any chat here.
      </p>
    </div>
  );

  return (
    <>
      <aside className="hidden w-72 shrink-0 flex-col border-r border-slate-800 bg-slate-950 md:flex">
        {panel}
      </aside>

      {open ? (
        <div className="fixed inset-0 z-40 md:hidden">
          <div
            className="absolute inset-0 bg-black/60"
            onClick={() => onOpenChange(false)}
            aria-hidden="true"
          />
          <div className="absolute inset-y-0 left-0 flex w-72 max-w-[85%] flex-col border-r border-slate-800 bg-slate-950">
            <div className="flex items-center justify-between border-b border-slate-800 px-4 py-3">
              <span className="text-sm font-semibold text-white">Chats</span>
              <button
                type="button"
                onClick={() => onOpenChange(false)}
                aria-label="Close chat list"
                className="rounded-md border border-slate-700 bg-slate-900 px-2.5 py-1 text-xs font-medium text-slate-300 hover:border-slate-600"
              >
                Close
              </button>
            </div>
            {panel}
          </div>
        </div>
      ) : null}
    </>
  );
}
