"use client";

import { useState } from "react";

import { SidebarShell } from "@/components/sidebar-shell";
import { CONVERSATION_TITLE_MAX, type ConversationOut } from "@/lib/api";

/**
 * Left-hand chat history.
 *
 * Presentational + local UI state only: the conversation list itself is owned by the page and
 * passed in. Rename/Delete call the parent's async handlers and only leave their editing state
 * once the backend has confirmed success.
 *
 * The rail chrome (title, New chat, nav, Recents label, user block) lives in SidebarShell; the
 * conversation rows are passed to it as children, both in the desktop aside and in the mobile
 * drawer. Titles are untrusted text and are rendered as plain text (never HTML).
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
  username: string | null;
  onLogout: () => void;
  collapsed?: boolean;
  onCollapse?: () => void;
};

function RenameIcon() {
  return (
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
      <path d="M12 20h9" />
      <path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4Z" />
    </svg>
  );
}

function DeleteIcon() {
  return (
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
      <path d="M3 6h18" />
      <path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
      <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
      <path d="M10 11v6" />
      <path d="M14 11v6" />
    </svg>
  );
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
  username,
  onLogout,
  collapsed = false,
  onCollapse,
}: ChatSidebarProps) {
  const [editingId, setEditingId] = useState<number | null>(null);
  const [editValue, setEditValue] = useState("");
  const [confirmingId, setConfirmingId] = useState<number | null>(null);
  const [busyId, setBusyId] = useState<number | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

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

  const handleNewChat = () => {
    onNewChat();
    onOpenChange(false);
  };

  const list = (
    <div className="pb-2">
      {actionError ? (
        <p role="alert" className="px-3 pb-1 text-xs text-rose-400">
          {actionError}
        </p>
      ) : null}

      {loading ? (
        <p className="px-3 py-3 text-xs text-neutral-500">Loading…</p>
      ) : error ? (
        <div className="px-3 py-3 text-xs text-neutral-400">
          <p>Could not load chats.</p>
          <button
            type="button"
            onClick={onRetry}
            className="mt-2 rounded-md border border-neutral-800 bg-neutral-900 px-2.5 py-1 text-xs font-medium text-neutral-300 hover:bg-neutral-800"
          >
            Retry
          </button>
        </div>
      ) : conversations.length === 0 ? (
        <p className="px-3 py-3 text-xs text-neutral-500">No chats yet</p>
      ) : (
        <ul className="space-y-0.5">
          {conversations.map((item) => {
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
                      className="min-w-0 flex-1 rounded-md border border-neutral-800 bg-neutral-900 px-2 py-1 text-sm text-neutral-100 outline-none focus:border-neutral-600"
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
                      className="rounded-md border border-neutral-800 bg-neutral-900 px-2 py-1 text-xs font-medium text-neutral-300 hover:bg-neutral-800"
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
                  <p className="whitespace-normal text-xs text-neutral-300">
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
                      className="rounded-md border border-neutral-800 bg-neutral-900 px-2.5 py-1 text-xs font-medium text-neutral-300 hover:bg-neutral-800"
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
                  className={`group flex items-center gap-1 rounded-lg px-1 ${
                    active ? "bg-neutral-900" : "hover:bg-neutral-900"
                  }`}
                >
                  <button
                    type="button"
                    onClick={() => handleSelect(item.id)}
                    aria-current={active ? "true" : undefined}
                    className="flex min-w-0 flex-1 items-center gap-2 px-1 py-1.5 text-left text-sm"
                  >
                    <span
                      aria-hidden="true"
                      className={active ? "text-teal-400" : "text-transparent"}
                    >
                      ▍
                    </span>
                    <span
                      className={`truncate ${
                        active ? "font-semibold text-neutral-100" : "text-neutral-300"
                      }`}
                    >
                      {item.title}
                    </span>
                  </button>
                  <button
                    type="button"
                    onClick={() => startRename(item)}
                    aria-label={`Rename ${item.title}`}
                    className="shrink-0 rounded-md p-1 text-neutral-400 opacity-0 hover:text-neutral-200 focus:opacity-100 group-hover:opacity-100 group-focus-within:opacity-100"
                  >
                    <RenameIcon />
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setEditingId(null);
                      setActionError(null);
                      setConfirmingId(item.id);
                    }}
                    aria-label={`Delete ${item.title}`}
                    className="shrink-0 rounded-md p-1 text-neutral-400 opacity-0 hover:text-rose-300 focus:opacity-100 group-hover:opacity-100 group-focus-within:opacity-100"
                  >
                    <DeleteIcon />
                  </button>
                </div>
              </li>
            );
          })}
        </ul>
      )}

      <p className="px-3 pt-2 text-xs text-neutral-500">
        Chats are saved to your account. Delete any chat here.
      </p>
    </div>
  );

  return (
    <>
      <aside className={collapsed ? "hidden" : "hidden shrink-0 md:flex"}>
        <SidebarShell
          username={username}
          onNewChat={handleNewChat}
          onLogout={onLogout}
          onCollapse={onCollapse}
        >
          {list}
        </SidebarShell>
      </aside>

      {open ? (
        <div className="fixed inset-0 z-40 md:hidden">
          <div
            className="absolute inset-0 bg-black/60"
            onClick={() => onOpenChange(false)}
            aria-hidden="true"
          />
          <div className="absolute inset-y-0 left-0 flex w-72 max-w-[85%] flex-col">
            <SidebarShell
              username={username}
              onNewChat={handleNewChat}
              onLogout={onLogout}
              onCollapse={() => onOpenChange(false)}
            >
              {list}
            </SidebarShell>
          </div>
        </div>
      ) : null}
    </>
  );
}
