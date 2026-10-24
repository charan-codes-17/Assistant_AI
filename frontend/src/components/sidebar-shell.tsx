"use client";

import Link from "next/link";
import { useState, type ReactNode } from "react";

/**
 * Presentational left rail. It owns no conversation data: the Recents area is passed in as
 * children, so the parent keeps all list behaviour. Only the Recents open/closed toggle is local.
 *
 * Width: 208px (md:w-52) on desktop with bg-neutral-950 and a right border; full width inside the
 * mobile drawer. All icons are inline SVG (lucide-react is not a dependency).
 *
 * activeHref marks the matching nav row with a teal bar and aria-current; it is optional so the
 * chat page keeps rendering the rail with no row marked.
 *
 * The Recents section (label + scrollable list) renders only when children are supplied: only the
 * chat page has a conversation list, so the non-chat pages show title, New chat, nav and the user
 * block with the spacer in between.
 */

type SidebarShellProps = {
  username: string | null;
  onNewChat: () => void;
  onLogout: () => void;
  onCollapse?: () => void;
  activeHref?: string;
  children?: ReactNode;
};

function PencilIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      className="h-4 w-4 shrink-0"
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

function ListIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      className="h-4 w-4 shrink-0"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M8 6h13" />
      <path d="M8 12h13" />
      <path d="M8 18h13" />
      <path d="M3 6h.01" />
      <path d="M3 12h.01" />
      <path d="M3 18h.01" />
    </svg>
  );
}

function BookIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      className="h-4 w-4 shrink-0"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
      <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2Z" />
    </svg>
  );
}

function UploadIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      className="h-4 w-4 shrink-0"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <path d="M17 8l-5-5-5 5" />
      <path d="M12 3v12" />
    </svg>
  );
}

function ChevronIcon({ open }: { open: boolean }) {
  return (
    <svg
      viewBox="0 0 24 24"
      className={`h-3 w-3 shrink-0 transition-transform ${open ? "rotate-90" : ""}`}
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="m9 18 6-6-6-6" />
    </svg>
  );
}

function ChevronLeftIcon() {
  return (
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
      <path d="m15 18-6-6 6-6" />
    </svg>
  );
}

const NAV_ROWS = [
  { href: "/tasks", label: "Tasks & Reminders", Icon: ListIcon },
  { href: "/memories", label: "Memory", Icon: BookIcon },
  { href: "/onboarding", label: "Import Knowledge", Icon: UploadIcon },
] as const;

export function SidebarShell({
  username,
  onNewChat,
  onLogout,
  onCollapse,
  activeHref,
  children,
}: SidebarShellProps) {
  const [recentsOpen, setRecentsOpen] = useState(true);
  const initial = username && username.length > 0 ? username.charAt(0).toUpperCase() : "U";
  // Only the chat page supplies a conversation list; without it there is nothing to toggle.
  const hasRecents = children !== undefined && children !== null;

  return (
    <div className="flex h-full min-h-0 w-full flex-col border-r border-neutral-800 bg-neutral-950 md:w-52">
      <div className="flex items-center justify-between px-3 py-3">
        <span className="text-sm font-semibold tracking-tight text-neutral-100">AI Assistant</span>
        {onCollapse ? (
          <button
            type="button"
            onClick={onCollapse}
            aria-label="Collapse sidebar"
            className="rounded-md p-1 text-neutral-400 transition-[background-color,border-color,transform] duration-[var(--dur-hover)] ease-[var(--ease-out)] hover:bg-neutral-900 hover:text-neutral-200 active:duration-[var(--dur-press)] active:scale-[0.98] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 focus-visible:ring-offset-1 focus-visible:ring-offset-black"
          >
            <ChevronLeftIcon />
          </button>
        ) : null}
      </div>

      <div className="px-2">
        <button
          type="button"
          onClick={onNewChat}
          className="flex w-full items-center gap-2 rounded-xl bg-neutral-900 px-3 py-2 text-sm font-medium text-neutral-200 transition-[background-color,border-color,transform] duration-[var(--dur-hover)] ease-[var(--ease-out)] hover:bg-neutral-800 active:duration-[var(--dur-press)] active:scale-[0.98] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 focus-visible:ring-offset-1 focus-visible:ring-offset-black"
        >
          <PencilIcon />
          New chat
        </button>
      </div>

      <nav className="mt-2 flex flex-col gap-0.5 px-2">
        {NAV_ROWS.map(({ href, label, Icon }) => {
          const isActive = activeHref === href;
          return (
            <Link
              key={href}
              href={href}
              aria-current={isActive ? "page" : undefined}
              className={`relative flex items-center gap-2 rounded-lg px-3 py-2 text-sm transition-[background-color,border-color,transform] duration-[var(--dur-hover)] ease-[var(--ease-out)] active:duration-[var(--dur-press)] active:scale-[0.98] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 focus-visible:ring-offset-1 focus-visible:ring-offset-black ${
                isActive
                  ? "bg-neutral-900 text-neutral-100 hover:bg-neutral-800"
                  : "text-neutral-200 hover:bg-neutral-900"
              }`}
            >
              {isActive ? (
                <span
                  aria-hidden="true"
                  className="absolute bottom-0 left-0 top-0 my-auto h-4 w-0.5 rounded-full bg-teal-500"
                />
              ) : null}
              <Icon />
              {label}
            </Link>
          );
        })}
      </nav>

      <div className="mt-4 flex min-h-0 flex-1 flex-col">
        {hasRecents ? (
          <>
            <button
              type="button"
              onClick={() => setRecentsOpen((value) => !value)}
              aria-expanded={recentsOpen}
              className="flex items-center gap-1 rounded-md px-4 py-1 text-xs font-medium uppercase tracking-wide text-neutral-500 transition-[background-color,border-color,transform] duration-[var(--dur-hover)] ease-[var(--ease-out)] hover:bg-neutral-900 hover:text-neutral-300 active:duration-[var(--dur-press)] active:scale-[0.98] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 focus-visible:ring-offset-1 focus-visible:ring-offset-black"
            >
              <ChevronIcon open={recentsOpen} />
              Recents
            </button>
            {recentsOpen ? (
              <div className="chat-scroll min-h-0 flex-1 overflow-y-auto px-2 pb-2">{children}</div>
            ) : null}
          </>
        ) : null}
      </div>

      <div className="border-t border-neutral-800 p-3">
        <div className="flex items-center gap-2">
          <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded bg-teal-500 text-xs font-semibold text-black">
            {initial}
          </span>
          <span className="truncate text-sm font-semibold text-neutral-200">
            {username ?? "signed in"}
          </span>
        </div>
        <button
          type="button"
          onClick={onLogout}
          className="mt-2 w-full rounded-md border border-neutral-800 bg-neutral-900 px-3 py-1.5 text-xs font-medium text-neutral-300 transition-[background-color,border-color,transform] duration-[var(--dur-hover)] ease-[var(--ease-out)] hover:bg-neutral-800 active:duration-[var(--dur-press)] active:scale-[0.98] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 focus-visible:ring-offset-1 focus-visible:ring-offset-black"
        >
          Log Out
        </button>
      </div>
    </div>
  );
}
