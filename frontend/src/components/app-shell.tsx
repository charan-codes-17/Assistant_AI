"use client";

import { useCallback, useEffect, useState, type ReactNode } from "react";
import { useRouter } from "next/navigation";

import { SidebarShell } from "@/components/sidebar-shell";
import { AuthError, getMe, logout } from "@/lib/api";

/**
 * Shared shell for the non-chat pages (/tasks, /memories, /onboarding): the same left rail as the
 * chat page, reusing SidebarShell, with the page content rendered in the main area.
 *
 * It only owns the authenticated username and the mobile drawer state. The rail's own chrome
 * (title, New chat, nav, Recents label, user block) lives in SidebarShell.
 */

type AppShellProps = {
  activeHref?: string;
  children: ReactNode;
};

export function AppShell({ activeHref, children }: AppShellProps) {
  const router = useRouter();

  const [username, setUsername] = useState<string | null>(null);
  const [drawerOpen, setDrawerOpen] = useState(false);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const me = await getMe();
        if (!cancelled) {
          setUsername(me.username ?? null);
        }
      } catch (err) {
        if (cancelled) {
          return;
        }
        if (err instanceof AuthError) {
          router.replace("/login");
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [router]);

  const handleLogout = useCallback(async () => {
    try {
      await logout();
    } catch (err) {
      // The cookie is cleared server-side when possible; any failure still leaves this page.
      void err;
    }
    router.replace("/login");
  }, [router]);

  const handleNewChat = useCallback(() => {
    setDrawerOpen(false);
    router.push("/");
  }, [router]);

  const railChildren = (
    <p className="px-3 py-3 text-xs text-neutral-500">Open a chat from the home page.</p>
  );

  return (
    <div className="flex min-h-screen bg-black text-neutral-200">
      <aside className="hidden shrink-0 md:flex">
        <SidebarShell
          username={username}
          onNewChat={handleNewChat}
          onLogout={handleLogout}
          activeHref={activeHref}
        >
          {railChildren}
        </SidebarShell>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <div className="flex items-center border-b border-neutral-800 px-3 py-2 md:hidden">
          <button
            type="button"
            onClick={() => setDrawerOpen(true)}
            aria-label="Open chats"
            className="rounded-md border border-neutral-800 bg-neutral-900 px-3 py-1.5 text-xs font-medium text-neutral-300 transition-[background-color,border-color,transform] duration-[var(--dur-hover)] ease-[var(--ease-out)] hover:border-neutral-700 hover:bg-neutral-800 active:duration-[var(--dur-press)] active:scale-[0.98] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 focus-visible:ring-offset-1 focus-visible:ring-offset-black"
          >
            Menu
          </button>
        </div>

        <main className="flex min-w-0 flex-1 flex-col">{children}</main>
      </div>

      {drawerOpen ? (
        <div className="fixed inset-0 z-40 md:hidden">
          <div
            className="absolute inset-0 bg-black/60"
            onClick={() => setDrawerOpen(false)}
            aria-hidden="true"
          />
          <div className="absolute inset-y-0 left-0 flex w-72 max-w-[85%] flex-col">
            <SidebarShell
              username={username}
              onNewChat={handleNewChat}
              onLogout={handleLogout}
              onCollapse={() => setDrawerOpen(false)}
              activeHref={activeHref}
            >
              {railChildren}
            </SidebarShell>
          </div>
        </div>
      ) : null}
    </div>
  );
}
