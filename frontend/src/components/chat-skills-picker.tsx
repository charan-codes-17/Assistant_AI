"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";

import {
  AuthError,
  getSkills,
  type FollowupResponse,
  type SkillOut,
  type SkillRunResponse,
} from "@/lib/api";

/** Everything the chat page needs to append one assistant message for a finished skill run. */
export type SkillResultPayload = {
  skillName: string;
  response: SkillRunResponse;
  followup: FollowupResponse | null;
};

type ChatSkillsPickerProps = {
  onSelect: (skill: SkillOut) => void;
  onClose: () => void;
};

/**
 * Compact skill menu for the chat composer: names only, anchored above the composer. Picking a
 * skill hands it to the page, which swaps the chat composer for the skill composer. No run and no
 * input happen here. All text is rendered as plain React nodes.
 */
export function ChatSkillsPicker({ onSelect, onClose }: ChatSkillsPickerProps) {
  const router = useRouter();
  const rootRef = useRef<HTMLDivElement>(null);
  const [skills, setSkills] = useState<SkillOut[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState(false);

  // Load the list once, on the first open.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const rows = await getSkills();
        if (cancelled) {
          return;
        }
        setSkills(rows);
      } catch (err) {
        if (cancelled) {
          return;
        }
        if (err instanceof AuthError) {
          router.replace("/login");
          return;
        }
        setLoadError(true);
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [router]);

  // Close on Escape and on any mousedown outside the menu. The sparkle toggle is exempt so a click
  // there stays a clean toggle instead of close-then-reopen.
  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        onClose();
      }
    }
    function onMouseDown(event: MouseEvent) {
      const target = event.target;
      if (!(target instanceof Element)) {
        return;
      }
      if (rootRef.current?.contains(target)) {
        return;
      }
      if (target.closest("[data-skills-toggle]")) {
        return;
      }
      onClose();
    }
    document.addEventListener("keydown", onKeyDown);
    document.addEventListener("mousedown", onMouseDown);
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      document.removeEventListener("mousedown", onMouseDown);
    };
  }, [onClose]);

  const all = skills ?? [];
  // Built-ins keep API order.
  const builtins = all.filter((skill) => skill.builtin);
  const custom = all.filter((skill) => !skill.builtin);

  return (
    <div
      ref={rootRef}
      role="dialog"
      aria-label="Skills menu"
      className="animate-fade-rise absolute bottom-full right-0 z-10 mb-2 w-64 max-w-full rounded-xl border border-neutral-800 bg-neutral-950 p-2 shadow-lg shadow-black/40"
    >
      <div className="chat-scroll max-h-[min(15rem,35vh)] overflow-y-auto">
        {loading ? <p className="px-2 py-2 text-xs text-neutral-500">Loading…</p> : null}

        {!loading && loadError ? (
          <p role="alert" className="px-2 py-2 text-xs text-neutral-500">
            Could not load skills
          </p>
        ) : null}

        {!loading && !loadError ? (
          <>
            {builtins.length > 0 ? (
              <div>
                <p className="px-2 pb-1 pt-1 text-xs font-medium uppercase tracking-wide text-neutral-500">
                  Built-in
                </p>
                <ul>
                  {builtins.map((skill) => (
                    <li key={skill.name}>
                      <button
                        type="button"
                        onClick={() => onSelect(skill)}
                        className="w-full rounded-lg px-2 py-1.5 text-left text-sm text-neutral-200 transition-[background-color,color,transform] duration-[var(--dur-hover)] ease-[var(--ease-out)] hover:bg-neutral-900 hover:text-teal-200 active:duration-[var(--dur-press)] active:scale-[0.99] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 focus-visible:ring-offset-1 focus-visible:ring-offset-black"
                      >
                        {skill.name}
                      </button>
                    </li>
                  ))}
                </ul>
              </div>
            ) : null}

            <div>
              <p className="px-2 pb-1 pt-2 text-xs font-medium uppercase tracking-wide text-neutral-500">
                Yours
              </p>
              {custom.length === 0 ? (
                <p className="px-2 pb-1 text-xs text-neutral-500">
                  No custom skills yet{" "}
                  <Link
                    href="/skills/create"
                    className="rounded text-teal-300 underline-offset-2 transition-[color,transform] duration-[var(--dur-hover)] ease-[var(--ease-out)] hover:text-teal-200 hover:underline active:scale-[0.97] active:duration-[var(--dur-press)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 focus-visible:ring-offset-1 focus-visible:ring-offset-black"
                  >
                    Create a skill
                  </Link>
                </p>
              ) : (
                <ul>
                  {custom.map((skill) => (
                    <li key={skill.name}>
                      <button
                        type="button"
                        onClick={() => onSelect(skill)}
                        className="w-full rounded-lg px-2 py-1.5 text-left text-sm text-neutral-200 transition-[background-color,color,transform] duration-[var(--dur-hover)] ease-[var(--ease-out)] hover:bg-neutral-900 hover:text-teal-200 active:duration-[var(--dur-press)] active:scale-[0.99] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 focus-visible:ring-offset-1 focus-visible:ring-offset-black"
                      >
                        {skill.name}
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </>
        ) : null}
      </div>
    </div>
  );
}
