"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";

import {
  ApiError,
  AuthError,
  SKILL_INPUT_MAX,
  getFollowup,
  runSkill,
  type FollowupResponse,
  type SkillOut,
} from "@/lib/api";
import type { SkillResultPayload } from "@/components/chat-skills-picker";

type ChatSkillComposerProps = {
  skill: SkillOut;
  onResult: (payload: SkillResultPayload) => void;
  onExit: () => void;
};

const GENERIC_ERROR = "Something went wrong, try again";

/** Friendly, status-based text; the raw server message is never shown. */
function statusMessage(status: number): string {
  if (status === 404) {
    return "Skill not found";
  }
  if (status === 422) {
    return "This skill needs some input";
  }
  if (status === 429) {
    return "Too many skill runs, try again in a few minutes";
  }
  return GENERIC_ERROR;
}

/**
 * Skill-mode composer that takes the place of the chat composer. It runs one skill and hands the
 * finished run to the page; it never calls /chat. Its own draft lives here, so entering and leaving
 * skill mode never touches the chat draft.
 */
export function ChatSkillComposer({ skill, onResult, onExit }: ChatSkillComposerProps) {
  const router = useRouter();
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const [input, setInput] = useState("");
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const needsInput = skill.input !== "none";
  const requireInput = skill.input === "required";
  const runDisabled = running || (requireInput && input.trim().length === 0);

  // Auto-grow up to about six lines, then let it scroll internally.
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) {
      return;
    }
    const LINE_HEIGHT = 24;
    const MAX_HEIGHT = LINE_HEIGHT * 6 + 8;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, MAX_HEIGHT)}px`;
  }, [input]);

  // Escape exits skill mode regardless of what is focused.
  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        onExit();
      }
    }
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [onExit]);

  async function run() {
    if (running) {
      return;
    }
    if (requireInput && input.trim().length === 0) {
      return;
    }
    setRunning(true);
    setError(null);
    try {
      const response = await runSkill(skill.name, input);
      let followup: FollowupResponse | null = null;
      if (response.pending_tasks_count > 0) {
        try {
          const fetched = await getFollowup(response.turn_id);
          followup = fetched.status === "ready" ? fetched : null;
        } catch {
          // The reply still renders; the candidate cards simply stay hidden.
          followup = null;
        }
      }
      onResult({ skillName: skill.name, response, followup });
      onExit();
    } catch (err) {
      if (err instanceof AuthError) {
        router.replace("/login");
        return;
      }
      setError(err instanceof ApiError ? statusMessage(err.status) : GENERIC_ERROR);
      setRunning(false);
    }
  }

  return (
    <div>
      {error ? (
        <p role="alert" className="mb-2 text-xs text-rose-400">
          {error}
        </p>
      ) : null}

      <div className="flex w-full items-end gap-2 rounded-2xl border border-neutral-800 bg-neutral-900 px-4 py-2 focus-within:border-neutral-600">
        <span className="flex shrink-0 items-center gap-1 py-1 text-sm font-medium text-teal-300">
          /{skill.name}
          <button
            type="button"
            onClick={onExit}
            aria-label="Exit skill mode"
            title="Exit skill mode"
            className="rounded p-0.5 text-teal-300 transition-[background-color,color,transform] duration-[var(--dur-hover)] ease-[var(--ease-out)] hover:bg-neutral-900 hover:text-teal-100 active:scale-[0.97] active:duration-[var(--dur-press)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 focus-visible:ring-offset-1 focus-visible:ring-offset-black"
          >
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
              <path d="M18 6 6 18" />
              <path d="m6 6 12 12" />
            </svg>
          </button>
        </span>

        {needsInput ? (
          <textarea
            ref={textareaRef}
            autoFocus
            rows={1}
            value={input}
            maxLength={SKILL_INPUT_MAX}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key !== "Enter" || event.shiftKey || event.nativeEvent.isComposing) {
                return;
              }
              event.preventDefault();
              if (runDisabled) {
                return;
              }
              void run();
            }}
            placeholder={
              requireInput ? `Input for ${skill.name}` : `Input for ${skill.name} (optional)`
            }
            aria-label={`Input for ${skill.name}`}
            className="chat-scroll min-w-0 flex-1 resize-none bg-transparent py-1 text-sm leading-6 text-neutral-100 placeholder:text-neutral-500 outline-none"
          />
        ) : (
          <span className="flex-1 py-1 text-sm text-neutral-500">No input needed</span>
        )}

        <button
          type="button"
          onClick={() => void run()}
          disabled={runDisabled}
          aria-label="Run skill"
          title="Run skill"
          className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full transition-[background-color,border-color,transform] duration-[var(--dur-hover)] ease-[var(--ease-out)] active:scale-[0.97] active:duration-[var(--dur-press)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 focus-visible:ring-offset-1 focus-visible:ring-offset-black disabled:cursor-not-allowed ${
            runDisabled ? "bg-neutral-800 text-neutral-600" : "bg-teal-500 text-black hover:bg-teal-400"
          }`}
        >
          {running ? (
            <svg
              viewBox="0 0 24 24"
              className="h-4 w-4 animate-spin"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              aria-hidden="true"
            >
              <circle cx="12" cy="12" r="9" className="opacity-25" />
              <path d="M21 12a9 9 0 0 0-9-9" />
            </svg>
          ) : (
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
          )}
        </button>
      </div>
    </div>
  );
}
