"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";

import { AppShell } from "@/components/app-shell";
import {
  ApiDetailError,
  AuthError,
  createSkill,
  getMe,
  getSkills,
  SKILL_DESCRIPTION_MAX,
  SKILL_INSTRUCTIONS_MAX,
  SKILL_NAME_PATTERN,
} from "@/lib/api";

/**
 * Client-side validation mirrors backend/api/schemas.py exactly, so the request only leaves the
 * browser when the backend would accept it. The backend stays the source of truth: its own
 * `detail` is shown verbatim when it still rejects the payload (409 duplicate, 422 reserved).
 */
const NAME_RE = new RegExp(SKILL_NAME_PATTERN);

const GENERIC_ERROR = "Something went wrong. Please try again.";

export default function CreateSkillPage() {
  const router = useRouter();

  const [ready, setReady] = useState(false);
  const [builtinNames, setBuiltinNames] = useState<Set<string>>(new Set());

  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [instructions, setInstructions] = useState("");

  const [formError, setFormError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

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

  // Built-in names are read from the API, never hardcoded: the reserved set is the backend's.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const rows = await getSkills();
        if (!cancelled) {
          setBuiltinNames(new Set(rows.filter((row) => row.builtin).map((row) => row.name)));
        }
      } catch {
        // Leave the set empty; the backend still rejects a reserved name with a 422.
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  function validate(): string | null {
    const trimmedName = name.trim();
    if (!trimmedName) {
      return "Enter a name.";
    }
    if (!NAME_RE.test(trimmedName)) {
      return 'Names are 2-40 characters of lowercase letters, digits, "-" or "_", and must start with a letter or digit.';
    }
    if (builtinNames.has(trimmedName)) {
      return "That name belongs to a built-in skill. Pick another name.";
    }
    if (description.trim().length > SKILL_DESCRIPTION_MAX) {
      return `Keep the description to ${SKILL_DESCRIPTION_MAX} characters.`;
    }
    if (!instructions.trim()) {
      return "The instructions cannot be empty.";
    }
    if (instructions.trim().length > SKILL_INSTRUCTIONS_MAX) {
      return `Keep the instructions to ${SKILL_INSTRUCTIONS_MAX} characters.`;
    }
    return null;
  }

  async function handleSubmit() {
    if (submitting) {
      return;
    }
    const error = validate();
    if (error) {
      setFormError(error);
      return;
    }

    setFormError(null);
    setSubmitting(true);
    try {
      await createSkill({
        name: name.trim(),
        description: description.trim(),
        instructions: instructions.trim(),
      });
      router.push("/skills");
    } catch (err) {
      if (err instanceof AuthError) {
        router.replace("/login");
        return;
      }
      if (err instanceof ApiDetailError && err.detail) {
        setFormError(err.detail);
      } else {
        setFormError(GENERIC_ERROR);
      }
    } finally {
      setSubmitting(false);
    }
  }

  if (!ready) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-black p-6 text-sm text-neutral-500">
        Loading…
      </div>
    );
  }

  return (
    <AppShell activeHref="/skills">
      <div className="mx-auto flex w-full max-w-3xl flex-1 flex-col gap-6 px-6 py-8">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-neutral-100">Create a skill</h1>
          <p className="mt-2 text-sm text-neutral-400">
            Describe what the skill should do. It runs against your own memory and never saves
            anything on its own.
          </p>
        </div>

        <section className="flex flex-col gap-4 rounded-xl border border-neutral-800 bg-neutral-950 p-5">
          <label className="flex flex-col gap-1 text-xs font-medium text-neutral-400">
            Name
            <input
              type="text"
              value={name}
              onChange={(event) => setName(event.target.value)}
              maxLength={40}
              placeholder="weekly-notes"
              className="w-full rounded-lg border border-neutral-800 bg-neutral-900 px-3 py-2 text-sm text-neutral-100 placeholder:text-neutral-600 outline-none focus:border-teal-600"
            />
            <span className="mt-0.5 font-normal text-neutral-500">
              Lowercase letters, digits, "-" or "_"; 2-40 characters.
            </span>
          </label>

          <label className="flex flex-col gap-1 text-xs font-medium text-neutral-400">
            Description
            <input
              type="text"
              value={description}
              onChange={(event) => setDescription(event.target.value)}
              maxLength={SKILL_DESCRIPTION_MAX}
              placeholder="One line about what this skill does"
              className="w-full rounded-lg border border-neutral-800 bg-neutral-900 px-3 py-2 text-sm text-neutral-100 placeholder:text-neutral-600 outline-none focus:border-teal-600"
            />
            <span className="mt-0.5 self-end font-normal text-neutral-500">
              {description.length} / {SKILL_DESCRIPTION_MAX}
            </span>
          </label>

          <label className="flex flex-col gap-1 text-xs font-medium text-neutral-400">
            Instructions
            <textarea
              value={instructions}
              onChange={(event) => setInstructions(event.target.value)}
              maxLength={SKILL_INSTRUCTIONS_MAX}
              placeholder="Tell the skill exactly what to write, and in what shape…"
              className="h-48 w-full resize-y rounded-lg border border-neutral-800 bg-neutral-900 p-2 text-sm text-neutral-100 outline-none placeholder:text-neutral-600 focus:border-teal-600"
            />
            <span className="mt-0.5 self-end font-normal text-neutral-500">
              {instructions.length} / {SKILL_INSTRUCTIONS_MAX}
            </span>
          </label>

          {formError ? (
            <p role="alert" className="text-xs text-rose-400">
              {formError}
            </p>
          ) : null}

          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              onClick={() => void handleSubmit()}
              disabled={submitting}
              className="rounded-lg border border-teal-700 bg-neutral-900 px-4 py-2 text-sm font-semibold text-teal-300 transition-[background-color,border-color,transform] duration-[var(--dur-hover)] ease-[var(--ease-out)] hover:bg-neutral-800 active:duration-[var(--dur-press)] active:scale-[0.98] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 focus-visible:ring-offset-1 focus-visible:ring-offset-black disabled:opacity-50"
            >
              {submitting ? "Creating…" : "Create skill"}
            </button>
            <button
              type="button"
              onClick={() => router.push("/skills")}
              disabled={submitting}
              className="rounded-lg border border-neutral-800 bg-neutral-900 px-4 py-2 text-sm font-medium text-neutral-300 transition-[background-color,border-color,transform] duration-[var(--dur-hover)] ease-[var(--ease-out)] hover:bg-neutral-800 active:duration-[var(--dur-press)] active:scale-[0.98] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 focus-visible:ring-offset-1 focus-visible:ring-offset-black disabled:opacity-50"
            >
              Cancel
            </button>
          </div>
        </section>
      </div>
    </AppShell>
  );
}
