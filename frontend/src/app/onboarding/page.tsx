"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";

import {
  ApiError,
  AuthError,
  confirmBatch,
  getMe,
  getOnboardingPrompt,
  importOnboarding,
  rejectBatch,
  type OnboardingResult,
} from "@/lib/api";

/**
 * Mirrors backend settings.ONBOARDING_MAX_CHARS (backend/config/settings.py). The backend
 * rejects payloads over this with a 413, so the textarea is capped to match.
 */
const ONBOARDING_MAX_CHARS = 12000;

const GENERIC_ERROR = "Something went wrong. Please try again.";

function initialChecked(ids: number[]): Record<number, boolean> {
  const next: Record<number, boolean> = {};
  for (const id of ids) {
    next[id] = true;
  }
  return next;
}

export default function OnboardingPage() {
  const router = useRouter();

  // Auth bootstrap
  const [ready, setReady] = useState(false);

  // "Bring your AI knowledge" flow
  const [prompt, setPrompt] = useState<string | null>(null);
  const [promptLoading, setPromptLoading] = useState(false);
  const [copied, setCopied] = useState(false);
  const [promptError, setPromptError] = useState<string | null>(null);

  const [text, setText] = useState("");
  const [importing, setImporting] = useState(false);
  const [importError, setImportError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const [result, setResult] = useState<OnboardingResult | null>(null);
  const [checked, setChecked] = useState<Record<number, boolean>>({});
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [savedCount, setSavedCount] = useState<number | null>(null);

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

  const groupEntries = result ? Object.entries(result.groups) : [];
  const candidateIds = groupEntries.flatMap(([, items]) => items.map((item) => item.record_id));
  const selectedIds = candidateIds.filter((id) => checked[id]);

  async function handleCopyPrompt() {
    if (promptLoading) {
      return;
    }
    setPromptError(null);
    setCopied(false);
    setPromptLoading(true);
    try {
      let value = prompt;
      if (value === null) {
        const data = await getOnboardingPrompt();
        value = data.import_prompt;
        setPrompt(value);
      }
      try {
        await navigator.clipboard.writeText(value);
        setCopied(true);
      } catch {
        // Clipboard can be blocked; the prompt stays visible for manual copy.
        setPromptError("Could not copy automatically. Select the text below and copy it.");
      }
    } catch (err) {
      if (err instanceof AuthError) {
        router.replace("/login");
        return;
      }
      setPromptError("Could not load the prompt. Try again.");
    } finally {
      setPromptLoading(false);
    }
  }

  async function handleImport() {
    if (importing) {
      return;
    }
    setNotice(null);
    const payload = text.trim();
    if (!payload) {
      setImportError("Paste your AI's reply first.");
      return;
    }
    if (text.length > ONBOARDING_MAX_CHARS) {
      setImportError(`That's over the ${ONBOARDING_MAX_CHARS} character limit.`);
      return;
    }

    setImportError(null);
    setSaveError(null);
    setResult(null);
    setSavedCount(null);
    setImporting(true);
    try {
      const data = await importOnboarding(text);
      const ids = Object.values(data.groups).flat().map((item) => item.record_id);
      setResult(data);
      setChecked(initialChecked(ids));
    } catch (err) {
      if (err instanceof AuthError) {
        router.replace("/login");
        return;
      }
      if (err instanceof ApiError && err.status === 413) {
        setImportError(`That's over the ${ONBOARDING_MAX_CHARS} character limit.`);
      } else {
        setImportError(GENERIC_ERROR);
      }
    } finally {
      setImporting(false);
    }
  }

  async function handleSave() {
    if (saving || !result) {
      return;
    }
    if (selectedIds.length === 0) {
      setSaveError("Select at least one memory to save.");
      return;
    }
    setSaveError(null);
    setSaving(true);
    try {
      const data = await confirmBatch(selectedIds);
      setSavedCount(data.confirmed.length);
    } catch (err) {
      if (err instanceof AuthError) {
        router.replace("/login");
        return;
      }
      setSaveError(GENERIC_ERROR);
    } finally {
      setSaving(false);
    }
  }

  async function handleDiscard() {
    if (saving || !result) {
      return;
    }
    setSaveError(null);
    setSaving(true);
    try {
      if (candidateIds.length > 0) {
        // Batch discard exists (POST /memory/reject-batch): reject every candidate not saved.
        await rejectBatch(candidateIds);
      }
      setResult(null);
      setChecked({});
      setText("");
      setPromptError(null);
      setNotice("Import discarded. Nothing was saved.");
    } catch (err) {
      if (err instanceof AuthError) {
        router.replace("/login");
        return;
      }
      setSaveError(GENERIC_ERROR);
    } finally {
      setSaving(false);
    }
  }

  if (!ready) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-slate-950 p-6 text-sm text-slate-500">
        Loading…
      </div>
    );
  }

  return (
    <div className="flex min-h-screen flex-col bg-slate-950 text-slate-100">
      <header className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 px-4 py-3">
        <div className="flex items-center gap-3">
          <div className="h-2.5 w-2.5 rounded-full bg-emerald-500" />
          <span className="text-sm font-semibold tracking-tight text-white">
            Personal Memory Assistant
          </span>
        </div>
        <Link
          href="/"
          className="rounded-md border border-slate-700 bg-slate-900 px-3 py-1.5 text-xs font-medium text-slate-300 hover:border-slate-600"
        >
          Go to chat
        </Link>
      </header>

      <main className="mx-auto flex w-full max-w-3xl flex-1 flex-col gap-6 px-4 py-8">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white">Set up your memory</h1>
          <p className="mt-2 text-sm text-slate-400">
            Choose how the assistant should start. You can change your mind at any time.
          </p>
        </div>

        <div className="grid gap-4 md:grid-cols-2">
          {/* Start fresh */}
          <section className="flex flex-col rounded-xl border border-slate-800 bg-slate-900 p-5">
            <h2 className="text-base font-semibold text-white">Start fresh</h2>
            <p className="mt-2 flex-1 text-sm text-slate-400">
              The assistant starts with no imported context and learns as you chat. You approve
              every memory.
            </p>
            <Link
              href="/"
              className="mt-4 rounded-md border border-emerald-700 bg-emerald-900/60 px-4 py-2 text-center text-sm font-semibold text-emerald-100 hover:bg-emerald-900"
            >
              Start chatting
            </Link>
          </section>

          {/* Bring your AI knowledge */}
          <section className="flex flex-col rounded-xl border border-slate-800 bg-slate-900 p-5">
            <h2 className="text-base font-semibold text-white">Bring your AI knowledge</h2>
            <p className="mt-2 text-sm text-slate-400">
              Run a prompt in your other AI, then paste its reply here. This app cannot access any
              other AI account.
            </p>

            <div className="mt-3 flex flex-wrap items-center gap-2">
              <button
                type="button"
                onClick={handleCopyPrompt}
                disabled={promptLoading}
                className="rounded-md border border-slate-700 bg-slate-950 px-3 py-1.5 text-xs font-medium text-slate-200 hover:border-slate-600 disabled:opacity-50"
              >
                {promptLoading ? "Loading…" : "Copy prompt"}
              </button>
              {copied ? <span className="text-xs text-emerald-400">Copied</span> : null}
            </div>
            {promptError ? (
              <p role="alert" className="mt-2 text-xs text-rose-400">
                {promptError}
              </p>
            ) : null}
            {prompt ? (
              <textarea
                readOnly
                value={prompt}
                aria-label="Prompt to run in your other AI"
                className="mt-2 h-24 w-full resize-none rounded-md border border-slate-800 bg-slate-950 p-2 text-xs text-slate-400 outline-none"
              />
            ) : null}

            <label className="mt-3 flex flex-col gap-1 text-xs font-medium text-slate-400">
              Paste the reply here
              <textarea
                value={text}
                onChange={(event) => setText(event.target.value)}
                maxLength={ONBOARDING_MAX_CHARS}
                placeholder="Paste your AI's reply…"
                className="h-40 w-full resize-y rounded-md border border-slate-700 bg-slate-950 p-2 text-sm text-slate-100 outline-none placeholder:text-slate-600 focus:border-emerald-700"
              />
            </label>
            <div className="mt-1 flex items-center justify-between text-xs text-slate-500">
              <span>Nothing is saved until you review and confirm.</span>
              <span>
                {text.length} / {ONBOARDING_MAX_CHARS}
              </span>
            </div>

            {importError ? (
              <p role="alert" className="mt-2 text-xs text-rose-400">
                {importError}
              </p>
            ) : null}

            <button
              type="button"
              onClick={handleImport}
              disabled={importing || text.trim().length === 0}
              className="mt-3 rounded-md border border-emerald-700 bg-emerald-900/60 px-4 py-2 text-sm font-semibold text-emerald-100 hover:bg-emerald-900 disabled:opacity-50"
            >
              {importing ? "Importing…" : "Import"}
            </button>
          </section>
        </div>

        {notice ? (
          <p className="rounded-lg border border-slate-800 bg-slate-900 p-3 text-xs text-slate-300">
            {notice}
          </p>
        ) : null}

        {/* Saved state */}
        {savedCount !== null ? (
          <section className="rounded-xl border border-emerald-800/60 bg-emerald-950/30 p-5">
            <h2 className="text-base font-semibold text-emerald-200">
              {savedCount === 1 ? "1 memory saved" : `${savedCount} memories saved`}
            </h2>
            <p className="mt-2 text-sm text-emerald-100/80">
              They are part of your memory now and can be recalled in chat.
            </p>
            <button
              type="button"
              onClick={() => router.push("/")}
              className="mt-4 rounded-md border border-emerald-700 bg-emerald-900/60 px-4 py-2 text-sm font-semibold text-emerald-100 hover:bg-emerald-900"
            >
              Go to chat
            </button>
          </section>
        ) : null}

        {/* Review state */}
        {savedCount === null && result ? (
          <section className="flex flex-col gap-4 rounded-xl border border-slate-800 bg-slate-900 p-5">
            <div>
              <h2 className="text-base font-semibold text-white">Review before saving</h2>
              <p className="mt-1 text-sm text-slate-400">
                Check the memories you want to keep. Nothing has been saved yet.
              </p>
            </div>

            {result.duplicates_skipped > 0 ? (
              <p className="text-xs text-slate-400">
                {result.duplicates_skipped} item
                {result.duplicates_skipped === 1 ? "" : "s"} skipped as duplicates.
              </p>
            ) : null}

            {result.errors.length > 0 ? (
              <ul className="list-disc pl-5 text-xs text-amber-400">
                {result.errors.map((message, index) => (
                  <li key={index}>{message}</li>
                ))}
              </ul>
            ) : null}

            {candidateIds.length === 0 ? (
              <p className="text-sm text-slate-500">No memories were found in that text.</p>
            ) : (
              <div className="flex flex-col gap-4">
                {groupEntries.map(([group, items]) => (
                  <div key={group}>
                    <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                      {group}
                    </h3>
                    <ul className="mt-2 flex flex-col gap-2">
                      {items.map((item) => (
                        <li key={item.record_id}>
                          <label className="flex cursor-pointer items-start gap-2 rounded-md border border-slate-800 bg-slate-950 p-3">
                            <input
                              type="checkbox"
                              checked={Boolean(checked[item.record_id])}
                              onChange={(event) =>
                                setChecked((prev) => ({
                                  ...prev,
                                  [item.record_id]: event.target.checked,
                                }))
                              }
                              className="mt-0.5"
                            />
                            <span className="flex-1">
                              <span className="block whitespace-pre-wrap text-sm text-slate-100">
                                {item.content}
                              </span>
                              <span className="mt-1 block text-xs text-slate-500">
                                {item.kind} · {item.certainty}
                              </span>
                            </span>
                          </label>
                        </li>
                      ))}
                    </ul>
                  </div>
                ))}
              </div>
            )}

            {saveError ? (
              <p role="alert" className="text-xs text-rose-400">
                {saveError}
              </p>
            ) : null}

            <div className="flex flex-wrap gap-2">
              <button
                type="button"
                onClick={handleSave}
                disabled={saving || selectedIds.length === 0}
                className="rounded-md border border-emerald-700 bg-emerald-900/60 px-4 py-2 text-sm font-semibold text-emerald-100 hover:bg-emerald-900 disabled:opacity-50"
              >
                {saving ? "Saving…" : `Save selected (${selectedIds.length})`}
              </button>
              <button
                type="button"
                onClick={handleDiscard}
                disabled={saving}
                className="rounded-md border border-slate-700 bg-slate-900 px-4 py-2 text-sm font-medium text-slate-300 hover:border-slate-600 disabled:opacity-50"
              >
                Discard import
              </button>
            </div>
          </section>
        ) : null}
      </main>
    </div>
  );
}
