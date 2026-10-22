"use client";

import { useState } from "react";

import type {
  FollowupResponse,
  InterventionDecision,
  MemoryUsedItem,
  TraceInfo,
} from "@/lib/api";

export type RecordState = "saved" | "discarded";
export type TaskState = "confirmed" | "discarded";

type MessageExtrasProps = {
  memoryUsed: MemoryUsedItem[];
  trace: TraceInfo | null;
  followup: FollowupResponse | null;
  recordState: Record<number, RecordState>;
  taskState: Record<number, TaskState>;
  busy: boolean;
  onRecord: (recordId: number, action: "save" | "discard") => void;
  onTask: (index: number, action: "save" | "discard") => void;
};

/** "20 Oct 2026", or null when there is no usable timestamp. */
function formatDate(value: string | null): string | null {
  if (!value) {
    return null;
  }
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return null;
  }
  return new Intl.DateTimeFormat("en-GB", {
    day: "numeric",
    month: "short",
    year: "numeric",
  }).format(parsed);
}

function formatDueDate(value: string | null): string {
  return formatDate(value) ?? "no date";
}

/**
 * Deterministic notice text, built in code. Shown only when the intervention flagged a
 * contradiction AND named a conflicting item. Matches the bot's wording; exactly one trailing
 * period is stripped before the template's own period is appended.
 */
function contradictionNotice(
  intervention: InterventionDecision,
): { text: string; question: string | null } | null {
  if (!intervention.contradiction) {
    return null;
  }
  const ref = intervention.conflicts_with;
  if (!ref) {
    return null;
  }
  const detail = intervention.contradiction_detail ?? `your earlier ${ref.kind}`;
  return {
    text: `Heads up: this may conflict with something you told me earlier: ${detail.replace(
      /\.$/,
      "",
    )}.`,
    question: intervention.clarifying_question,
  };
}

function Inspector({ items }: { items: MemoryUsedItem[] }) {
  const [open, setOpen] = useState(false);

  if (items.length === 0) {
    return null;
  }
  const topConfidence = Math.max(...items.map((item) => item.confidence));

  return (
    <div className="mt-2">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
        className="rounded-full border border-emerald-800/60 bg-emerald-950/40 px-2.5 py-0.5 text-xs font-medium text-emerald-300 hover:border-emerald-700 hover:text-emerald-200"
      >
        relevance {topConfidence}%
      </button>

      {open ? (
        <ul className="mt-2 space-y-2">
          {items.map((item) => {
            const date = formatDate(item.created_at);
            return (
              <li
                key={item.record_id}
                className="rounded-lg border border-slate-800 bg-slate-950/60 p-3"
              >
                <p className="text-sm leading-relaxed text-slate-200">{item.content}</p>
                <p className="mt-2 text-xs text-slate-500">
                  {item.kind} · {item.certainty} · {item.source}
                  {date ? ` · ${date}` : ""}
                </p>
              </li>
            );
          })}
        </ul>
      ) : null}
    </div>
  );
}

function Notice({ text, question }: { text: string; question: string | null }) {
  return (
    <div className="mt-2 rounded-lg border border-amber-800/60 bg-amber-950/30 p-3 text-xs leading-relaxed text-amber-200">
      <p>{text}</p>
      {question ? <p className="mt-1">{question}</p> : null}
    </div>
  );
}

function PendingRecords({
  followup,
  recordState,
  busy,
  onRecord,
}: {
  followup: FollowupResponse;
  recordState: Record<number, RecordState>;
  busy: boolean;
  onRecord: (recordId: number, action: "save" | "discard") => void;
}) {
  if (followup.pending_records.length === 0) {
    return null;
  }
  return (
    <ul className="mt-2 space-y-2">
      {followup.pending_records.map((record) => {
        const state = recordState[record.id];
        return (
          <li
            key={record.id}
            className="rounded-lg border border-slate-800 bg-slate-950/60 p-3"
          >
            <p className="text-sm leading-relaxed text-slate-200">{record.content}</p>
            <p className="mt-1 text-xs text-slate-500">
              {record.kind} · {record.certainty}
            </p>
            {state ? (
              <p className="mt-2 text-xs text-slate-400">
                {state === "saved" ? "Saved" : "Discarded"}
              </p>
            ) : (
              <div className="mt-2 flex gap-2">
                <button
                  type="button"
                  disabled={busy}
                  onClick={() => onRecord(record.id, "save")}
                  className="rounded-md border border-emerald-800/60 bg-emerald-950/40 px-2.5 py-1 text-xs font-medium text-emerald-300 hover:border-emerald-700 disabled:opacity-50"
                >
                  ✅ Save
                </button>
                <button
                  type="button"
                  disabled={busy}
                  onClick={() => onRecord(record.id, "discard")}
                  className="rounded-md border border-slate-700 bg-slate-900 px-2.5 py-1 text-xs font-medium text-slate-300 hover:border-slate-600 disabled:opacity-50"
                >
                  ❌ Discard
                </button>
              </div>
            )}
          </li>
        );
      })}
    </ul>
  );
}

function PendingTasks({
  followup,
  taskState,
  busy,
  onTask,
}: {
  followup: FollowupResponse;
  taskState: Record<number, TaskState>;
  busy: boolean;
  onTask: (index: number, action: "save" | "discard") => void;
}) {
  if (followup.pending_tasks.length === 0) {
    return null;
  }
  return (
    <ul className="mt-2 space-y-2">
      {followup.pending_tasks.map((task) => {
        const state = taskState[task.index];
        return (
          <li
            key={task.index}
            className="rounded-lg border border-slate-800 bg-slate-950/60 p-3"
          >
            <p className="text-sm leading-relaxed text-slate-200">{task.title}</p>
            <p className="mt-1 text-xs text-slate-500">{formatDueDate(task.due_at)}</p>
            {state ? (
              <p className="mt-2 text-xs text-slate-400">
                {state === "confirmed" ? "Confirmed" : "Discarded"}
              </p>
            ) : (
              <div className="mt-2 flex gap-2">
                <button
                  type="button"
                  disabled={busy}
                  onClick={() => onTask(task.index, "save")}
                  className="rounded-md border border-emerald-800/60 bg-emerald-950/40 px-2.5 py-1 text-xs font-medium text-emerald-300 hover:border-emerald-700 disabled:opacity-50"
                >
                  ✅
                </button>
                <button
                  type="button"
                  disabled={busy}
                  onClick={() => onTask(task.index, "discard")}
                  className="rounded-md border border-slate-700 bg-slate-900 px-2.5 py-1 text-xs font-medium text-slate-300 hover:border-slate-600 disabled:opacity-50"
                >
                  ❌
                </button>
              </div>
            )}
          </li>
        );
      })}
    </ul>
  );
}

function TraceDisclosure({ trace }: { trace: TraceInfo }) {
  return (
    <details className="mt-2 text-xs text-slate-500">
      <summary className="cursor-pointer select-none hover:text-slate-400">Trace</summary>
      <div className="mt-1 space-y-0.5 font-mono">
        <p>retrieval_ran: {trace.retrieval_ran ? "true" : "false"}</p>
        <p>candidates_considered: {trace.candidates_considered}</p>
        {trace.errors.length > 0 ? (
          trace.errors.map((message, index) => (
            <p key={index}>error: {message}</p>
          ))
        ) : (
          <p>errors: none</p>
        )}
      </div>
    </details>
  );
}

/**
 * Everything rendered under one assistant message, in the required order:
 * Inspector badge, then contradiction notice, then pending records/tasks, then the trace.
 */
export function MessageExtras({
  memoryUsed,
  trace,
  followup,
  recordState,
  taskState,
  busy,
  onRecord,
  onTask,
}: MessageExtrasProps) {
  const notice = followup?.status === "ready" ? contradictionNotice(followup.intervention) : null;

  return (
    <>
      <Inspector items={memoryUsed} />
      {notice ? <Notice text={notice.text} question={notice.question} /> : null}
      {followup?.status === "ready" ? (
        <>
          <PendingRecords
            followup={followup}
            recordState={recordState}
            busy={busy}
            onRecord={onRecord}
          />
          <PendingTasks
            followup={followup}
            taskState={taskState}
            busy={busy}
            onTask={onTask}
          />
        </>
      ) : null}
      {trace ? <TraceDisclosure trace={trace} /> : null}
    </>
  );
}
