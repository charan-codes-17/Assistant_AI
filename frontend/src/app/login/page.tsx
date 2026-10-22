"use client";

import { useState, type FormEvent } from "react";
import { useRouter } from "next/navigation";

import { ApiError, AuthError, login } from "@/lib/api";

const BANNER = "Demo only: please don't send real personal data.";

export default function LoginPage() {
  const router = useRouter();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (submitting) {
      return;
    }
    setSubmitting(true);
    setError(null);
    try {
      await login(username, password);
      router.replace("/");
    } catch (err) {
      // One generic message for every credential failure: never reveal which part was wrong.
      if (err instanceof AuthError) {
        setError("Wrong username or password");
      } else if (err instanceof ApiError && err.status === 429) {
        setError("Too many attempts, try later");
      } else if (err instanceof ApiError && err.status === 503) {
        setError("Login unavailable");
      } else {
        setError("Something went wrong. Please try again.");
      }
      setSubmitting(false);
    }
  }

  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-slate-950 p-6 text-slate-100">
      <div className="w-full max-w-md rounded-xl border border-slate-800 bg-slate-900 p-6 shadow-2xl">
        <div className="flex items-center gap-3">
          <div className="h-3 w-3 rounded-full bg-emerald-500" />
          <span className="text-xs font-semibold uppercase tracking-wider text-emerald-400">
            Personal Memory Assistant
          </span>
        </div>

        <h1 className="mt-4 text-2xl font-bold tracking-tight text-white">Sign in</h1>

        <p className="mt-3 rounded-lg border border-amber-800/60 bg-amber-950/40 p-3 text-xs text-amber-300">
          {BANNER}
        </p>

        <form onSubmit={handleSubmit} className="mt-5 flex flex-col gap-4">
          <label className="flex flex-col gap-1 text-xs font-medium text-slate-400">
            Username
            <input
              type="text"
              name="username"
              value={username}
              onChange={(event) => setUsername(event.target.value)}
              autoComplete="username"
              required
              maxLength={100}
              className="rounded-md border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100 outline-none placeholder:text-slate-600 focus:border-emerald-700"
            />
          </label>

          <label className="flex flex-col gap-1 text-xs font-medium text-slate-400">
            Password
            <input
              type="password"
              name="password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              autoComplete="current-password"
              required
              maxLength={128}
              className="rounded-md border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100 outline-none placeholder:text-slate-600 focus:border-emerald-700"
            />
          </label>

          {error ? (
            <p role="alert" className="text-xs text-rose-400">
              {error}
            </p>
          ) : null}

          <button
            type="submit"
            disabled={submitting}
            className="rounded-md border border-emerald-700 bg-emerald-900/60 px-4 py-2 text-sm font-semibold text-emerald-100 hover:bg-emerald-900 disabled:opacity-50"
          >
            {submitting ? "Signing in…" : "Sign in"}
          </button>
        </form>
      </div>
    </div>
  );
}
