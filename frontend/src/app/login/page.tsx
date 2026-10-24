"use client";

import Link from "next/link";
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
    <div className="flex min-h-screen flex-col items-center justify-center bg-black p-6 text-neutral-100">
      <div className="w-full max-w-md rounded-2xl border border-neutral-800 bg-neutral-950 p-6 shadow-2xl">
        <div className="flex items-center gap-3">
          <div className="h-3 w-3 rounded-full bg-teal-500" />
          <span className="text-xs font-semibold uppercase tracking-wider text-teal-400">
            Personal Memory Assistant
          </span>
        </div>

        <h1 className="mt-4 text-2xl font-bold tracking-tight text-neutral-100">Sign in</h1>

        <form onSubmit={handleSubmit} className="mt-5 flex flex-col gap-4">
          <label className="flex flex-col gap-1 text-xs font-medium text-neutral-400">
            Username
            <input
              type="text"
              name="username"
              value={username}
              onChange={(event) => setUsername(event.target.value)}
              autoComplete="username"
              required
              maxLength={100}
              className="rounded-lg border border-neutral-800 bg-neutral-900 px-3 py-2 text-sm text-neutral-100 outline-none placeholder:text-neutral-500 focus:border-neutral-600"
            />
          </label>

          <label className="flex flex-col gap-1 text-xs font-medium text-neutral-400">
            Password
            <input
              type="password"
              name="password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              autoComplete="current-password"
              required
              maxLength={128}
              className="rounded-lg border border-neutral-800 bg-neutral-900 px-3 py-2 text-sm text-neutral-100 outline-none placeholder:text-neutral-500 focus:border-neutral-600"
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
            className="rounded-lg bg-teal-500 px-4 py-2 text-sm font-semibold text-black transition-[background-color,border-color,transform] duration-[var(--dur-hover)] ease-[var(--ease-out)] hover:bg-teal-400 active:duration-[var(--dur-press)] active:scale-[0.98] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 focus-visible:ring-offset-1 focus-visible:ring-offset-black disabled:opacity-50"
          >
            {submitting ? "Signing in…" : "Sign in"}
          </button>
        </form>

        <p className="mt-4 text-center text-xs text-neutral-400">
          Need an account?{" "}
          <Link href="/signup" className="font-medium text-teal-400 hover:text-teal-300">
            Create one
          </Link>
        </p>

        <p className="mt-4 text-center text-xs text-neutral-500">{BANNER}</p>
      </div>
    </div>
  );
}
