"use client";

import Link from "next/link";
import { useState, type FormEvent } from "react";
import { useRouter } from "next/navigation";

import { ApiError, signup } from "@/lib/api";

const BANNER = "Demo only: please don't send real personal data.";

const USERNAME_PATTERN = /^[A-Za-z0-9_]{3,100}$/;
const VALIDATION_HINT =
  "Username: 3-100 characters, letters, numbers or underscore. Password: 12-128 characters.";

/** Local pre-check mirroring the server's rules. Returns a hint, or null when the input is fine. */
function localError(username: string, password: string, inviteCode: string): string | null {
  if (!USERNAME_PATTERN.test(username)) {
    return VALIDATION_HINT;
  }
  if (password.length < 12 || password.length > 128) {
    return VALIDATION_HINT;
  }
  if (inviteCode.length > 64) {
    return "Invite code must be 64 characters or fewer.";
  }
  return null;
}

export default function SignupPage() {
  const router = useRouter();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [inviteCode, setInviteCode] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (submitting) {
      return;
    }
    setError(null);

    const invalid = localError(username, password, inviteCode);
    if (invalid) {
      setError(invalid);
      return;
    }

    setSubmitting(true);
    try {
      await signup(username, password, inviteCode);
      router.replace("/");
    } catch (err) {
      // A wrong, empty or unset invite code is the same 403, and the user cap is also a 403:
      // one message for all of them, so nothing reveals which one it was.
      if (err instanceof ApiError && err.status === 403) {
        setError("That invite code isn't valid.");
      } else if (err instanceof ApiError && err.status === 409) {
        setError("That username is taken.");
      } else if (err instanceof ApiError && err.status === 429) {
        setError("Too many attempts. Try again in 15 minutes.");
      } else if (err instanceof ApiError && err.status === 422) {
        setError(VALIDATION_HINT);
      } else {
        setError("Something went wrong. Try again.");
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

        <h1 className="mt-4 text-2xl font-bold tracking-tight text-white">Create account</h1>

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
              minLength={3}
              maxLength={100}
              pattern="[A-Za-z0-9_]+"
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
              autoComplete="new-password"
              required
              minLength={12}
              maxLength={128}
              className="rounded-md border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100 outline-none placeholder:text-slate-600 focus:border-emerald-700"
            />
          </label>

          <label className="flex flex-col gap-1 text-xs font-medium text-slate-400">
            Invite code
            <input
              type="text"
              name="invite_code"
              value={inviteCode}
              onChange={(event) => setInviteCode(event.target.value)}
              autoComplete="off"
              maxLength={64}
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
            {submitting ? "Creating account…" : "Create account"}
          </button>
        </form>

        <p className="mt-4 text-center text-xs text-slate-400">
          Already have an account?{" "}
          <Link href="/login" className="font-medium text-emerald-400 hover:underline">
            Sign in
          </Link>
        </p>
      </div>
    </div>
  );
}
