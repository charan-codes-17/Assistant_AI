export default function Home() {
  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col items-center justify-center p-6">
      <main className="max-w-2xl w-full bg-slate-900 border border-slate-800 rounded-xl p-8 shadow-2xl flex flex-col gap-6">
        <div className="flex items-center gap-3">
          <div className="h-3 w-3 rounded-full bg-emerald-500 animate-pulse" />
          <span className="text-xs uppercase tracking-wider font-semibold text-emerald-400">
            Phase 0 — Environment Setup
          </span>
        </div>

        <h1 className="text-3xl font-bold tracking-tight text-white">
          Persistent-Memory Personal AI Assistant
        </h1>

        <p className="text-slate-400 text-sm leading-relaxed">
          Nebius x NVIDIA Global AI Hackathon (Personal AI Track). A cross-session
          memory assistant powered by NVIDIA Nemotron models, PostgreSQL + pgvector,
          and FastAPI.
        </p>

        <div className="bg-amber-950/40 border border-amber-800/60 rounded-lg p-4 text-xs text-amber-300">
          <strong>Notice:</strong> Demo system only. Do not submit or store real personal data.
        </div>

        <div className="flex flex-wrap items-center gap-4 pt-4 border-t border-slate-800 text-xs text-slate-500">
          <span>Backend API: <code>http://localhost:8000</code></span>
          <span>•</span>
          <span>Health Check: <code>/health</code></span>
          <span>•</span>
          <span>Next.js 14+ App Router</span>
        </div>
      </main>
    </div>
  );
}
