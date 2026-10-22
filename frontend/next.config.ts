import type { NextConfig } from "next";

// Server-side only. Never prefix this with NEXT_PUBLIC_: the browser must never know the
// backend origin, and it must never appear in the client bundle. Browser code only calls
// same-origin "/api/..." paths; this rewrite forwards them to the backend.
const API_BASE_URL = process.env.API_BASE_URL ?? "http://localhost:8000";

const nextConfig: NextConfig = {
  cacheComponents: true,
  partialPrefetching: true,
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${API_BASE_URL}/:path*`,
      },
    ];
  },
};

export default nextConfig;
