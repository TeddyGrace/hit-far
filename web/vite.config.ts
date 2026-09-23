import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In dev, the API runs on :8000 and Vite proxies /api to it (same-origin cookies).
export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": "http://localhost:8000" } },
});
