import { defineConfig } from "vite";

export default defineConfig({
  server: { port: 5173 },
  // load .wgsl files as raw strings
  assetsInclude: ["**/*.wgsl"],
});
