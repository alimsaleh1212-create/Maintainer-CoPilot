import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "dist",
    rollupOptions: {
      input: "src/main.tsx",
      output: {
        entryFileNames: "widget.js",
        chunkFileNames: "widget-chunk-[hash].js",
        assetFileNames: "widget.[ext]",
        // Inline small assets to keep single-file output
        inlineDynamicImports: false,
      },
    },
  },
});
