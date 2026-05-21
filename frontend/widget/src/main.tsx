import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";

declare global {
  interface Window {
    __WIDGET_CONFIG__?: { widgetId: string; apiOrigin: string };
  }
}

// Read config from either the server-injected global or URL params (dev mode).
const cfg = window.__WIDGET_CONFIG__;
const params = new URLSearchParams(window.location.search);
const widgetId = cfg?.widgetId ?? params.get("widget_id") ?? "";
const apiHost = cfg?.apiOrigin ?? params.get("api_host") ?? window.location.origin;

const container = document.getElementById("root");
if (container && widgetId) {
  createRoot(container).render(
    <StrictMode>
      <App widgetId={widgetId} apiHost={apiHost} />
    </StrictMode>,
  );
}
