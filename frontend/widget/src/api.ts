import type { ChatResponse, WidgetConfig } from "./types";

let _apiHost = "http://localhost:8000";

export function setApiHost(host: string): void {
  _apiHost = host.replace(/\/$/, "");
}

export async function fetchWidgetConfig(widgetId: string): Promise<WidgetConfig> {
  const resp = await fetch(`${_apiHost}/widgets/${widgetId}/config`);
  if (!resp.ok) throw new Error(`Config fetch failed: ${resp.status}`);
  return resp.json() as Promise<WidgetConfig>;
}

export async function sendChat(
  message: string,
  conversationId: string | null,
  widgetId: string,
  token: string | null,
): Promise<ChatResponse> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (token) headers["Authorization"] = `Bearer ${token}`;

  const body: Record<string, string> = { message, widget_id: widgetId };
  if (conversationId) body["conversation_id"] = conversationId;

  const resp = await fetch(`${_apiHost}/chat`, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    // Surface 401 specifically so the UI can prompt the user to sign in.
    const err = new Error(`Chat request failed: ${resp.status}`);
    (err as Error & { status?: number }).status = resp.status;
    throw err;
  }
  return resp.json() as Promise<ChatResponse>;
}

export async function loginUser(
  email: string,
  password: string,
): Promise<{ access_token: string }> {
  const form = new URLSearchParams({ username: email, password });
  const resp = await fetch(`${_apiHost}/auth/login`, { method: "POST", body: form });
  if (!resp.ok) {
    const body = await resp.text();
    throw new Error(`Login failed (${resp.status}): ${body.slice(0, 200)}`);
  }
  return resp.json() as Promise<{ access_token: string }>;
}
