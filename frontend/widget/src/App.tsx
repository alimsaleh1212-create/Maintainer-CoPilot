import { useEffect, useRef, useState } from "react";
import { fetchWidgetConfig, loginUser, setApiHost } from "./api";
import { Chat } from "./Chat";
import type { WidgetConfig } from "./types";

interface AppProps {
  widgetId: string;
  apiHost: string;
}

function LoginForm({
  primary,
  onLogin,
  onCancel,
}: {
  primary: string;
  onLogin: (token: string) => void;
  onCancel: () => void;
}) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      const data = await loginUser(email, password);
      onLogin(data.access_token);
    } catch {
      setError("Invalid email or password");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div
      style={{
        height: "100%",
        background: "#0f172a",
        display: "flex",
        flexDirection: "column",
        fontFamily: "'Inter', system-ui, sans-serif",
      }}
    >
      {/* Header */}
      <div style={{ padding: "14px 16px", background: "#1e293b", borderBottom: "1px solid #334155" }}>
        <div style={{ fontSize: 13.5, fontWeight: 600, color: "#f1f5f9" }}>Sign in to chat</div>
      </div>

      <form onSubmit={(e) => void handleSubmit(e)} style={{ padding: 20, display: "flex", flexDirection: "column", gap: 14, flex: 1 }}>
        <div style={{ textAlign: "center", marginBottom: 8 }}>
          <div style={{
            width: 48, height: 48, borderRadius: 14, margin: "0 auto 12px",
            background: `linear-gradient(135deg, ${primary}, ${primary}bb)`,
            display: "flex", alignItems: "center", justifyContent: "center",
          }}>
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" /><circle cx="12" cy="7" r="4" />
            </svg>
          </div>
          <p style={{ fontSize: 12.5, color: "#94a3b8", margin: 0 }}>
            Sign in to enable memory features and personalized responses
          </p>
        </div>

        {error && (
          <div style={{ background: "#450a0a", border: "1px solid #7f1d1d", borderRadius: 8, padding: "8px 12px", fontSize: 12.5, color: "#fca5a5" }}>
            {error}
          </div>
        )}

        {[
          { label: "Email", type: "email", value: email, onChange: setEmail },
          { label: "Password", type: "password", value: password, onChange: setPassword },
        ].map(({ label, type, value, onChange }) => (
          <div key={label} style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            <label style={{ fontSize: 12, fontWeight: 500, color: "#94a3b8", textTransform: "uppercase", letterSpacing: "0.05em" }}>
              {label}
            </label>
            <input
              type={type}
              value={value}
              onChange={(e) => onChange(e.target.value)}
              required
              style={{
                background: "#1e293b",
                border: "1px solid #334155",
                borderRadius: 10,
                padding: "10px 14px",
                color: "#f1f5f9",
                fontSize: 13.5,
                outline: "none",
                fontFamily: "inherit",
                transition: "border-color 0.2s",
              }}
            />
          </div>
        ))}

        <button
          type="submit"
          disabled={loading}
          style={{
            background: `linear-gradient(135deg, ${primary}, ${primary}cc)`,
            border: "none",
            borderRadius: 10,
            padding: "11px",
            color: "#fff",
            fontSize: 13.5,
            fontWeight: 600,
            cursor: loading ? "not-allowed" : "pointer",
            opacity: loading ? 0.7 : 1,
            fontFamily: "inherit",
            transition: "opacity 0.2s",
            marginTop: 4,
          }}
        >
          {loading ? "Signing in…" : "Sign in"}
        </button>

        <button
          type="button"
          onClick={onCancel}
          style={{
            background: "transparent",
            border: "1px solid #334155",
            borderRadius: 10,
            padding: "10px",
            color: "#94a3b8",
            fontSize: 13,
            cursor: "pointer",
            fontFamily: "inherit",
          }}
        >
          Continue as guest
        </button>
      </form>
    </div>
  );
}

export function App({ widgetId, apiHost }: AppProps) {
  const [open, setOpen] = useState(false);
  const [config, setConfig] = useState<WidgetConfig | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [showLogin, setShowLogin] = useState(false);
  const [unread, setUnread] = useState(0);
  const panelRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setApiHost(apiHost);
    fetchWidgetConfig(widgetId)
      .then(setConfig)
      .catch((e: unknown) => setError(String(e)));
  }, [widgetId, apiHost]);

  useEffect(() => {
    if (open) setUnread(0);
  }, [open]);

  if (error || !config) return null;

  const primary = config.theme?.primaryColor ?? "#22c55e";
  const position = config.theme?.position ?? "bottom-right";
  const posStyle = position === "bottom-left"
    ? { left: 20, right: "auto" as const }
    : { right: 20, left: "auto" as const };

  return (
    <>
      <style>{`
        @keyframes panelIn {
          from { opacity: 0; transform: translateY(12px) scale(0.97); }
          to { opacity: 1; transform: translateY(0) scale(1); }
        }
        @keyframes bubblePulse {
          0%, 100% { box-shadow: 0 0 0 0 ${primary}40; }
          50% { box-shadow: 0 0 0 8px transparent; }
        }
        * { box-sizing: border-box; }
      `}</style>

      <div style={{ position: "fixed", bottom: 20, zIndex: 2147483647, fontFamily: "'Inter', system-ui, sans-serif", ...posStyle }}>
        {/* Chat panel */}
        {open && (
          <div
            ref={panelRef}
            style={{
              width: 368,
              height: 540,
              background: "#0f172a",
              borderRadius: 20,
              boxShadow: "0 24px 64px rgba(0,0,0,0.5), 0 0 0 1px #1e293b",
              display: "flex",
              flexDirection: "column",
              overflow: "hidden",
              marginBottom: 12,
              animation: "panelIn 0.22s cubic-bezier(0.16, 1, 0.3, 1) forwards",
            }}
          >
            {showLogin ? (
              <LoginForm
                primary={primary}
                onLogin={(t) => { setToken(t); setShowLogin(false); }}
                onCancel={() => setShowLogin(false)}
              />
            ) : (
              <>
                {!token && (
                  <div style={{
                    padding: "8px 14px",
                    background: "linear-gradient(135deg, #1e3a2f, #0f2419)",
                    borderBottom: "1px solid #166534",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    fontSize: 12,
                    color: "#86efac",
                  }}>
                    <span>Sign in for memory &amp; personalization</span>
                    <button
                      onClick={() => setShowLogin(true)}
                      style={{
                        background: "transparent",
                        border: "1px solid #22c55e",
                        borderRadius: 6,
                        padding: "3px 10px",
                        color: "#22c55e",
                        fontSize: 11,
                        cursor: "pointer",
                        fontFamily: "inherit",
                        fontWeight: 500,
                      }}
                    >
                      Sign in
                    </button>
                  </div>
                )}
                <Chat config={config} token={token} />
              </>
            )}
          </div>
        )}

        {/* Bubble */}
        <button
          onClick={() => setOpen((o) => !o)}
          aria-label={open ? "Close chat" : "Open Maintainer's Copilot chat"}
          style={{
            width: 52,
            height: 52,
            borderRadius: "50%",
            background: open
              ? "#1e293b"
              : `linear-gradient(135deg, ${primary}, ${primary}cc)`,
            border: open ? "1px solid #334155" : "none",
            boxShadow: open
              ? "0 4px 16px rgba(0,0,0,0.3)"
              : `0 4px 20px ${primary}60`,
            cursor: "pointer",
            color: "#fff",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            marginLeft: "auto",
            transition: "all 0.2s cubic-bezier(0.16, 1, 0.3, 1)",
            position: "relative",
            animation: !open && unread > 0 ? "bubblePulse 2s infinite" : "none",
          }}
        >
          {open ? (
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          ) : (
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
            </svg>
          )}
          {/* Unread badge */}
          {!open && unread > 0 && (
            <span style={{
              position: "absolute",
              top: -2,
              right: -2,
              width: 18,
              height: 18,
              borderRadius: "50%",
              background: "#ef4444",
              color: "#fff",
              fontSize: 10,
              fontWeight: 700,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              border: "2px solid #0f172a",
            }}>
              {unread}
            </span>
          )}
        </button>
      </div>
    </>
  );
}
