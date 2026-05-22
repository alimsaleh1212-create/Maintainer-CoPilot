import { useEffect, useRef, useState } from "react";
import { sendChat } from "./api";
import type { Message, WidgetConfig } from "./types";

interface Props {
  config: WidgetConfig;
  token: string | null;
}

function TypingIndicator() {
  return (
    <div style={{ display: "flex", gap: 6, padding: "12px 16px", alignItems: "center" }}>
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          style={{
            width: 7,
            height: 7,
            borderRadius: "50%",
            background: "#94a3b8",
            display: "inline-block",
            animation: `bounce 1.2s ease-in-out ${i * 0.2}s infinite`,
          }}
        />
      ))}
      <style>{`
        @keyframes bounce {
          0%, 80%, 100% { transform: translateY(0); opacity: 0.5; }
          40% { transform: translateY(-6px); opacity: 1; }
        }
      `}</style>
    </div>
  );
}

// Maps the backend's tool schema names to a short label + monochrome SVG.
// Keep this in sync with backend/app/services/chatbot.py::_TOOL_SCHEMAS.
const TOOL_DISPLAY: Record<string, { label: string; icon: JSX.Element }> = {
  rag_search: {
    label: "RAG search",
    icon: (
      <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="11" cy="11" r="7" /><line x1="21" y1="21" x2="16.5" y2="16.5" />
      </svg>
    ),
  },
  classify_issue: {
    label: "Classify",
    icon: (
      <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
        <path d="M20.59 13.41L13.42 20.58a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z" />
        <line x1="7" y1="7" x2="7.01" y2="7" />
      </svg>
    ),
  },
  extract_entities: {
    label: "NER",
    icon: (
      <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
        <polyline points="16 18 22 12 16 6" /><polyline points="8 6 2 12 8 18" />
      </svg>
    ),
  },
  summarize_text: {
    label: "Summarize",
    icon: (
      <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
        <line x1="4" y1="6" x2="20" y2="6" /><line x1="4" y1="12" x2="20" y2="12" /><line x1="4" y1="18" x2="14" y2="18" />
      </svg>
    ),
  },
  write_memory: {
    label: "Memory",
    icon: (
      <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
        <path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z" />
      </svg>
    ),
  },
};

function ToolChips({ tools }: { tools: string[] }) {
  if (!tools.length) return null;
  // Deduplicate while preserving order so a tool called twice shows once.
  const unique = Array.from(new Set(tools));
  return (
    <div
      style={{
        display: "flex",
        flexWrap: "wrap",
        gap: 6,
        marginTop: 8,
        marginLeft: 36, // align under the bot bubble (28px avatar + 8px gap)
        opacity: 0.92,
      }}
    >
      {unique.map((tool) => {
        const meta = TOOL_DISPLAY[tool] ?? { label: tool, icon: null };
        return (
          <span
            key={tool}
            title={`Tool used: ${tool}`}
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: 4,
              padding: "2px 8px",
              borderRadius: 4,
              background: "rgba(34, 197, 94, 0.10)",
              border: "1px solid rgba(34, 197, 94, 0.30)",
              color: "#22c55e",
              fontFamily: "'JetBrains Mono', ui-monospace, SFMono-Regular, monospace",
              fontSize: 10.5,
              fontWeight: 500,
              letterSpacing: "0.01em",
              lineHeight: 1.4,
              whiteSpace: "nowrap",
            }}
          >
            {meta.icon}
            <span>{meta.label}</span>
          </span>
        );
      })}
    </div>
  );
}

function MessageBubble({ msg, primary }: { msg: Message; primary: string }) {
  const isUser = msg.role === "user";
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        animation: "fadeSlide 0.2s ease-out forwards",
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent: isUser ? "flex-end" : "flex-start",
        }}
      >
        {!isUser && (
          <div
            style={{
              width: 28,
              height: 28,
              borderRadius: "50%",
              background: "linear-gradient(135deg, #22c55e, #16a34a)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              flexShrink: 0,
              marginRight: 8,
              marginTop: 2,
            }}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10" /><path d="M12 8v4l3 3" />
            </svg>
          </div>
        )}
        <div
          style={{
            maxWidth: "78%",
            padding: "10px 14px",
            borderRadius: isUser ? "16px 16px 4px 16px" : "16px 16px 16px 4px",
            background: isUser
              ? `linear-gradient(135deg, ${primary}, ${primary}dd)`
              : "#1e293b",
            color: isUser ? "#fff" : "#e2e8f0",
            fontSize: 13.5,
            lineHeight: 1.55,
            whiteSpace: "pre-wrap",
            wordBreak: "break-word",
            boxShadow: isUser
              ? `0 2px 12px ${primary}40`
              : "0 1px 4px rgba(0,0,0,0.3)",
            border: isUser ? "none" : "1px solid #334155",
          }}
        >
          {msg.content}
        </div>
      </div>
      {!isUser && msg.toolsUsed && msg.toolsUsed.length > 0 && (
        <ToolChips tools={msg.toolsUsed} />
      )}
    </div>
  );
}

export function Chat({ config, token }: Props) {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "greeting",
      role: "assistant",
      content: config.greeting || "Hi! I'm the Maintainer's Copilot. Paste an issue title or body and I'll help triage, find docs, or summarize it.",
      timestamp: Date.now(),
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const primary = config.theme?.primaryColor ?? "#22c55e";

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function handleSend() {
    const text = input.trim();
    if (!text || loading) return;

    const userMsg: Message = {
      id: String(Date.now()),
      role: "user",
      content: text,
      timestamp: Date.now(),
    };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {
      const data = await sendChat(text, conversationId, config.id, token);
      if (data.conversation_id) setConversationId(data.conversation_id);
      setMessages((prev) => [
        ...prev,
        {
          id: String(Date.now() + 1),
          role: "assistant",
          content: data.response,
          timestamp: Date.now(),
          toolsUsed: data.tools_used ?? [],
        },
      ]);
    } catch (err) {
      const status = (err as Error & { status?: number }).status;
      const content =
        status === 401
          ? "Please sign in first — chat requires an account so I can keep your conversation memory."
          : `Something went wrong. Please try again.${status ? ` (HTTP ${status})` : ""}`;
      setMessages((prev) => [
        ...prev,
        {
          id: String(Date.now() + 2),
          role: "assistant",
          content,
          timestamp: Date.now(),
        },
      ]);
    } finally {
      setLoading(false);
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void handleSend();
    }
  }

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        height: "100%",
        background: "#0f172a",
        fontFamily: "'Inter', system-ui, -apple-system, sans-serif",
        overflow: "hidden",
      }}
    >
      <style>{`
        @keyframes fadeSlide {
          from { opacity: 0; transform: translateY(6px); }
          to { opacity: 1; transform: translateY(0); }
        }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }
        textarea { resize: none; }
        textarea::placeholder { color: #64748b; }
      `}</style>

      {/* Header */}
      <div
        style={{
          padding: "14px 16px",
          background: "linear-gradient(135deg, #1e293b, #0f172a)",
          borderBottom: "1px solid #1e293b",
          display: "flex",
          alignItems: "center",
          gap: 10,
          flexShrink: 0,
        }}
      >
        <div
          style={{
            width: 32,
            height: 32,
            borderRadius: 10,
            background: `linear-gradient(135deg, ${primary}, ${primary}cc)`,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            flexShrink: 0,
          }}
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="16 18 22 12 16 6" /><polyline points="8 6 2 12 8 18" />
          </svg>
        </div>
        <div>
          <div style={{ fontSize: 13.5, fontWeight: 600, color: "#f1f5f9", letterSpacing: "-0.01em" }}>
            Maintainer's Copilot
          </div>
          <div style={{ fontSize: 11, color: "#22c55e", display: "flex", alignItems: "center", gap: 4, marginTop: 1 }}>
            <span style={{ width: 6, height: 6, borderRadius: "50%", background: "#22c55e", display: "inline-block" }} />
            Online
          </div>
        </div>
      </div>

      {/* Messages */}
      <div
        style={{
          flex: 1,
          overflowY: "auto",
          padding: "16px 14px",
          display: "flex",
          flexDirection: "column",
          gap: 12,
        }}
      >
        {messages.map((m) => (
          <MessageBubble key={m.id} msg={m} primary={primary} />
        ))}
        {loading && (
          <div style={{ display: "flex", alignItems: "flex-start", gap: 8 }}>
            <div
              style={{
                width: 28, height: 28, borderRadius: "50%",
                background: "linear-gradient(135deg, #22c55e, #16a34a)",
                display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0, marginTop: 2,
              }}
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="10" /><path d="M12 8v4l3 3" />
              </svg>
            </div>
            <div style={{ background: "#1e293b", border: "1px solid #334155", borderRadius: "16px 16px 16px 4px", padding: "2px 4px" }}>
              <TypingIndicator />
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div
        style={{
          padding: "10px 14px 12px",
          background: "#0f172a",
          borderTop: "1px solid #1e293b",
          flexShrink: 0,
        }}
      >
        <div
          style={{
            display: "flex",
            gap: 8,
            alignItems: "flex-end",
            background: "#1e293b",
            border: "1px solid #334155",
            borderRadius: 14,
            padding: "8px 8px 8px 14px",
            transition: "border-color 0.2s",
          }}
        >
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Paste an issue or ask a question…"
            disabled={loading}
            rows={1}
            style={{
              flex: 1,
              background: "transparent",
              border: "none",
              outline: "none",
              color: "#f1f5f9",
              fontSize: 13.5,
              lineHeight: 1.5,
              fontFamily: "inherit",
              maxHeight: 100,
              overflowY: "auto",
            }}
            onInput={(e) => {
              const el = e.currentTarget;
              el.style.height = "auto";
              el.style.height = `${Math.min(el.scrollHeight, 100)}px`;
            }}
          />
          <button
            onClick={() => void handleSend()}
            disabled={loading || !input.trim()}
            aria-label="Send message"
            style={{
              width: 32,
              height: 32,
              borderRadius: 10,
              border: "none",
              background: input.trim() && !loading
                ? `linear-gradient(135deg, ${primary}, ${primary}cc)`
                : "#334155",
              color: "#fff",
              cursor: input.trim() && !loading ? "pointer" : "not-allowed",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              flexShrink: 0,
              transition: "all 0.2s",
              opacity: loading ? 0.5 : 1,
            }}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <line x1="22" y1="2" x2="11" y2="13" /><polygon points="22 2 15 22 11 13 2 9 22 2" />
            </svg>
          </button>
        </div>
        <div style={{ textAlign: "center", marginTop: 8, fontSize: 10.5, color: "#475569" }}>
          Powered by Maintainer's Copilot
        </div>
      </div>
    </div>
  );
}
