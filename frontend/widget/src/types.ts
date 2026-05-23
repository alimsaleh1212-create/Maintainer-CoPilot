export interface WidgetConfig {
  public_widget_id: string;
  greeting: string;
  theme: {
    primaryColor?: string;
    position?: "bottom-right" | "bottom-left";
  };
  enabled_tools: string[];
  enabled?: boolean;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: number;
  toolsUsed?: string[];
}

export interface ChatResponse {
  response: string;
  conversation_id: string;
  tools_used?: string[];
  citations?: unknown[];
}
