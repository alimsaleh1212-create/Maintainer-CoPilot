export interface WidgetConfig {
  id: string;
  greeting: string;
  theme: {
    primaryColor?: string;
    position?: "bottom-right" | "bottom-left";
  };
  enabled_tools: string[];
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: number;
}

export interface ChatResponse {
  response: string;
  conversation_id: string;
  tool_calls_made?: Array<{ tool: string; result: string }>;
}
