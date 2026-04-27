export interface DataSummary {
  shape: number[];
  columns: string[];
  dtypes: Record<string, string>;
  numeric_cols: string[];
  categorical_cols: string[];
  datetime_cols: string[];
  missing_pct: Record<string, number>;
  memory_mb: number;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  figures?: string[];
}

export interface ToolResult {
  success: boolean;
  text: string;
  figure_json?: string | null;
}

export interface ApiConfig {
  api_key: string;
  base_url: string;
  model: string;
  llm_provider: 'embedded' | 'ollama' | 'remote';
}

export interface LlmStatus {
  provider: string;
  model: string;
  base_url: string;
  available: boolean;
  message: string;
}
