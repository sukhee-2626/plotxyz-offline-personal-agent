export type FieldType = "text" | "textarea" | "number" | "checkbox";

export interface AgentField {
  name: string;
  label: string;
  type: FieldType;
  rows?: number;
  placeholder?: string;
}

export interface AgentConfig {
  slug: string;
  title: string;
  tagline: string;
  purpose: string;
  accent: string;
  features: string[];
  limitations: string[];
  fields: AgentField[];
  sample: Record<string, string | number | boolean>;
  repo: string;
}

export const agent: AgentConfig = {
  "slug": "plotxyz-offline-personal-agent",
  "title": "Offline Personal",
  "tagline": "A private assistant that still works when the network does not.",
  "purpose": "Provides a chat interface with deterministic local behaviors (safe arithmetic, extractive summaries, note-to-plan) and optional inference through a configured OpenAI-compatible or Ollama endpoint.",
  "accent": "#38bdf8",
  "features": [
    "Deterministic offline fallback",
    "Optional local-model inference",
    "Safe arithmetic and extractive summaries"
  ],
  "limitations": [
    "The fallback is rule-based and cannot answer open-domain questions.",
    "Model inference requires a separately configured endpoint."
  ],
  "fields": [
    {
      "name": "message",
      "label": "Message",
      "type": "textarea",
      "rows": 4,
      "placeholder": ""
    },
    {
      "name": "context",
      "label": "Context (optional)",
      "type": "textarea",
      "rows": 5,
      "placeholder": ""
    },
    {
      "name": "use_provider",
      "label": "Use configured model provider",
      "type": "checkbox",
      "rows": 4,
      "placeholder": ""
    }
  ],
  "sample": {
    "message": "Summarize my project notes.",
    "context": "The launch is scheduled for June 3. The team agreed to cut scope. Marketing will prepare the announcement.",
    "use_provider": false
  },
  "repo": "https://github.com/sukhee-2626/plotxyz-offline-personal-agent"
};
