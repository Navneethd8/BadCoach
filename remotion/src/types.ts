export type DemoResult = {
  action?: string;
  confidence?: number;
  quality: string;
  quality_numeric: number;
  recommendations?: string[];
  tactical_analysis?: {
    technique?: { label: string; confidence?: number };
    placement?: { label: string; confidence?: number };
    position?: { label: string; confidence?: number };
    intent?: { label: string; confidence?: number };
  };
  timeline?: Array<{
    timestamp: string;
    label: string;
    confidence: number;
    metrics?: Record<string, { label?: string } | string>;
  }>;
};
