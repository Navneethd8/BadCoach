import type { CSSProperties } from "react";

export const BRAND = "#6c9c8d";
export const MONO = "'Iosevka Charon Mono', monospace";
export const SANS = "'Inter', system-ui, sans-serif";

export const CLIP_MAX_HEIGHT = 260;
export const DROPZONE_MIN_HEIGHT = 220;

/** Dashed frame around the clip — must match across upload / analyzing / results. */
export const dropzoneShellStyle = (
  borderColor = "#c4c4c4",
  backgroundColor: string | "transparent" = "#000",
): CSSProperties => ({
  border: `1px dashed ${borderColor}`,
  borderRadius: 8,
  minHeight: DROPZONE_MIN_HEIGHT,
  overflow: "hidden",
  backgroundColor,
});

export const qualityTextColor = (quality: string) => {
  const q = quality.toLowerCase();
  if (q.includes("elite") || q.includes("expert") || q.includes("advanced")) return BRAND;
  if (q.includes("proficient") || q.includes("good")) return "#0d9488";
  if (q.includes("competent")) return "#d97706";
  if (q.includes("developing") || q.includes("emerging")) return "#ea580c";
  return "#e11d48";
};

export const qualityBarColor = (quality: string) => {
  const q = quality.toLowerCase();
  if (q.includes("elite") || q.includes("expert")) return BRAND;
  if (q.includes("advanced")) return "#5a8578";
  if (q.includes("proficient")) return "#06b6d4";
  if (q.includes("competent")) return "#f59e0b";
  if (q.includes("developing") || q.includes("emerging")) return "#f97316";
  return "#f43f5e";
};
