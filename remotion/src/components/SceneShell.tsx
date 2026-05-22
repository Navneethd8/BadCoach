import { AbsoluteFill } from "remotion";

/** Shared wrapper — fonts load via index.css (Material Symbols + Inter + Iosevka Charon Mono). */
export const SceneShell = ({ children }: { children: React.ReactNode }) => (
  <AbsoluteFill>{children}</AbsoluteFill>
);
