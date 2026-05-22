import { interpolate, useCurrentFrame } from "remotion";
import { Icon } from "./Icon";
import { RESULTS_TIMING } from "../sceneTiming";
import { BRAND, MONO, qualityBarColor, qualityTextColor } from "../theme";
import type { DemoResult } from "../types";

const TACTICAL = [
  { key: "technique", color: "#2563eb", border: "rgba(37,99,235,0.3)", bg: "rgba(37,99,235,0.08)", icon: "pan_tool_alt" },
  { key: "placement", color: "#7c3aed", border: "rgba(124,58,237,0.3)", bg: "rgba(124,58,237,0.08)", icon: "explore" },
  { key: "position", color: "#e11d48", border: "rgba(225,29,72,0.3)", bg: "rgba(225,29,72,0.08)", icon: "location_on" },
  { key: "intent", color: "#d97706", border: "rgba(217,119,6,0.3)", bg: "rgba(217,119,6,0.08)", icon: "psychology" },
] as const;

const panel: React.CSSProperties = {
  padding: 16,
  borderRadius: 8,
  backgroundColor: "#fff",
  border: "1px solid #e5e5e5",
};

export const AnalysisResultsBody = ({
  result,
  opacity = 1,
}: {
  result: DemoResult;
  opacity?: number;
}) => {
  const frame = useCurrentFrame();
  const { coachTipBase, coachTipStagger, timelineBase, timelineStagger } = RESULTS_TIMING;

  return (
    <div style={{ opacity }}>
      <div style={{ ...panel, marginBottom: 14 }}>
        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 12 }}>
          <div>
            <span style={{ fontSize: 11, color: "#737373", display: "block", marginBottom: 4 }}>Execution Quality</span>
            <div style={{ fontSize: 22, fontWeight: 700, color: qualityTextColor(result.quality) }}>
              {result.quality}
            </div>
          </div>
          <div style={{ textAlign: "right" }}>
            <span style={{ fontSize: 11, color: "#737373", display: "block", marginBottom: 4, fontFamily: MONO }}>Score</span>
            <div style={{ fontSize: 18, fontWeight: 600, color: "#0a0a0a" }}>
              {result.quality_numeric.toFixed(1)} / 10
            </div>
          </div>
        </div>
        <div style={{ width: "100%", height: 6, backgroundColor: "#e5e5e5", borderRadius: 999, overflow: "hidden" }}>
          <div
            style={{
              height: "100%",
              width: `${(result.quality_numeric / 10) * 100}%`,
              backgroundColor: qualityBarColor(result.quality),
              borderRadius: 999,
            }}
          />
        </div>

        {result.tactical_analysis && (
          <div style={{ marginTop: 16, paddingTop: 14, borderTop: "1px solid #e5e5e5" }}>
            <span style={{ fontSize: 11, color: "#737373", display: "block", marginBottom: 10 }}>Tactical Metrics</span>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
              {TACTICAL.map(({ key, color, border, bg, icon }) => {
                const m = result.tactical_analysis?.[key as keyof typeof result.tactical_analysis];
                if (!m?.label) return null;
                return (
                  <span
                    key={key}
                    style={{
                      display: "inline-flex",
                      alignItems: "center",
                      gap: 6,
                      padding: "4px 8px",
                      borderRadius: 4,
                      fontSize: 10,
                      fontWeight: 600,
                      color,
                      border: `1px solid ${border}`,
                      backgroundColor: bg,
                    }}
                  >
                    <Icon name={icon} size={12} />
                    {m.label.replace(/_/g, " ")}
                  </span>
                );
              })}
            </div>
          </div>
        )}
      </div>

      {result.recommendations && result.recommendations.length > 0 && (
        <div
          style={{
            padding: 14,
            marginBottom: 14,
            borderRadius: 8,
            backgroundColor: "rgba(108,156,141,0.08)",
            border: "1px solid rgba(108,156,141,0.25)",
          }}
        >
          <span
            style={{
              fontSize: 11,
              color: BRAND,
              fontWeight: 600,
              display: "flex",
              alignItems: "center",
              gap: 6,
              marginBottom: 10,
            }}
          >
            <Icon name="tips_and_updates" size={14} />
            Coach&apos;s Recommendations
          </span>
          <ul style={{ margin: 0, padding: 0, listStyle: "none" }}>
            {result.recommendations.map((tip, idx) => (
              <li
                key={tip}
                style={{
                  fontSize: 13,
                  color: "#404040",
                  marginBottom: 8,
                  display: "flex",
                  gap: 8,
                  opacity: interpolate(frame, [coachTipBase + idx * coachTipStagger, coachTipBase + idx * coachTipStagger + 10], [0, 1], {
                    extrapolateLeft: "clamp",
                    extrapolateRight: "clamp",
                  }),
                }}
              >
                <span style={{ color: BRAND }}>•</span>
                {tip}
              </li>
            ))}
          </ul>
        </div>
      )}

      {result.timeline && result.timeline.length > 0 && (
        <div style={panel}>
          <span style={{ fontSize: 11, color: "#737373", display: "flex", gap: 6, marginBottom: 12 }}>
            <Icon name="timeline" size={14} />
            Play-by-Play Breakdown
          </span>
          <div style={{ borderLeft: "2px solid #e5e5e5", marginLeft: 8, paddingLeft: 16 }}>
            {result.timeline.map((event, idx) => (
              <div
                key={`${event.timestamp}-${event.label}`}
                style={{
                  marginBottom: 20,
                  opacity: interpolate(frame, [timelineBase + idx * timelineStagger, timelineBase + idx * timelineStagger + 10], [0, 1], {
                    extrapolateLeft: "clamp",
                    extrapolateRight: "clamp",
                  }),
                }}
              >
                <span style={{ fontSize: 11, fontFamily: MONO, color: "#737373" }}>{event.timestamp}</span>
                <div style={{ fontSize: 16, fontWeight: 600, color: "#262626" }}>
                  {event.label.replace(/_/g, " ")}
                  <span style={{ fontSize: 10, color: "#a3a3a3", marginLeft: 8 }}>
                    {(event.confidence * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
