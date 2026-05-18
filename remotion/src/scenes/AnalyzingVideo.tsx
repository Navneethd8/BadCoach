import { interpolate, useCurrentFrame } from "remotion";
import { AppChrome } from "../components/AppChrome";
import { Card } from "../components/Card";
import { Icon } from "../components/Icon";
import { PageTitle } from "../components/PageTitle";
import { PhoneViewport } from "../components/PhoneViewport";
import { SceneShell } from "../components/SceneShell";
import { ScrollableContent } from "../components/ScrollableContent";
import { UploadClipCard } from "../components/UploadClipCard";
import { ANALYZING_TIMING } from "../sceneTiming";
import { BRAND, MONO } from "../theme";

const LOADING_STEPS = [
  { icon: "movie_filter", label: "Splitting clip into frames" },
  { icon: "directions_run", label: "Tracing poses" },
  { icon: "query_stats", label: "Analyzing strokes" },
  { icon: "rate_review", label: "Generating feedback" },
];

export const AnalyzingVideo = () => {
  const frame = useCurrentFrame();
  const { stepStarts, allStepsDoneAt, scrollFrom, scrollTo, scrollDistance } = ANALYZING_TIMING;

  const activeStep = stepStarts.reduce((acc, start, idx) => (frame >= start ? idx : acc), 0);
  const allDone = frame >= allStepsDoneAt;

  return (
    <SceneShell>
      <PhoneViewport>
        <AppChrome active="analyze">
          <ScrollableContent
            scrollFromFrame={scrollFrom}
            scrollToFrame={scrollTo}
            scrollDistance={scrollDistance}
          >
            <PageTitle />
            <UploadClipCard
              footer={
                <button
                  type="button"
                  style={{
                    marginTop: 16,
                    width: "100%",
                    minHeight: 44,
                    border: "none",
                    borderRadius: 999,
                    backgroundColor: "rgba(108,156,141,0.45)",
                    color: "#fafafa",
                    fontFamily: MONO,
                    fontSize: 14,
                    fontWeight: 700,
                    letterSpacing: "0.04em",
                    textTransform: "lowercase",
                  }}
                >
                  <span style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
                    <span
                      style={{
                        width: 14,
                        height: 14,
                        border: "2px solid rgba(255,255,255,0.35)",
                        borderTopColor: "#fff",
                        borderRadius: "50%",
                        display: "inline-block",
                        transform: `rotate(${frame * 12}deg)`,
                      }}
                    />
                    {allDone ? "Done" : "Analyzing..."}
                  </span>
                </button>
              }
            />

            <Card title="analysis results" icon="analytics">
              <div style={{ padding: "12px 0" }}>
                {LOADING_STEPS.map((step, idx) => {
                  const isDone = allDone || idx < activeStep;
                  const isActive = !allDone && idx === activeStep;
                  const pulse = isActive ? 0.5 + 0.5 * Math.sin(frame / 6) : 1;
                  return (
                    <div
                      key={step.label}
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: 12,
                        padding: "10px 12px",
                        marginBottom: 6,
                        borderRadius: 6,
                        backgroundColor: isActive ? "#f5f5f5" : "transparent",
                        opacity: interpolate(frame, [idx * 6, idx * 6 + 8], [0.4, 1], {
                          extrapolateLeft: "clamp",
                          extrapolateRight: "clamp",
                        }),
                      }}
                    >
                      <div
                        style={{
                          width: 24,
                          height: 24,
                          borderRadius: "50%",
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                          backgroundColor: isDone ? BRAND : isActive ? "#e5e5e5" : "#f0f0f0",
                        }}
                      >
                        {isDone ? (
                          <Icon name="check" size={14} style={{ color: "#fff" }} />
                        ) : isActive ? (
                          <span
                            style={{
                              width: 8,
                              height: 8,
                              borderRadius: "50%",
                              backgroundColor: "#34d399",
                              opacity: pulse,
                            }}
                          />
                        ) : (
                          <span style={{ width: 6, height: 6, borderRadius: "50%", backgroundColor: "#d4d4d4" }} />
                        )}
                      </div>
                      <Icon
                        name={step.icon}
                        size={18}
                        style={{ color: isDone ? BRAND : isActive ? "#404040" : "#a3a3a3" }}
                      />
                      <span
                        style={{
                          fontSize: 14,
                          color: isDone ? "#a3a3a3" : isActive ? "#262626" : "#a3a3a3",
                          fontWeight: isActive ? 600 : 400,
                        }}
                      >
                        {step.label}
                        {isActive ? "..." : ""}
                      </span>
                    </div>
                  );
                })}
              </div>
            </Card>
          </ScrollableContent>
        </AppChrome>
      </PhoneViewport>
    </SceneShell>
  );
};
