import { spring, useCurrentFrame, useVideoConfig } from "remotion";
import demoResult from "../data/demo-result.json";
import { AnalysisResultsBody } from "../components/AnalysisResultsBody";
import { AppChrome } from "../components/AppChrome";
import { Card } from "../components/Card";
import { PageTitle } from "../components/PageTitle";
import { PhoneViewport } from "../components/PhoneViewport";
import { SceneShell } from "../components/SceneShell";
import { ScrollableContent } from "../components/ScrollableContent";
import { UploadClipCard } from "../components/UploadClipCard";
import { RESULTS_TIMING } from "../sceneTiming";
import { BRAND, MONO } from "../theme";
import type { DemoResult } from "../types";

const result = demoResult as DemoResult;

export const ResultsVideo = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const { revealAt, scrollFrom, scrollTo, scrollDistance } = RESULTS_TIMING;
  const reveal = spring({ frame: frame - revealAt, fps, config: { damping: 20 } });

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
                    backgroundColor: BRAND,
                    color: "#fafafa",
                    fontFamily: MONO,
                    fontSize: 14,
                    fontWeight: 700,
                    letterSpacing: "0.04em",
                    textTransform: "lowercase",
                    boxShadow: "0 2px 12px rgba(0,0,0,0.12)",
                  }}
                >
                  analyze stroke
                </button>
              }
            />

            <Card title="analysis results" icon="analytics">
              <AnalysisResultsBody result={result} opacity={reveal} />
            </Card>
          </ScrollableContent>
        </AppChrome>
      </PhoneViewport>
    </SceneShell>
  );
};
