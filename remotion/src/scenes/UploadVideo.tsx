import { interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";
import { AppChrome } from "../components/AppChrome";
import { Card } from "../components/Card";
import { ClipDropzone } from "../components/ClipDropzone";
import { Icon } from "../components/Icon";
import { PageTitle } from "../components/PageTitle";
import { PhoneViewport } from "../components/PhoneViewport";
import { SceneShell } from "../components/SceneShell";
import { UploadClipCard } from "../components/UploadClipCard";
import { UPLOAD_TIMING } from "../sceneTiming";
import { BRAND, MONO } from "../theme";

export const UploadVideo = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const { dragStart, dragEnd, previewAt, cursorHide } = UPLOAD_TIMING;

  const phaseDrag = interpolate(frame, [dragStart, dragEnd], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const previewOpacity = spring({
    frame: frame - previewAt,
    fps,
    config: { damping: 18, stiffness: 120 },
  });

  const showPreview = frame >= previewAt;
  const dropActive = phaseDrag > 0 && !showPreview;
  const cursorY = interpolate(frame, [8, dragEnd + 8], [72, 150], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <SceneShell>
      <PhoneViewport>
        <AppChrome active="analyze">
          <PageTitle />
          <UploadClipCard
            dropzoneBorderColor={dropActive ? BRAND : "#c4c4c4"}
            dropzoneBackground={dropActive ? "rgba(108,156,141,0.06)" : showPreview ? "#000" : "transparent"}
            footer={
              <button
                type="button"
                style={{
                  marginTop: 16,
                  width: "100%",
                  minHeight: 44,
                  border: "none",
                  borderRadius: 999,
                  backgroundColor: showPreview ? BRAND : "rgba(108,156,141,0.45)",
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
          >
            {showPreview ? (
              <div style={{ opacity: previewOpacity }}>
                <ClipDropzone />
              </div>
            ) : (
              <div style={{ textAlign: "center", padding: 28, color: "#737373" }}>
                <Icon name="video_file" size={40} style={{ display: "block", margin: "0 auto 12px", color: "#a3a3a3" }} />
                <p style={{ margin: 0, fontSize: 14, fontWeight: 500, color: "#404040" }}>Drag & drop video here</p>
                <p style={{ margin: "6px 0 0", fontSize: 12, color: "#737373" }}>or click to select file</p>
                {dropActive && (
                  <p style={{ margin: "14px 0 0", fontSize: 12, color: BRAND, fontWeight: 600 }}>Drop to upload…</p>
                )}
              </div>
            )}
          </UploadClipCard>

          <Card title="analysis results" icon="analytics">
            <div style={{ textAlign: "center", padding: "36px 0", color: "#a3a3a3" }}>
              <Icon name="pending" size={32} style={{ display: "block", margin: "0 auto 10px", color: "#d4d4d4" }} />
              <p style={{ margin: 0, fontSize: 14 }}>Upload a clip to get started</p>
            </div>
          </Card>
        </AppChrome>

        {frame > 6 && frame < cursorHide && !showPreview && (
          <div
            style={{
              position: "absolute",
              left: interpolate(frame, [dragStart, dragEnd + 6], [101, 116], {
                extrapolateLeft: "clamp",
                extrapolateRight: "clamp",
              }),
              top: cursorY,
              width: 10,
              height: 10,
              borderRadius: "50%",
              backgroundColor: "rgba(255,255,255,0.85)",
              border: "2px solid rgba(0,0,0,0.15)",
              boxShadow: "0 4px 16px rgba(0,0,0,0.2)",
              pointerEvents: "none",
            }}
          />
        )}
      </PhoneViewport>
    </SceneShell>
  );
};
