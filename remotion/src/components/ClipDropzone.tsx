import { OffthreadVideo, staticFile } from "remotion";
import { Icon } from "./Icon";
import { BRAND, CLIP_MAX_HEIGHT } from "../theme";

type ClipDropzoneProps = {
  showChrome?: boolean;
};

/** Video + optional footer chrome only — outer dashed frame lives on UploadClipCard. */
export const ClipDropzone = ({ showChrome = true }: ClipDropzoneProps) => (
  <>
    <OffthreadVideo
      src={staticFile("rally-clip.mov")}
      style={{
        width: "100%",
        maxHeight: CLIP_MAX_HEIGHT,
        objectFit: "contain",
        backgroundColor: "#000",
        display: "block",
      }}
    />
    {showChrome && (
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "8px 12px",
          backgroundColor: "#fff",
          borderTop: "1px solid #e5e5e5",
        }}
      >
        <span style={{ fontSize: 12, color: "#737373", display: "flex", gap: 6, alignItems: "center" }}>
          <Icon name="check_circle" size={13} style={{ color: BRAND }} />
          Clip ready to analyze
        </span>
        <span style={{ fontSize: 12, color: "#a3a3a3" }}>Change Video</span>
      </div>
    )}
  </>
);
