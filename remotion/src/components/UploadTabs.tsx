import { Icon } from "./Icon";
import { MONO } from "../theme";

export const UploadTabs = () => (
  <div
    style={{
      display: "flex",
      gap: 4,
      marginBottom: 14,
      padding: 4,
      width: "fit-content",
      borderRadius: 8,
      backgroundColor: "#ececec",
    }}
  >
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 6,
        padding: "6px 12px",
        borderRadius: 6,
        backgroundColor: "#fff",
        boxShadow: "0 1px 2px rgba(0,0,0,0.06)",
        fontFamily: MONO,
        fontSize: 12,
        fontWeight: 600,
      }}
    >
      <Icon name="upload" size={14} />
      Upload
    </div>
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 6,
        padding: "6px 12px",
        color: "#737373",
        fontFamily: MONO,
        fontSize: 12,
        fontWeight: 600,
      }}
    >
      <Icon name="videocam" size={14} />
      Record
    </div>
  </div>
);
