import { Card } from "./Card";
import { ClipDropzone } from "./ClipDropzone";
import { UploadTabs } from "./UploadTabs";
import { dropzoneShellStyle } from "../theme";

type UploadClipCardProps = {
  footer?: React.ReactNode;
  /** Custom dropzone interior (e.g. empty upload state). Defaults to clip preview. */
  children?: React.ReactNode;
  dropzoneBorderColor?: string;
  dropzoneBackground?: string;
};

/**
 * Upload card with tabs + dashed clip frame — identical layout in all three compositions.
 */
export const UploadClipCard = ({
  footer,
  children,
  dropzoneBorderColor = "#c4c4c4",
  dropzoneBackground = "#000",
}: UploadClipCardProps) => (
  <Card>
    <div style={{ pointerEvents: "none" }}>
      <UploadTabs />
      <div style={dropzoneShellStyle(dropzoneBorderColor, dropzoneBackground)}>
        {children ?? <ClipDropzone />}
      </div>
      {footer}
    </div>
  </Card>
);
