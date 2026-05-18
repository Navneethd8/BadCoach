import { AbsoluteFill, useVideoConfig } from "remotion";

/** iPhone 14 logical width — UI is authored at this width, then scaled to 1080×1920. */
export const MOBILE_WIDTH = 390;
export const MOBILE_HEIGHT = 844;

export const PhoneViewport = ({ children }: { children: React.ReactNode }) => {
  const { width } = useVideoConfig();
  const scale = width / MOBILE_WIDTH;

  return (
    <AbsoluteFill style={{ backgroundColor: "#fafafa", overflow: "hidden" }}>
      <div
        style={{
          position: "relative",
          width: MOBILE_WIDTH,
          height: MOBILE_HEIGHT,
          transform: `scale(${scale})`,
          transformOrigin: "top left",
          overflow: "hidden",
        }}
      >
        {children}
      </div>
    </AbsoluteFill>
  );
};
