import { interpolate, useCurrentFrame } from "remotion";

type ScrollableContentProps = {
  children: React.ReactNode;
  /** Frame range where content scrolls up (negative translateY). */
  scrollFromFrame?: number;
  scrollToFrame?: number;
  /** Logical px to move content up (negative = scroll down the page). */
  scrollDistance?: number;
};

/** Simulates finger-scroll inside the phone so content below the fold is visible. */
export const ScrollableContent = ({
  children,
  scrollFromFrame = 40,
  scrollToFrame = 200,
  scrollDistance = -280,
}: ScrollableContentProps) => {
  const frame = useCurrentFrame();
  const scrollY = interpolate(
    frame,
    [scrollFromFrame, scrollToFrame],
    [0, scrollDistance],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" },
  );

  return <div style={{ transform: `translateY(${scrollY}px)` }}>{children}</div>;
};
