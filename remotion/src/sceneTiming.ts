/** Shared frame timings @ 30fps — keep scenes and Root durations in sync. */
export const FPS = 30;

/** ~2s — ends on “Clip ready” after drop + preview. */
export const UPLOAD_DURATION = 62;
export const UPLOAD_TIMING = {
  dragStart: 14,
  dragEnd: 22,
  previewAt: 26,
  cursorHide: 48,
} as const;

/** ~3.5s — ends when all loading steps are checked off. */
export const ANALYZING_DURATION = 105;
export const ANALYZING_TIMING = {
  stepStarts: [0, 20, 40, 60] as number[],
  allStepsDoneAt: 78,
  scrollFrom: 28,
  scrollTo: 82,
  scrollDistance: -300,
} as const;

/** ~7s — scrolls to full play-by-play (10 timeline events); 1s trimmed from intro. */
export const RESULTS_DURATION = 210;
export const RESULTS_TIMING = {
  revealAt: 0,
  scrollFrom: 0,
  scrollTo: 160,
  scrollDistance: -980,
  coachTipBase: 20,
  coachTipStagger: 10,
  timelineBase: 48,
  timelineStagger: 10,
} as const;

/** 01 + 02 + 03 @ 30fps (~12.6s). */
export const FULL_FLOW_DURATION =
  UPLOAD_DURATION + ANALYZING_DURATION + RESULTS_DURATION;
