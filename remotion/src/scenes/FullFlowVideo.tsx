import { Series } from "remotion";
import { AnalyzingVideo } from "./AnalyzingVideo";
import { ResultsVideo } from "./ResultsVideo";
import { UploadVideo } from "./UploadVideo";
import {
  ANALYZING_DURATION,
  RESULTS_DURATION,
  UPLOAD_DURATION,
} from "../sceneTiming";

/** Upload → analyze → results in one continuous phone demo. */
export const FullFlowVideo = () => {
  return (
    <Series>
      <Series.Sequence durationInFrames={UPLOAD_DURATION}>
        <UploadVideo />
      </Series.Sequence>
      <Series.Sequence durationInFrames={ANALYZING_DURATION}>
        <AnalyzingVideo />
      </Series.Sequence>
      <Series.Sequence durationInFrames={RESULTS_DURATION}>
        <ResultsVideo />
      </Series.Sequence>
    </Series>
  );
};
