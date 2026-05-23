import { Composition } from "remotion";
import { AnalyzingVideo } from "./scenes/AnalyzingVideo";
import { FullFlowVideo } from "./scenes/FullFlowVideo";
import { ResultsVideo } from "./scenes/ResultsVideo";
import { UploadVideo } from "./scenes/UploadVideo";
import {
  ANALYZING_DURATION,
  FPS,
  FULL_FLOW_DURATION,
  RESULTS_DURATION,
  UPLOAD_DURATION,
} from "./sceneTiming";

const WIDTH = 1080;
const HEIGHT = 1920;

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition
        id="01-upload"
        component={UploadVideo}
        durationInFrames={UPLOAD_DURATION}
        fps={FPS}
        width={WIDTH}
        height={HEIGHT}
      />
      <Composition
        id="02-analyzing"
        component={AnalyzingVideo}
        durationInFrames={ANALYZING_DURATION}
        fps={FPS}
        width={WIDTH}
        height={HEIGHT}
      />
      <Composition
        id="03-results"
        component={ResultsVideo}
        durationInFrames={RESULTS_DURATION}
        fps={FPS}
        width={WIDTH}
        height={HEIGHT}
      />
      <Composition
        id="04-full-flow"
        component={FullFlowVideo}
        durationInFrames={FULL_FLOW_DURATION}
        fps={FPS}
        width={WIDTH}
        height={HEIGHT}
      />
    </>
  );
};
