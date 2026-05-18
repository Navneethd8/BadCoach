/**
 * POST sample clip to IsoCourt backend and write demo-result.json.
 * Usage: node scripts/fetch-demo-result.mjs
 * Requires backend at http://127.0.0.1:8000
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const API = process.env.VITE_API_URL || "http://127.0.0.1:8000";
const clipPath =
  process.env.CLIP_PATH ||
  path.join(__dirname, "../public/rally-clip.mov");
const outPath = path.join(__dirname, "../src/data/demo-result.json");

if (!fs.existsSync(clipPath)) {
  console.error("Clip not found:", clipPath);
  process.exit(1);
}

const ext = path.extname(clipPath).toLowerCase();
const mime =
  ext === ".mov" ? "video/quicktime" : ext === ".mp4" ? "video/mp4" : "application/octet-stream";
const clip = fs.readFileSync(clipPath);
const form = new FormData();
form.append("file", new Blob([clip], { type: mime }), path.basename(clipPath));
console.log("Analyzing", clipPath);

console.log("Creating clip job…");
const jobRes = await fetch(`${API}/clips/jobs`, { method: "POST", body: form });
if (!jobRes.ok) {
  console.error("Job create failed:", jobRes.status, await jobRes.text());
  process.exit(1);
}
const { job_id: jobId } = await jobRes.json();
console.log("Job", jobId);

const streamRes = await fetch(`${API}/clips/jobs/${jobId}/stream`);
if (!streamRes.ok || !streamRes.body) {
  console.error("Stream failed:", streamRes.status);
  process.exit(1);
}

const reader = streamRes.body.getReader();
const decoder = new TextDecoder();
let buffer = "";
let summary = null;

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  buffer += decoder.decode(value, { stream: true });
  const parts = buffer.split("\n\n");
  buffer = parts.pop() || "";
  for (const part of parts) {
    const dataLine = part.split("\n").find((l) => l.startsWith("data: "));
    if (!dataLine) continue;
    const parsed = JSON.parse(dataLine.slice(6));
    if ((parsed.event === "done" || parsed.event === "complete") && parsed.summary) {
      summary = parsed.summary;
    }
    if (parsed.event === "error") {
      console.error("Analysis error:", parsed.error);
      process.exit(1);
    }
  }
}

if (!summary) {
  console.error("No summary in stream");
  process.exit(1);
}

const timeline = summary.timeline?.map(({ pose_image, ...rest }) => rest) ?? [];
const payload = { ...summary, timeline };
fs.writeFileSync(outPath, JSON.stringify(payload, null, 2));
console.log("Wrote", outPath);
