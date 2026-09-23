import { RefObject, useEffect, useRef } from "react";
import { PoseFrames } from "../api";

const LOW_VIS = 0.5;

interface Props {
  src: string;
  fps: number;
  numFrames: number;
  pose: PoseFrames | null;
  frame: number;
  showSkeleton: boolean;
  videoRef: RefObject<HTMLVideoElement | null>;
  onFrame: (frame: number) => void;
}

/**
 * <video> with a canvas skeleton on top. Frame index = round(mediaTime * fps), which matches the
 * backend because pose was run on this exact constant-frame-rate proxy.
 */
export default function VideoOverlay({ src, fps, numFrames, pose, frame, showSkeleton, videoRef, onFrame }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Track the displayed frame.
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    const clamp = (f: number) => Math.max(0, Math.min(numFrames - 1, f));
    if (typeof video.requestVideoFrameCallback === "function") {
      let handle = 0;
      const cb = (_now: number, meta: VideoFrameCallbackMetadata) => {
        onFrame(clamp(Math.round(meta.mediaTime * fps)));
        handle = video.requestVideoFrameCallback(cb);
      };
      handle = video.requestVideoFrameCallback(cb);
      return () => video.cancelVideoFrameCallback(handle);
    }
    const fallback = () => onFrame(clamp(Math.floor(video.currentTime * fps + 1e-6)));
    video.addEventListener("timeupdate", fallback);
    video.addEventListener("seeked", fallback);
    return () => {
      video.removeEventListener("timeupdate", fallback);
      video.removeEventListener("seeked", fallback);
    };
  }, [videoRef, fps, numFrames, onFrame]);

  // Draw the skeleton for the current frame.
  useEffect(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    const draw = () => {
      const w = video.clientWidth;
      const h = video.clientHeight;
      const dpr = window.devicePixelRatio || 1;
      if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
        canvas.width = w * dpr;
        canvas.height = h * dpr;
        canvas.style.width = `${w}px`;
        canvas.style.height = `${h}px`;
      }
      const ctx = canvas.getContext("2d")!;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      const kp = pose?.frames[frame];
      if (!showSkeleton || !pose || !kp) return;

      // object-fit: contain letterboxing
      const scale = Math.min(w / pose.width, h / pose.height);
      const ox = (w - pose.width * scale) / 2;
      const oy = (h - pose.height * scale) / 2;
      const P = (j: number) => [ox + kp[j * 3] * scale, oy + kp[j * 3 + 1] * scale, kp[j * 3 + 2]] as const;

      ctx.lineWidth = 3;
      ctx.lineCap = "round";
      for (const [a, b] of pose.connections) {
        const [x1, y1, v1] = P(a);
        const [x2, y2, v2] = P(b);
        ctx.strokeStyle = Math.min(v1, v2) < LOW_VIS ? "rgba(255,120,80,0.7)" : "rgba(80,230,160,0.9)";
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
      }
      for (let j = 0; j < kp.length / 3; j++) {
        if (j > 0 && j < 11) continue; // skip face detail except nose
        const [x, y, v] = P(j);
        ctx.fillStyle = v < LOW_VIS ? "#ff7850" : "#ffffff";
        ctx.beginPath();
        ctx.arc(x, y, 3.5, 0, Math.PI * 2);
        ctx.fill();
      }
    };

    draw();
    const ro = new ResizeObserver(draw);
    ro.observe(video);
    return () => ro.disconnect();
  }, [videoRef, pose, frame, showSkeleton]);

  return (
    <div className="video-wrap">
      <video ref={videoRef} src={src} playsInline muted preload="auto" />
      <canvas ref={canvasRef} />
    </div>
  );
}
