import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Focus, Maximize2, Minus, Plus, ScanSearch } from "lucide-react";
import type { DiffMode, DiffPair } from "../types";
import { clamp, heatColor, pixelError } from "../imageMath";
import { isCanvasZoomGesture } from "../interactions";

type LoadedImage = {
  image: CanvasImageSource;
  width: number;
  height: number;
  pixels: ImageData;
};

type Transform = { zoom: number; x: number; y: number };
type Point = { x: number; y: number };

function useLoadedImage(url: string | null) {
  const [result, setResult] = useState<{ value: LoadedImage | null; loading: boolean; error: string | null }>({
    value: null,
    loading: Boolean(url),
    error: null,
  });
  useEffect(() => {
    let active = true;
    if (!url) {
      setResult({ value: null, loading: false, error: "Image is unavailable" });
      return undefined;
    }
    setResult({ value: null, loading: true, error: null });
    const image = new Image();
    image.onload = () => {
      if (!active) return;
      const canvas = document.createElement("canvas");
      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;
      const context = canvas.getContext("2d", { willReadFrequently: true });
      if (!context) {
        setResult({ value: null, loading: false, error: "Canvas is unavailable" });
        return;
      }
      context.drawImage(image, 0, 0);
      setResult({
        value: { image, width: image.naturalWidth, height: image.naturalHeight, pixels: context.getImageData(0, 0, canvas.width, canvas.height) },
        loading: false,
        error: null,
      });
    };
    image.onerror = () => active && setResult({ value: null, loading: false, error: "Could not load image" });
    image.src = url;
    return () => {
      active = false;
      image.src = "";
    };
  }, [url]);
  return result;
}

function buildDiff(
  first: LoadedImage | null,
  second: LoadedImage | null,
  mode: DiffMode,
  gain: number,
  threshold: number,
  opacity: number,
): LoadedImage | null {
  if (!first || !second) return null;
  const { width, height } = first.pixels;
  if (width !== second.pixels.width || height !== second.pixels.height) return null;
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) return null;
  const output = context.createImageData(width, height);
  const a = first.pixels.data;
  const b = second.pixels.data;
  for (let offset = 0; offset < output.data.length; offset += 4) {
    const error = pixelError(a, b, offset);
    const visible = error.mean >= threshold;
    const color = mode === "heatmap"
      ? heatColor((error.mean * gain) / 255)
      : [clamp(error.red * gain, 0, 255), clamp(error.green * gain, 0, 255), clamp(error.blue * gain, 0, 255)];
    for (let channel = 0; channel < 3; channel += 1) {
      const overlay = visible ? color[channel] : 0;
      output.data[offset + channel] = Math.round(b[offset + channel] * (1 - opacity) + overlay * opacity);
    }
    output.data[offset + 3] = 255;
  }
  context.putImageData(output, 0, 0);
  return { image: canvas, width, height, pixels: output };
}

type CanvasPanelProps = {
  title: string;
  badge?: string;
  loaded: LoadedImage | null;
  loading?: boolean;
  error?: string | null;
  transform: Transform;
  setTransform: (next: Transform | ((current: Transform) => Transform)) => void;
  onInspect: (point: Point | null) => void;
};

function CanvasPanel({ title, badge, loaded, loading, error, transform, setTransform, onInspect }: CanvasPanelProps) {
  const shellRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const dragRef = useRef<{ pointer: number; x: number; y: number } | null>(null);

  const draw = useCallback(() => {
    const shell = shellRef.current;
    const canvas = canvasRef.current;
    if (!shell || !canvas) return;
    const rect = shell.getBoundingClientRect();
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(rect.width * ratio));
    canvas.height = Math.max(1, Math.round(rect.height * ratio));
    canvas.style.width = `${rect.width}px`;
    canvas.style.height = `${rect.height}px`;
    const context = canvas.getContext("2d");
    if (!context) return;
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.fillStyle = "#070a12";
    context.fillRect(0, 0, rect.width, rect.height);
    if (!loaded) return;
    const fit = Math.min(rect.width / loaded.width, rect.height / loaded.height);
    const scale = fit * transform.zoom;
    const width = loaded.width * scale;
    const height = loaded.height * scale;
    const left = (rect.width - width) / 2 + transform.x;
    const top = (rect.height - height) / 2 + transform.y;
    context.imageSmoothingEnabled = transform.zoom < 8;
    context.drawImage(loaded.image, left, top, width, height);
  }, [loaded, transform]);

  useEffect(() => {
    draw();
    const observer = new ResizeObserver(draw);
    if (shellRef.current) observer.observe(shellRef.current);
    return () => observer.disconnect();
  }, [draw]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return undefined;
    const handleWheel = (event: WheelEvent) => {
      if (!isCanvasZoomGesture(event)) return;
      event.preventDefault();
      const factor = event.deltaY < 0 ? 1.15 : 1 / 1.15;
      setTransform((current) => ({ ...current, zoom: clamp(current.zoom * factor, 1, 32) }));
    };
    canvas.addEventListener("wheel", handleWheel, { passive: false });
    return () => canvas.removeEventListener("wheel", handleWheel);
  }, [setTransform]);

  const pointFromEvent = (event: React.PointerEvent<HTMLCanvasElement>): Point | null => {
    if (!loaded || !shellRef.current) return null;
    const rect = shellRef.current.getBoundingClientRect();
    const fit = Math.min(rect.width / loaded.width, rect.height / loaded.height);
    const scale = fit * transform.zoom;
    const x = Math.floor((event.clientX - rect.left - (rect.width - loaded.width * scale) / 2 - transform.x) / scale);
    const y = Math.floor((event.clientY - rect.top - (rect.height - loaded.height * scale) / 2 - transform.y) / scale);
    return x >= 0 && y >= 0 && x < loaded.width && y < loaded.height ? { x, y } : null;
  };

  return (
    <section className="canvas-card">
      <header><span>{title}</span>{badge && <small>{badge}</small>}</header>
      <div ref={shellRef} className="canvas-shell">
        <canvas
          ref={canvasRef}
          onPointerDown={(event) => {
            event.currentTarget.setPointerCapture(event.pointerId);
            dragRef.current = { pointer: event.pointerId, x: event.clientX, y: event.clientY };
          }}
          onPointerMove={(event) => {
            if (dragRef.current?.pointer === event.pointerId) {
              const dx = event.clientX - dragRef.current.x;
              const dy = event.clientY - dragRef.current.y;
              dragRef.current = { pointer: event.pointerId, x: event.clientX, y: event.clientY };
              setTransform((current) => ({ ...current, x: current.x + dx, y: current.y + dy }));
            }
            onInspect(pointFromEvent(event));
          }}
          onPointerUp={() => { dragRef.current = null; }}
          onPointerLeave={() => { dragRef.current = null; onInspect(null); }}
        />
        {loading && <div className="canvas-message"><span className="spinner" />Loading</div>}
        {!loading && !loaded && <div className="canvas-message error">{error ?? "Unavailable"}</div>}
      </div>
    </section>
  );
}

type Props = {
  urls: { prediction: string | null; input: string | null; gt: string | null };
  pair: DiffPair;
  onPairChange: (pair: DiffPair) => void;
};

export function ImageWorkspace({ urls, pair, onPairChange }: Props) {
  const prediction = useLoadedImage(urls.prediction);
  const input = useLoadedImage(urls.input);
  const gt = useLoadedImage(urls.gt);
  const [mode, setMode] = useState<DiffMode>("heatmap");
  const [gain, setGain] = useState(4);
  const [threshold, setThreshold] = useState(0);
  const [opacity, setOpacity] = useState(0.85);
  const [transform, setTransform] = useState<Transform>({ zoom: 1, x: 0, y: 0 });
  const [point, setPoint] = useState<Point | null>(null);

  useEffect(() => setTransform({ zoom: 1, x: 0, y: 0 }), [urls.prediction, urls.input, urls.gt]);

  const sources = { prediction: prediction.value, input: input.value, gt: gt.value };
  const [firstName, secondName] = pair.split("-") as [keyof typeof sources, keyof typeof sources];
  const diff = useMemo(
    () => buildDiff(sources[firstName], sources[secondName], mode, gain, threshold, opacity),
    [sources[firstName], sources[secondName], mode, gain, threshold, opacity],
  );

  const inspector = useMemo(() => {
    if (!point) return null;
    const sample = (loaded: LoadedImage | null) => {
      if (!loaded || point.x >= loaded.pixels.width || point.y >= loaded.pixels.height) return null;
      const offset = (point.y * loaded.pixels.width + point.x) * 4;
      return Array.from(loaded.pixels.data.slice(offset, offset + 3));
    };
    const pred = sample(prediction.value);
    const inp = sample(input.value);
    const truth = sample(gt.value);
    const selectedA = sample(sources[firstName]);
    const selectedB = sample(sources[secondName]);
    const error = selectedA && selectedB
      ? selectedA.reduce((sum, channel, index) => sum + Math.abs(channel - selectedB[index]), 0) / 3
      : null;
    return { pred, inp, truth, error };
  }, [point, prediction.value, input.value, gt.value, sources, firstName, secondName]);

  const panels = [
    { title: "Prediction", result: prediction },
    { title: "Input", result: input },
    { title: "Ground truth", result: gt },
  ];

  return (
    <div className="workspace">
      <div className="workspace-toolbar">
        <div className="segmented" aria-label="Difference image pair">
          {(["prediction-gt", "input-gt", "prediction-input"] as DiffPair[]).map((value) => (
            <button key={value} className={pair === value ? "active" : ""} onClick={() => onPairChange(value)}>
              {value.replace("-", " → ")}
            </button>
          ))}
        </div>
        <div className="zoom-controls">
          <span className="zoom-hint">Ctrl/⌘ + wheel</span>
          <button title="Zoom out" onClick={() => setTransform((value) => ({ ...value, zoom: clamp(value.zoom / 1.5, 1, 32) }))}><Minus size={16} /></button>
          <span>{transform.zoom.toFixed(transform.zoom < 10 ? 1 : 0)}×</span>
          <button title="Zoom in" onClick={() => setTransform((value) => ({ ...value, zoom: clamp(value.zoom * 1.5, 1, 32) }))}><Plus size={16} /></button>
          <button title="Fit images" onClick={() => setTransform({ zoom: 1, x: 0, y: 0 })}><Maximize2 size={16} /></button>
          <button title="One-to-one pixels" onClick={() => setTransform({ zoom: 4, x: 0, y: 0 })}><Focus size={16} /></button>
        </div>
      </div>

      <div className="canvas-grid">
        {panels.map(({ title, result }) => (
          <CanvasPanel
            key={title}
            title={title}
            loaded={result.value}
            loading={result.loading}
            error={result.error}
            transform={transform}
            setTransform={setTransform}
            onInspect={setPoint}
          />
        ))}
        <CanvasPanel
          title="Pixel difference"
          badge={`${firstName} − ${secondName}`}
          loaded={diff}
          error={sources[firstName] && sources[secondName] ? "Image dimensions differ" : "Source image unavailable"}
          transform={transform}
          setTransform={setTransform}
          onInspect={setPoint}
        />
      </div>

      <div className="diff-controls">
        <label>
          Display
          <select value={mode} onChange={(event) => setMode(event.target.value as DiffMode)}>
            <option value="heatmap">Scalar heatmap</option>
            <option value="rgb">Absolute RGB</option>
          </select>
        </label>
        <label>
          Amplification <strong>{gain.toFixed(1)}×</strong>
          <input type="range" min="1" max="12" step="0.5" value={gain} onChange={(event) => setGain(Number(event.target.value))} />
        </label>
        <label>
          Threshold <strong>{threshold}</strong>
          <input type="range" min="0" max="64" step="1" value={threshold} onChange={(event) => setThreshold(Number(event.target.value))} />
        </label>
        <label>
          Overlay <strong>{Math.round(opacity * 100)}%</strong>
          <input type="range" min="0" max="1" step="0.05" value={opacity} onChange={(event) => setOpacity(Number(event.target.value))} />
        </label>
        <div className="pixel-inspector">
          <ScanSearch size={16} />
          {point && inspector ? (
            <>
              <span>x {point.x} · y {point.y}</span>
              <span>Pred {inspector.pred?.join(",") ?? "—"}</span>
              <span>Input {inspector.inp?.join(",") ?? "—"}</span>
              <span>GT {inspector.truth?.join(",") ?? "—"}</span>
              <strong>MAE {inspector.error?.toFixed(1) ?? "—"}</strong>
            </>
          ) : <span>Hover an image to inspect pixels</span>}
        </div>
      </div>
    </div>
  );
}
