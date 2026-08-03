import type { Metrics, ViewRow } from "../types";

type MetricSets = {
  prediction: Metrics | null;
  input: Metrics | null;
  gt_low_res: Metrics | null;
  gt_high_res: Metrics | null;
};

const labels: Array<[keyof MetricSets, string]> = [
  ["prediction", "Prediction"],
  ["input", "Input GS"],
  ["gt_low_res", "GT low-res GS"],
  ["gt_high_res", "GT high-res GS"],
];

function value(metrics: Metrics | null, metric: keyof Metrics) {
  if (!metrics) return "—";
  return metrics[metric].toFixed(metric === "psnr" ? 2 : 4);
}

export function IterationMetricMatrix({
  metricSets,
  iteration,
  referenceIteration,
}: {
  metricSets: MetricSets | null;
  iteration: string | null;
  referenceIteration: string;
}) {
  return (
    <section className="metric-overview panel">
      <div className="metric-overview-heading">
        <div>
          <span className="eyebrow">ITERATION METRICS</span>
          <h2>Quality across Gaussian sources</h2>
        </div>
        <span>Prediction: step {iteration ? Number(iteration).toLocaleString() : "—"} · Baselines: step {Number(referenceIteration).toLocaleString()}</span>
      </div>
      <div className="metric-set-grid metric-set-header">
        <span>Source</span><span>PSNR ↑</span><span>SSIM ↑</span><span>LPIPS ↓</span>
      </div>
      {labels.map(([key, label]) => {
        const metrics = metricSets?.[key] ?? null;
        return (
          <div className={`metric-set-grid ${key === "prediction" ? "current" : ""}`} key={key}>
            <span className="metric-source"><strong>{label}</strong><small>{key === "prediction" ? "current" : "reference"}</small></span>
            <strong>{value(metrics, "psnr")}</strong>
            <strong>{value(metrics, "ssim")}</strong>
            <strong>{value(metrics, "lpips")}</strong>
          </div>
        );
      })}
    </section>
  );
}

export function ViewMetricStrip({ view }: { view: ViewRow }) {
  const rows: Array<[string, Metrics | null, string]> = [
    ["Prediction", view.metrics, "current"],
    ["Input GS", view.input_metrics, "reference"],
    ["GT low-res", view.gt_low_res_metrics, "reference"],
    ["GT high-res", view.gt_high_res_metrics, "reference"],
  ];
  return (
    <div className="view-metric-strip">
      {rows.map(([label, metrics, source]) => (
        <div key={label} className={label === "Prediction" ? "current" : ""}>
          <span><strong>{label}</strong><small>{source}</small></span>
          <span><small>PSNR</small>{value(metrics, "psnr")}</span>
          <span><small>SSIM</small>{value(metrics, "ssim")}</span>
          <span><small>LPIPS</small>{value(metrics, "lpips")}</span>
        </div>
      ))}
    </div>
  );
}
