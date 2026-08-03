export type Metrics = {
  psnr: number;
  ssim: number;
  lpips: number;
};

export type Iteration = {
  name: string;
  step: number;
  scene_count: number;
  metrics: Metrics | null;
  metric_sets: {
    prediction: Metrics | null;
    input: Metrics | null;
    gt_low_res: Metrics | null;
    gt_high_res: Metrics | null;
  };
  is_default: boolean;
  is_reference: boolean;
  modified_ns: number;
};

export type IterationPayload = {
  eval_dir: string;
  reference_iteration: string;
  reference_available: boolean;
  default_iteration: string | null;
  iteration_count: number;
  warnings: string[];
  reference_scene_count: number;
  items: Iteration[];
};

export type MetricRow = {
  metrics: Metrics | null;
  input_metrics: Metrics | null;
  gt_low_res_metrics: Metrics | null;
  gt_high_res_metrics: Metrics | null;
  gain: Metrics | null;
};

export type SceneRow = MetricRow & {
  scene_idx: number | null;
  scene_name: string;
  reference_available: boolean;
  view_metrics_available: boolean;
};

export type ViewRow = MetricRow & {
  image_id: number | null;
  image_name: string;
  images: {
    prediction: boolean;
    input: boolean;
    gt: boolean;
  };
};

export type Page<T> = {
  items: T[];
  total: number;
  page: number;
  page_size: number;
  pages: number;
  sort: string;
  order: "asc" | "desc";
};

export type DiffPair = "prediction-gt" | "input-gt" | "prediction-input";
export type DiffMode = "heatmap" | "rgb";
