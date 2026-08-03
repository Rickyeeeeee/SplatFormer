import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import type { SortingState } from "@tanstack/react-table";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Database,
  Image as ImageIcon,
  RefreshCw,
  Search,
  Sparkles,
} from "lucide-react";
import { api, imageURL } from "./api";
import { ImageWorkspace } from "./components/ImageWorkspace";
import { IterationMetricMatrix, ViewMetricStrip } from "./components/MetricOverview";
import { MetricTable } from "./components/ConfigurableMetricTable";
import type { DiffPair, Metrics, SceneRow, ViewRow } from "./types";

function queryParam(name: string) {
  return new URLSearchParams(window.location.search).get(name);
}

function parseSort(value: string | null, fallback: SortingState): SortingState {
  if (!value) return fallback;
  const [id, direction] = value.split(":");
  return id ? [{ id, desc: direction !== "asc" }] : fallback;
}

function formatStep(value: string) {
  return Number(value).toLocaleString();
}

function metricSummary(metrics: Metrics | null, metric: keyof Metrics) {
  if (!metrics) return "—";
  return metrics[metric].toFixed(metric === "psnr" ? 2 : 4);
}

function ErrorState({ error }: { error: unknown }) {
  return (
    <div className="error-state">
      <AlertTriangle size={20} />
      <div><strong>Could not load evaluation data</strong><span>{error instanceof Error ? error.message : String(error)}</span></div>
    </div>
  );
}

export default function Dashboard() {
  const queryClient = useQueryClient();
  const [iteration, setIteration] = useState<string | null>(() => queryParam("iteration"));
  const [scene, setScene] = useState<string | null>(() => queryParam("scene"));
  const [view, setView] = useState<string | null>(() => queryParam("view"));
  const [sceneSearch, setSceneSearch] = useState("");
  const [viewSearch, setViewSearch] = useState("");
  const [sceneSorting, setSceneSorting] = useState<SortingState>(() => parseSort(queryParam("sceneSort"), [{ id: "gain.psnr", desc: true }]));
  const [viewSorting, setViewSorting] = useState<SortingState>(() => parseSort(queryParam("viewSort"), [{ id: "gain.psnr", desc: true }]));
  const [pair, setPair] = useState<DiffPair>(() => {
    const value = queryParam("pair");
    return value === "input-gt" || value === "prediction-input" ? value : "prediction-gt";
  });

  const iterationQuery = useQuery({ queryKey: ["iterations"], queryFn: api.iterations });
  const iterationPayload = iterationQuery.data;

  useEffect(() => {
    if (!iterationPayload?.items.length) return;
    const exists = iterationPayload.items.some((item) => item.name === iteration);
    if (!exists) setIteration(iterationPayload.default_iteration ?? iterationPayload.items[0].name);
  }, [iterationPayload, iteration]);

  const scenesQuery = useQuery({
    queryKey: ["scenes", iteration, sceneSearch],
    queryFn: () => api.scenes(iteration!, sceneSearch),
    enabled: Boolean(iteration),
  });
  const scenes = useMemo(() => scenesQuery.data?.items ?? [], [scenesQuery.data]);

  useEffect(() => {
    if (!scenesQuery.data) return;
    if (!scenes.some((item) => item.scene_name === scene)) {
      setScene(scenes[0]?.scene_name ?? null);
      setView(null);
    }
  }, [scenesQuery.data, scenes, scene]);

  const viewsQuery = useQuery({
    queryKey: ["views", iteration, scene, viewSearch],
    queryFn: () => api.views(iteration!, scene!, viewSearch),
    enabled: Boolean(iteration && scene),
  });
  const views = useMemo(() => viewsQuery.data?.items ?? [], [viewsQuery.data]);

  useEffect(() => {
    if (!viewsQuery.data) return;
    if (!views.some((item) => item.image_name === view)) setView(views[0]?.image_name ?? null);
  }, [viewsQuery.data, views, view]);

  useEffect(() => {
    const params = new URLSearchParams();
    if (iteration) params.set("iteration", iteration);
    if (scene) params.set("scene", scene);
    if (view) params.set("view", view);
    if (sceneSorting[0]) params.set("sceneSort", `${sceneSorting[0].id}:${sceneSorting[0].desc ? "desc" : "asc"}`);
    if (viewSorting[0]) params.set("viewSort", `${viewSorting[0].id}:${viewSorting[0].desc ? "desc" : "asc"}`);
    params.set("pair", pair);
    window.history.replaceState(null, "", `${window.location.pathname}?${params.toString()}`);
  }, [iteration, scene, view, sceneSorting, viewSorting, pair]);

  const refresh = useMutation({
    mutationFn: api.refresh,
    onSuccess: async () => {
      await queryClient.invalidateQueries();
    },
  });

  const activeIteration = iterationPayload?.items.find((item) => item.name === iteration) ?? null;
  const activeScene = scenes.find((item) => item.scene_name === scene) ?? null;
  const activeView = views.find((item) => item.image_name === view) ?? null;
  const imageURLs = useMemo(() => ({
    prediction: iteration && scene && view && activeView?.images.prediction ? imageURL(iteration, scene, view, "prediction") : null,
    input: iteration && scene && view && activeView?.images.input ? imageURL(iteration, scene, view, "input") : null,
    gt: iteration && scene && view && activeView?.images.gt ? imageURL(iteration, scene, view, "gt") : null,
  }), [iteration, scene, view, activeView]);

  if (iterationQuery.isLoading) {
    return <main className="boot-state"><span className="brand-mark"><Sparkles /></span><h1>Indexing evaluations</h1><span className="spinner" /></main>;
  }
  if (iterationQuery.error) return <main className="boot-state"><ErrorState error={iterationQuery.error} /></main>;

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark"><Sparkles size={20} /></span>
          <div><strong>Evaluation Explorer</strong><span>SplatFormer image diagnostics</span></div>
        </div>
        <div className="dataset-path"><Database size={14} /><span title={iterationPayload?.eval_dir}>{iterationPayload?.eval_dir}</span></div>
        <div className="header-actions">
          <label className="iteration-select">
            <span>Checkpoint</span>
            <select value={iteration ?? ""} onChange={(event) => { setIteration(event.target.value); setScene(null); setView(null); }}>
              {iterationPayload?.items.map((item) => (
                <option key={item.name} value={item.name}>Step {formatStep(item.name)}{item.is_default ? " · final" : ""}</option>
              ))}
            </select>
          </label>
          <button className="refresh-button" onClick={() => refresh.mutate()} disabled={refresh.isPending}>
            <RefreshCw size={16} className={refresh.isPending ? "spinning" : ""} /> Refresh
          </button>
        </div>
      </header>

      {iterationPayload?.warnings.length ? (
        <div className="warning-strip"><AlertTriangle size={14} />{iterationPayload.warnings.length} incomplete iteration{iterationPayload.warnings.length === 1 ? "" : "s"} hidden</div>
      ) : null}

      <section className="summary-row">
        <div className="summary-card accent"><Activity size={18} /><span>Selected step</span><strong>{iteration ? formatStep(iteration) : "—"}</strong><small>{activeIteration?.scene_count ?? 0} scenes</small></div>
        <div className="summary-card"><BarChart3 size={18} /><span>Mean PSNR</span><strong>{metricSummary(activeIteration?.metrics ?? null, "psnr")}</strong><small>prediction vs GT</small></div>
        <div className="summary-card"><BarChart3 size={18} /><span>Mean SSIM</span><strong>{metricSummary(activeIteration?.metrics ?? null, "ssim")}</strong><small>prediction vs GT</small></div>
        <div className="summary-card"><BarChart3 size={18} /><span>Mean LPIPS</span><strong>{metricSummary(activeIteration?.metrics ?? null, "lpips")}</strong><small>lower is better</small></div>
      </section>

      <IterationMetricMatrix
        metricSets={activeIteration?.metric_sets ?? null}
        iteration={iteration}
        referenceIteration={iterationPayload?.reference_iteration ?? "00000000"}
      />

      <section className="browser-grid">
        <article className="panel scene-panel">
          <div className="panel-heading">
            <div><span className="eyebrow">01 · SCENES</span><h2>Rank evaluation scenes</h2></div>
            <span className="count-badge">{scenesQuery.data?.total ?? 0}</span>
          </div>
          <div className="panel-tools">
            <label className="search-field"><Search size={15} /><input value={sceneSearch} onChange={(event) => setSceneSearch(event.target.value)} placeholder="Find scene…" /></label>
            <div className="presets">
              <button onClick={() => setSceneSorting([{ id: "gain.psnr", desc: true }])}>Most improved</button>
              <button onClick={() => setSceneSorting([{ id: "metrics.psnr", desc: false }])}>Worst quality</button>
            </div>
          </div>
          {scenesQuery.error ? <ErrorState error={scenesQuery.error} /> : (
            <MetricTable<SceneRow>
              data={scenes}
              kind="scene"
              selected={scene}
              onSelect={(row) => { setScene(row.scene_name); setView(null); }}
              sorting={sceneSorting}
              onSortingChange={setSceneSorting}
              iteration={iteration ?? ""}
            />
          )}
        </article>

        <article className="panel view-panel">
          <div className="panel-heading">
            <div><span className="eyebrow">02 · VIEWS</span><h2>{activeScene ? activeScene.scene_name.slice(0, 16) : "Select a scene"}</h2></div>
            <span className="count-badge">{viewsQuery.data?.total ?? 0}</span>
          </div>
          <div className="panel-tools">
            <label className="search-field"><Search size={15} /><input value={viewSearch} onChange={(event) => setViewSearch(event.target.value)} placeholder="Find view…" /></label>
            <div className="presets">
              <button onClick={() => setViewSorting([{ id: "gain.lpips", desc: true }])}>Best gain</button>
              <button onClick={() => setViewSorting([{ id: "metrics.lpips", desc: true }])}>Highest error</button>
            </div>
          </div>
          {viewsQuery.error ? <ErrorState error={viewsQuery.error} /> : (
            <MetricTable<ViewRow>
              data={views}
              kind="view"
              selected={view}
              onSelect={(row) => setView(row.image_name)}
              sorting={viewSorting}
              onSortingChange={setViewSorting}
              iteration={iteration ?? ""}
              scene={scene ?? undefined}
            />
          )}
        </article>
      </section>

      <section className="panel comparison-panel">
        <div className="panel-heading comparison-heading">
          <div><span className="eyebrow">03 · PIXEL LAB</span><h2>{activeView ? `${activeView.image_name} comparison` : "Select a camera view"}</h2></div>
          <div className="availability">
            <span className={activeView?.images.prediction ? "ready" : "missing"}><ImageIcon size={13} />Prediction</span>
            <span className={activeView?.images.input ? "ready" : "missing"}><ImageIcon size={13} />Input</span>
            <span className={activeView?.images.gt ? "ready" : "missing"}><ImageIcon size={13} />GT</span>
          </div>
        </div>
        {activeView && <ViewMetricStrip view={activeView} />}
        {activeView ? <ImageWorkspace urls={imageURLs} pair={pair} onPairChange={setPair} /> : <div className="empty-workspace"><ImageIcon size={26} /><span>Choose a view to start comparing pixels.</span></div>}
      </section>

      <footer>
        <span>Positive Δ always means improvement</span>
        <span>PSNR/SSIM Δ = prediction − input · LPIPS Δ = input − prediction</span>
      </footer>
    </main>
  );
}
