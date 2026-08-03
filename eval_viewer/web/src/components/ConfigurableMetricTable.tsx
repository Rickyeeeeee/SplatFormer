import { useEffect, useMemo, useState } from "react";
import {
  type ColumnDef,
  type SortingState,
  type VisibilityState,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  useReactTable,
} from "@tanstack/react-table";
import { ArrowDown, ArrowUp, ArrowUpDown, Columns3, ImageOff } from "lucide-react";
import { imageURL } from "../api";
import type { Metrics, SceneRow, ViewRow } from "../types";

type Row = SceneRow | ViewRow;
type MetricGroup = "metrics" | "input_metrics" | "gt_low_res_metrics" | "gt_high_res_metrics" | "gain";

type Props<T extends Row> = {
  data: T[];
  kind: "scene" | "view";
  selected: string | null;
  onSelect: (row: T) => void;
  sorting: SortingState;
  onSortingChange: (sorting: SortingState) => void;
  iteration: string;
  scene?: string;
};

const COLUMN_LABELS: Record<string, string> = {
  "metrics.psnr": "Prediction PSNR",
  "input_metrics.psnr": "Input PSNR",
  "gt_low_res_metrics.psnr": "Low-res PSNR",
  "gt_high_res_metrics.psnr": "High-res PSNR",
  "metrics.ssim": "Prediction SSIM",
  "metrics.lpips": "Prediction LPIPS",
  "gain.psnr": "PSNR gain",
  "gain.ssim": "SSIM gain",
  "gain.lpips": "LPIPS gain",
};

function loadVisibility(kind: "scene" | "view"): VisibilityState {
  try {
    const value = localStorage.getItem(`eval-viewer-${kind}-columns`);
    return value ? JSON.parse(value) as VisibilityState : {};
  } catch {
    return {};
  }
}

function metricValue(row: Row, group: MetricGroup, metric: keyof Metrics) {
  return row[group]?.[metric];
}

const numberSort = (a: { original: Row }, b: { original: Row }, columnId: string) => {
  const [group, metric] = columnId.split(".") as [MetricGroup, keyof Metrics];
  return (metricValue(a.original, group, metric) ?? Number.NEGATIVE_INFINITY)
    - (metricValue(b.original, group, metric) ?? Number.NEGATIVE_INFINITY);
};

function metricCell(value: number | null | undefined, metric: keyof Metrics, gain = false) {
  if (value == null) return <span className="metric-missing">—</span>;
  return (
    <span className={gain ? (value >= 0 ? "metric-positive" : "metric-negative") : "metric-value"}>
      {gain && value > 0 ? "+" : ""}{value.toFixed(metric === "psnr" ? 2 : 4)}
    </span>
  );
}

function sortIcon(direction: false | "asc" | "desc") {
  if (direction === "asc") return <ArrowUp size={13} />;
  if (direction === "desc") return <ArrowDown size={13} />;
  return <ArrowUpDown size={13} />;
}

export function MetricTable<T extends Row>({
  data,
  kind,
  selected,
  onSelect,
  sorting,
  onSortingChange,
  iteration,
  scene,
}: Props<T>) {
  const [columnVisibility, setColumnVisibility] = useState<VisibilityState>(() => loadVisibility(kind));

  useEffect(() => {
    try {
      localStorage.setItem(`eval-viewer-${kind}-columns`, JSON.stringify(columnVisibility));
    } catch {
      // The viewer still works when local storage is disabled.
    }
  }, [columnVisibility, kind]);

  const columns = useMemo<ColumnDef<T>[]>(() => {
    const identity: ColumnDef<T> = kind === "scene"
      ? {
          id: "scene_name",
          accessorFn: (row) => (row as SceneRow).scene_name,
          header: "Scene",
          enableHiding: false,
          cell: ({ row }) => {
            const item = row.original as SceneRow;
            return (
              <div className="identity-cell">
                <span className="identity-title">{item.scene_name.slice(0, 12)}</span>
                <span className="identity-subtitle">#{item.scene_idx ?? "?"} · {item.reference_available ? "references ready" : "prediction only"}</span>
              </div>
            );
          },
        }
      : {
          id: "image_name",
          accessorFn: (row) => (row as ViewRow).image_name,
          header: "View",
          enableHiding: false,
          cell: ({ row }) => {
            const item = row.original as ViewRow;
            return (
              <div className="view-identity">
                {item.images.prediction && scene ? (
                  <img src={imageURL(iteration, scene, item.image_name, "prediction")} loading="lazy" alt="" />
                ) : <span className="thumb-empty"><ImageOff size={15} /></span>}
                <span>{item.image_name}</span>
              </div>
            );
          },
        };

    const metricColumn = (
      group: MetricGroup,
      metric: keyof Metrics,
      header: string,
      gain = false,
    ): ColumnDef<T> => ({
      id: `${group}.${metric}`,
      accessorFn: (row) => metricValue(row, group, metric),
      header,
      sortingFn: numberSort,
      sortUndefined: "last",
      cell: ({ row }) => metricCell(metricValue(row.original, group, metric), metric, gain),
    });

    return [
      identity,
      metricColumn("metrics", "psnr", "Pred PSNR"),
      metricColumn("input_metrics", "psnr", "Input PSNR"),
      metricColumn("gt_low_res_metrics", "psnr", "Low-res PSNR"),
      metricColumn("gt_high_res_metrics", "psnr", "High-res PSNR"),
      metricColumn("metrics", "ssim", "Pred SSIM"),
      metricColumn("metrics", "lpips", "Pred LPIPS"),
      metricColumn("gain", "psnr", "Δ PSNR", true),
      metricColumn("gain", "ssim", "Δ SSIM", true),
      metricColumn("gain", "lpips", "Δ LPIPS", true),
    ];
  }, [iteration, kind, scene]);

  const table = useReactTable({
    data,
    columns,
    state: { sorting, columnVisibility },
    onColumnVisibilityChange: setColumnVisibility,
    onSortingChange: (updater) => {
      const next = typeof updater === "function" ? updater(sorting) : updater;
      onSortingChange(next.slice(0, 1));
    },
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  });

  const configurableColumns = table.getAllLeafColumns().filter((column) => column.getCanHide());
  const visibleCount = configurableColumns.filter((column) => column.getIsVisible()).length;
  const setPreset = (preset: "all" | "psnr") => {
    setColumnVisibility(Object.fromEntries(
      configurableColumns.map((column) => [column.id, preset === "all" || column.id.endsWith(".psnr")]),
    ));
  };

  return (
    <div className="table-region">
      <div className="column-control-row">
        <details className="column-select">
          <summary><Columns3 size={14} />Columns <span>{visibleCount}/{configurableColumns.length}</span></summary>
          <div className="column-menu">
            <div className="column-presets">
              <button type="button" onClick={() => setPreset("all")}>Show all</button>
              <button type="button" onClick={() => setPreset("psnr")}>PSNR only</button>
            </div>
            {configurableColumns.map((column) => (
              <label key={column.id}>
                <input
                  type="checkbox"
                  checked={column.getIsVisible()}
                  onChange={column.getToggleVisibilityHandler()}
                />
                <span>{COLUMN_LABELS[column.id] ?? column.id}</span>
              </label>
            ))}
          </div>
        </details>
      </div>
      <div className="table-shell">
        <table className="metric-table">
          <thead>
            {table.getHeaderGroups().map((group) => (
              <tr key={group.id}>
                {group.headers.map((header) => (
                  <th key={header.id}>
                    <button
                      type="button"
                      className="sort-button"
                      onClick={header.column.getToggleSortingHandler()}
                      disabled={!header.column.getCanSort()}
                    >
                      {flexRender(header.column.columnDef.header, header.getContext())}
                      {header.column.getCanSort() && sortIcon(header.column.getIsSorted())}
                    </button>
                  </th>
                ))}
              </tr>
            ))}
          </thead>
          <tbody>
            {table.getRowModel().rows.map((row) => {
              const key = kind === "scene"
                ? (row.original as SceneRow).scene_name
                : (row.original as ViewRow).image_name;
              return (
                <tr
                  key={row.id}
                  className={selected === key ? "selected" : ""}
                  onClick={() => onSelect(row.original)}
                  tabIndex={0}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" || event.key === " ") onSelect(row.original);
                  }}
                >
                  {row.getVisibleCells().map((cell) => (
                    <td key={cell.id}>{flexRender(cell.column.columnDef.cell, cell.getContext())}</td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
        {!data.length && <div className="empty-table">No matching {kind === "scene" ? "scenes" : "views"}.</div>}
      </div>
    </div>
  );
}
