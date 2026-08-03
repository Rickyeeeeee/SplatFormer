import type { IterationPayload, Page, SceneRow, ViewRow } from "./types";

async function requestJSON<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = (await response.json()) as { detail?: string };
      detail = body.detail ?? detail;
    } catch {
      // Keep the HTTP status when the response is not JSON.
    }
    throw new Error(detail);
  }
  return (await response.json()) as T;
}

export const api = {
  iterations: () => requestJSON<IterationPayload>("/api/iterations"),
  refresh: () => requestJSON<IterationPayload>("/api/refresh", { method: "POST" }),
  scenes: (iteration: string, search: string) =>
    requestJSON<Page<SceneRow>>(
      `/api/iterations/${encodeURIComponent(iteration)}/scenes?page_size=500&sort=scene_name&order=asc&search=${encodeURIComponent(search)}`,
    ),
  views: (iteration: string, scene: string, search: string) =>
    requestJSON<Page<ViewRow>>(
      `/api/iterations/${encodeURIComponent(iteration)}/scenes/${encodeURIComponent(scene)}/views?page_size=500&sort=image_name&order=asc&search=${encodeURIComponent(search)}`,
    ),
};

export function imageURL(iteration: string, scene: string, image: string, kind: "prediction" | "input" | "gt") {
  return `/api/images/${encodeURIComponent(iteration)}/${encodeURIComponent(scene)}/${encodeURIComponent(image)}/${kind}`;
}
