export function isCanvasZoomGesture(event: Pick<WheelEvent, "ctrlKey" | "metaKey">) {
  return event.ctrlKey || event.metaKey;
}
