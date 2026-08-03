import { describe, expect, it } from "vitest";
import { isCanvasZoomGesture } from "./interactions";

describe("canvas wheel interaction", () => {
  it("leaves an ordinary wheel gesture available for page scrolling", () => {
    expect(isCanvasZoomGesture({ ctrlKey: false, metaKey: false })).toBe(false);
  });

  it("uses control or command wheel gestures for image zoom", () => {
    expect(isCanvasZoomGesture({ ctrlKey: true, metaKey: false })).toBe(true);
    expect(isCanvasZoomGesture({ ctrlKey: false, metaKey: true })).toBe(true);
  });
});
