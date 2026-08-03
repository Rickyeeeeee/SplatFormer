import { describe, expect, it } from "vitest";
import { clamp, heatColor, pixelError } from "./imageMath";

describe("image difference math", () => {
  it("clamps amplification values", () => {
    expect(clamp(-2, 0, 1)).toBe(0);
    expect(clamp(0.5, 0, 1)).toBe(0.5);
    expect(clamp(3, 0, 1)).toBe(1);
  });

  it("computes channel and mean absolute errors", () => {
    const a = new Uint8ClampedArray([100, 80, 20, 255]);
    const b = new Uint8ClampedArray([90, 100, 50, 255]);
    expect(pixelError(a, b, 0)).toEqual({ red: 10, green: 20, blue: 30, mean: 20 });
  });

  it("maps normalized errors to stable heat colors", () => {
    expect(heatColor(0)).toEqual([10, 18, 70]);
    expect(heatColor(0.5)).toEqual([56, 190, 155]);
    expect(heatColor(1)).toEqual([210, 44, 54]);
  });
});
