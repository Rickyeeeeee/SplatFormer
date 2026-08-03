export function clamp(value: number, minimum: number, maximum: number) {
  return Math.max(minimum, Math.min(maximum, value));
}

const HEAT_STOPS: Array<[number, [number, number, number]]> = [
  [0, [10, 18, 70]],
  [0.25, [32, 114, 189]],
  [0.5, [56, 190, 155]],
  [0.75, [246, 200, 68]],
  [1, [210, 44, 54]],
];

export function heatColor(value: number): [number, number, number] {
  const t = clamp(value, 0, 1);
  for (let i = 1; i < HEAT_STOPS.length; i += 1) {
    const [rightAt, rightColor] = HEAT_STOPS[i];
    const [leftAt, leftColor] = HEAT_STOPS[i - 1];
    if (t <= rightAt) {
      const mix = (t - leftAt) / (rightAt - leftAt);
      return leftColor.map((channel, index) => Math.round(channel + (rightColor[index] - channel) * mix)) as [number, number, number];
    }
  }
  return HEAT_STOPS[HEAT_STOPS.length - 1][1];
}

export function pixelError(a: Uint8ClampedArray, b: Uint8ClampedArray, offset: number) {
  const red = Math.abs(a[offset] - b[offset]);
  const green = Math.abs(a[offset + 1] - b[offset + 1]);
  const blue = Math.abs(a[offset + 2] - b[offset + 2]);
  return { red, green, blue, mean: (red + green + blue) / 3 };
}
