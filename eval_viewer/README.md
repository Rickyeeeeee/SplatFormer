# SplatFormer Evaluation Explorer

A read-only web viewer for numeric evaluation iterations produced by `train-sr-2stage.py`.

## Setup

Install the Python server dependencies in the environment used for SplatFormer:

```bash
pip install -r eval_viewer/requirements.txt
```

Install Node.js 20 or newer, then build the frontend once:

```bash
cd eval_viewer/web
npm install
npm run build
cd ../..
```

## Run

```bash
python scripts/run_eval_viewer.py \
  --eval-dir outputs/objaverse_train_sr_2stage_4to1_low_res/eval \
  --host 0.0.0.0 \
  --port 8000
```

Open `http://localhost:8000`. The highest complete numeric iteration is selected automatically. Press **Refresh** to discover evaluation folders created after the server started.

By default, iteration `00000000` supplies the input and GT panels. These are cropped in memory from its `GT | input | prediction` comparison strips and matched to later iterations by scene and image filename. Override that source with `--reference-iteration` if needed.

## Controls

- Use the mouse wheel normally to scroll the page. Hold **Ctrl** (Linux/Windows) or **Command** (macOS) while scrolling over an image to zoom all four canvases together.
- The iteration metric matrix reads `metrics.json` from the selected step and fills input, low-resolution GT, and high-resolution GT baselines from the reference iteration when they are not repeated in the selected step.
- Scene and view tables include prediction, input, low-resolution GT, and high-resolution GT PSNR columns. Use each tab’s **Columns** menu to hide fields or switch to the PSNR-only preset; choices are saved locally.
- Selecting a camera view shows the corresponding four per-view metric files above the image workspace.

## Development

Run the Python server on port 8000, then start Vite in another terminal:

```bash
cd eval_viewer/web
npm run dev
```

Vite serves the UI at `http://localhost:5173` and proxies `/api` to the Python server.

## Checks

```bash
python -m unittest discover -s tests -p 'test_eval_viewer*.py' -v
cd eval_viewer/web
npm test
npm run build
```
