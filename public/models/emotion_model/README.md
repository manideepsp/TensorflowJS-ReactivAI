# Emotion Model Artifacts

Place the TensorFlow.js model files here:

- `model.json`
- `group*-shard*.bin`

These files must be served statically. The loader uses `import.meta.env.BASE_URL` so paths resolve correctly under GitHub Pages subpaths.
