# CHANGELOG


## v0.1.0 (2026-04-20)

### Bug Fixes

- **ci**: Use python -m build in semantic-release (no uv in psr container)
  ([`ea40dc9`](https://github.com/NoUnique/robust-lid/commit/ea40dc91257097d2e73f75a5b0ccf3015a7c713a))

### Build System

- Set up PyPI publishing via semantic-release and GitHub Actions
  ([`4a28cbe`](https://github.com/NoUnique/robust-lid/commit/4a28cbe0ede2ffd5a41b03ec877f54ee7886bcf7))

### Features

- Add parallel execution and --low-memory mode with RSS benchmark
  ([`97c1a15`](https://github.com/NoUnique/robust-lid/commit/97c1a15f28a72491d3bd9fb3707bbcb20e4380a2))

- Add predict_batch for batched LID inference
  ([`c7eb947`](https://github.com/NoUnique/robust-lid/commit/c7eb9473af7fb7c7ae2adda0f092f8a5a919d324))

- Add rlid / robust-lid CLI
  ([`82ac16e`](https://github.com/NoUnique/robust-lid/commit/82ac16ea4093f39217c85107d1cfa17a8f8a3035))

- Add weighted LID ensemble with script gating and multi-dataset benchmarks
  ([`bbb5950`](https://github.com/NoUnique/robust-lid/commit/bbb595014c058be0359d57d22cc7d584b20bb9c1))

- Enable fast_mode by default — drop pure-Python backends
  ([`f437acb`](https://github.com/NoUnique/robust-lid/commit/f437acb84728971ca36235d8fbad293aa4fff6f8))

- Initialize robust-lid package with ensemble models
  ([`5a43334`](https://github.com/NoUnique/robust-lid/commit/5a4333480a9092c63868580dd3370c2047aaa54c))

- **cli**: Silence fastText load warnings and add --verbose progress
  ([`0063a9d`](https://github.com/NoUnique/robust-lid/commit/0063a9d8a2cd7bdf268933ecf324009053a8ee2d))
