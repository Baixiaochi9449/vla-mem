# pi05_v2_deepseek

`pi05_v2_deepseek` is a minimal memory-augmented PI05 variant.

## What It Adds

- A history branch that reads the most recent keyframes selected by fixed negative time deltas.
- A lightweight two-stage memory encoder:
  - per-image patch-token pooling and projection into the action expert width
  - multi-view fusion followed by temporal fusion across keyframes
- A dynamic LoRA hypernetwork that generates per-sample low-rank updates for the action expert.

## Default Design

- Historical keyframes: `[-12, -4, -1]`
- Max cameras per keyframe: `3`
- Dynamic LoRA rank: `8`
- Dynamic LoRA basis count: `4`
- LoRA targets: the last action expert layer `q_proj` and `v_proj`

## Data Flow

1. The processor splits offline windows into:
   - current frame and current state for the original PI05 prefix prompt
   - history-only state/image windows for the memory branch
2. The policy keeps an online observation deque during inference.
3. The memory encoder converts `K x V` historical images plus aligned state history into a single latent vector.
4. The hypernetwork turns that latent vector into dynamic LoRA parameters.
5. The wrapped action expert projections consume those LoRA parameters during the standard PI05 flow-matching forward.

## Current Scope

- Fixed-lag keyframe selection only.
- Dynamic LoRA is injected only into the action expert, not into the PaliGemma prefix branch.
- The implementation keeps the original PI05 training and sampling loops intact as much as possible.