```bash
          ┌────────────┐
Battery 1 │  Cycles 1–20  ──> X1, y1 (RUL @ 20)
          │  Cycles 2–21  ──> X2, y2 (RUL @ 21)
          │    ...        ──> ...
          └────────────┘

          ┌────────────┐
Battery 2 │  Cycles 1–20  ──> X1, y1
          │  Cycles 2–21  ──> X2, y2
          └────────────┘

          Combine all into:
          X = [X1, X2, ..., Xn]  → shape (n_samples, 20, num_features)
          y = [y1, y2, ..., yn]  → shape (n_samples,)

```
