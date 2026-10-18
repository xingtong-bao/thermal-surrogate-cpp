# S8.4 — A design-exploration loop: does the surrogate make the same decision as the solver?

Setup (`tools/exploration_demo.py`, results `exploration_demo.json/.png`):
one seen chip (test sample 12000), the cooling knob r (convective
resistance, with h tied to it as in every sample of the dataset), 12 values
across the training range, question "what is the weakest cooling that keeps
Tmax ≤ 360 K?". The surrogate answers the sweep in 0.43 s; HotSpot needs
32.5 s (~75×).

| | slope dTmax/d(ln r) | chosen r (largest meeting the limit) |
|---|---|---|
| surrogate | 32.2 K | 0.0064 K/W |
| HotSpot, raw | 16.2 K | 0.028 K/W |
| HotSpot, offset-corrected at the dataset's design point | 16.2 K | 0.0039 K/W |

Reading. The surrogate's response to cooling is twice the solver's (ratio
1.98), so the two disagree on the design even after HotSpot's package offset
is removed: the surrogate would accept a cooler that the solver says is 40 %
too weak, or reject one that would do. Together with S2b.1/S8.1 (power
response 0.1–0.3 of the truth) this closes the question a design-space
exploration tool must ask: **this surrogate's trusted region is the
configurations it was trained on, not their neighbourhood**. Its value in an
exploration loop is as a screen (fast, right ranking within a family), with
the solver deciding anything that moves a physical knob — and with each
solver answer fed back into the model (S8.3).

Caveat: HotSpot's absolute temperatures differ from the dataset's generator
(S5.3); the slope comparison does not depend on that offset, the raw
decision does.
