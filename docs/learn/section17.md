# ShopRank lessons: section 17, the demo, the model card and honest limits (built 21 Nov 2026, study after the build)

Files: `src/shoprank/serve/demo.py`, `tests/test_demo.py`, `docs/MODEL_CARD.md`, `README.md`.
Answer the check questions first; the key is at the bottom.

---

## Part 1: A demo is a thin skin over the same model

**Picture:** a shop window. The window shows the same shoes that are sold inside; it does not have its own
special pair. If the window shoes were different, the customer would be misled.

**What we did:** the Gradio demo does not load a model of its own. It calls the same pieces as the API:

```
 browser (Gradio page)
        |  query + "title | brand | colour | bullets" lines
        v
 parse_products()  ->  RankRequest  (the API's own input rules: blank query, too long, no title -> error)
        v
 serving_text()    ->  "title brand colour bullets"  (exactly the training text)
        v
 OnnxScorer (INT8 model_int8.onnx, CPU)  ->  scores  ->  sorted best first, ties by product id
```

So a bug fixed in one place is fixed in both, and the demo cannot drift from the served model
("training/serving skew" has a cousin: "demo/serving skew").

**Real example from his data style:** query `running shoes without laces`, four products. The model put
"Slip-on running shoes, no laces" first (score 1.23) and "Elastic no-tie shoelaces" last (0.21). 9 ms on CPU.
Negation is the weakest slice on dev (0.84 vs 0.87 overall), so this example is a good one to show *and* to
talk about: one right answer is not proof the slice is solved.

| Word | Meaning |
|---|---|
| Gradio | Python library that turns a function into a web page with inputs and outputs |
| `gr.Blocks` | Gradio's layout mode: you place components and wire events yourself |
| `gr.Error` | raising it shows a red message in the page instead of a crash |
| `concurrency_limit=1` | one model call at a time, the same rule the API uses (section 15) |
| `share=False`, `127.0.0.1` | the page is only reachable from this laptop, no public link |

**Interview answer:** "The demo is a thin Gradio layer over the same scorer and the same input validation as
the FastAPI service, so what people see in the demo is exactly the served INT8 model, including its limits."

## Part 2: Test the logic without the UI

**Picture:** a car's engine is tested on a stand before it goes into the car. You do not need the seats and
the paint to know the engine runs.

**What we did:** all the logic lives in two plain functions, `parse_products` and `rank_rows`. Gradio is
imported only inside `build_demo`. The tests (`tests/test_demo.py`) use a fake scorer, so they run in 0.6 s,
need no model file and no gradio, and still check:
- line parsing (`a | b | c | d | e` keeps `d | e` as the bullets),
- the text sent to the model equals the training join,
- best-first order,
- bad input is refused with the API's rules (blank query, no products, no title, query over 200 characters).

Gradio sits in its own dependency group (`uv sync --group demo`), so the API and CI do not install it.

**Interview answer:** "I keep UI code thin and put the logic in pure functions with unit tests and a fake
model; the UI itself I checked by hand in a browser."

## Part 3: A model card and honest limits

**Picture:** the label on a medicine box: what it is for, the dose, who should not take it, side effects.
Nobody trusts a medicine without a label.

**What we wrote (`docs/MODEL_CARD.md`):** what the model does (re-ranks, does not retrieve), how it was
trained (distilled from the DeBERTa ensemble), test nDCG with its interval, speed, intended use, and the
limits: English only, Amazon-style products, weak on negation, noisy labels (75 of 100 audited confident
disagreements looked like label errors), scores are not probabilities, no fairness analysis.

The README carries the same limits under "Honest limits". A limit you state yourself is a sign of care; a
limit the interviewer finds is a sign you did not look.

**Interview answer:** "Every model I ship has a card: intended use, data, test results with intervals, latency,
and known failure modes. For ShopRank the main ones are negation queries and label noise, which caps realistic
nDCG near 0.90."

---

## Check questions

**Q1.** Why does the demo build a `RankRequest` instead of writing its own input checks?
- A. Gradio requires Pydantic models.
- B. So the demo and the API accept and refuse exactly the same input.
- C. It makes the model faster.
- D. To create product ids.

**Q2.** The demo ranks "Slip-on running shoes, no laces" first for "running shoes without laces". What can
you honestly claim?
- A. The negation problem is solved.
- B. The model handles this example; the dev slice (0.84) says negation is still its weakest area.
- C. The demo is better than the test set.
- D. Nothing, demos are never evidence.

**Q3.** Why is `import gradio` inside `build_demo` and not at the top of `demo.py`?
- A. Python does not allow it at the top.
- B. So the tests and CI can import the logic without installing gradio (it lives in the `demo` group).
- C. It makes the page load faster.
- D. Gradio must be imported after the model.

**Q4.** Which of these belongs in a model card's "known limits"? (choose the best)
- A. "Achieves state of the art."
- B. "Scores are not calibrated probabilities; do not threshold them across queries."
- C. The training loss curve.
- D. The GPU model used for training.

**Q5.** `launch(server_name="127.0.0.1", share=False)`. What does it protect against?
- A. Slow scoring.
- B. Anyone else on the network or the internet reaching the demo and the laptop's CPU.
- C. Wrong rankings.
- D. Gradio updates.

## Answer key
1. **B.** One set of rules, two front doors. If they differed, the demo could accept input the API refuses.
2. **B.** One example is an anecdote; the slice nDCG on 158 dev queries is the evidence.
3. **B.** The pure functions import cleanly without the UI package, so `uv sync --locked` + pytest in CI works.
4. **B.** It tells a user how *not* to use the model. A and C are not limits; D is a detail, not a risk.
5. **B.** `127.0.0.1` listens only on this machine; `share=False` creates no public gradio.live link.
