# ShopRank public page: research

Research stage for one public page plus a live demo. Readers: a recruiter with 30 seconds and an ML hiring
manager with 3 minutes. Compiled 2026-11-21. Every claim has a source link. Where a source is secondary, a
forum post or my own measurement, the text says so. Facts about this repo come from `README.md`,
`src/shoprank/serve/*.py` and file sizes on disk.

---

## 1. Rules for our page

### What the research says (short)

- **People decide in about 10 seconds.** The first 10 seconds of a page visit decide whether people stay. If
  a page holds them for about 30 seconds, they often stay 2 minutes or more. The advice is to state the
  value within 10 seconds. (NN/g, based on Microsoft Research data from 205,873 pages:
  https://www.nngroup.com/articles/how-long-do-users-stay-on-web-pages/)
- **Attention sits at the top.** In 2018, 57% of viewing time was above the fold and 74% was in the first
  two screenfuls. Over 65% of above-the-fold time went to the top half of the screen. (NN/g eyetracking, 120
  participants: https://www.nngroup.com/articles/scrolling-and-attention/)
- **People scan; they don't read.** NN/g describes four scanning patterns: F, spotted, layer-cake and
  commitment. Layer-cake (reading headings, then dipping into one section) is the most effective. The F
  pattern happens when a page has no subheadings or bullets, on desktop and on mobile alike.
  (https://www.nngroup.com/articles/text-scanning-patterns-eyetracking/,
  https://www.nngroup.com/articles/f-shaped-pattern-reading-web-content/)
- **Put the conclusion first (inverted pyramid).** Readers can stop at any point and still have the main
  point. (https://www.nngroup.com/articles/inverted-pyramid/) The original web-reading study found that 79%
  of users scanned and 16% read word by word. Users disliked "marketese" (boastful promotional language) and
  preferred objective, concise text. (1997 study, now archived:
  https://www.nngroup.com/alertbox/9710a.htm and
  https://www.nngroup.com/articles/how-people-read-online/)
- **Recruiters screen fast and by layout.** In the Ladders 2018 eye-tracking update, the first screen of a
  resume took 7.4 seconds on average. Simple layouts with clear sections, headings and bulleted results did
  well. Clutter, multiple columns, long sentences and missing headers did badly. Job titles got the most
  attention. (Coverage of the 2018 study: https://www.hrdive.com/news/eye-tracking-study-shows-recruiters-look-at-resumes-for-7-seconds/541582/.
  The original 2012 report, 30 recruiters over 10 weeks:
  https://www.bu.edu/com/files/2018/10/TheLadders-EyeTracking-StudyC2.pdf)
- **Engineers trust visible work over claims.** In interviews, employers treated cues on GitHub profiles as
  more reliable signals of skill and motivation than a resume, because the work is visible and hard to
  fake. (Marlow & Dabbish, CSCW 2013: https://www.cs.cmu.edu/~xia/resources/Documents/Marlow-cscw13.pdf) A
  2024 interview study with 15 recruiters from Brazil, Canada and the US found that profiles with "icons,
  figures, and generic text" got in the way. Recruiters valued concrete signals such as languages used and
  commit history. (Oliveira, Souza & Figueiredo, IJSEKE 2024:
  https://labsoft-ufmg.github.io/publications/2024/ijseke-johnatan-2024.pdf)
- **ML readers look for a results table, reproduction commands and released weights.** The Papers with Code
  ML Code Completeness Checklist has five items: dependencies, training code, evaluation code, pretrained
  models, and a README with a results table plus the commands that produce it. In NeurIPS 2019 repos,
  pretrained models and results had the largest positive association with GitHub stars.
  (https://medium.com/paperswithcode/ml-code-completeness-checklist-e9127b168501)
- **Report like a model card.** State intended use, evaluation procedure and results, and break results
  down by group. (Mitchell et al., FAT* 2019: https://arxiv.org/abs/1810.03993)
- **Caveat.** I found no credible 2023-2026 survey of what ML hiring managers want from portfolio pages. The
  statistics that circulate (for example "83% trust GitHub more than resumes") trace back to vendor blogs
  with no primary source, so I left them out. The rules below rest on the reading research and the
  smaller-scale hiring studies above.

### Rules (numbered, each with its source)

1. **The headline states the result as a number, in plain words, against a baseline.** For example: "Puts
   the right product first more often than keyword search: nDCG 0.867 vs 0.820 on Amazon's official test
   set." The 10-second decision window and the inverted pyramid both require this.
   ([NN/g dwell time](https://www.nngroup.com/articles/how-long-do-users-stay-on-web-pages/),
   [NN/g inverted pyramid](https://www.nngroup.com/articles/inverted-pyramid/))
2. **The first screen (about 390x844 on mobile, 1366x768 on a laptop) must show five things:**
   - (a) the name and one line on what it does;
   - (b) the headline number from rule 1;
   - (c) one real before/after example: the same query ranked by BM25 and by ShopRank, with the correct
     product's position marked;
   - (d) two buttons, "Try it live" and "Code on GitHub";
   - (e) a role-relevant line of skills, such as "learning-to-rank · distillation · ONNX INT8 · FastAPI".

   The top half of the viewport gets most of the attention, so put (a)-(c) there. Recruiters look longest
   at job titles, which is the reason for (e).
   ([NN/g scrolling](https://www.nngroup.com/articles/scrolling-and-attention/),
   [Ladders 2018](https://www.hrdive.com/news/eye-tracking-study-shows-recruiters-look-at-resumes-for-7-seconds/541582/))
3. **Fit everything a 3-minute reader needs into the first two to three screenfuls.** That space gets 74-81%
   of viewing time. Put the long material (method, ablations, lab notebook) behind links or collapsed
   sections lower down. ([NN/g scrolling](https://www.nngroup.com/articles/scrolling-and-attention/))
4. **Headings alone tell the story (layer-cake).** Each H2 is a claim that starts with its informative
   words, such as "INT8 model on a laptop CPU scores 230 pairs/s, 2.6x the teacher ensemble on a GPU" (README numbers), not "Method". A
   reader who sees only the first two words of each heading still gets the gist.
   ([NN/g F-pattern](https://www.nngroup.com/articles/f-shaped-pattern-reading-web-content/),
   [NN/g scanning patterns](https://www.nngroup.com/articles/text-scanning-patterns-eyetracking/))
5. **Make the key numbers and words stand out (spotted pattern).** Bold the metric values, use descriptive
   link text (not "click here") and use bullets for lists. Keep this sparing so the emphasis still means
   something. ([NN/g scanning patterns](https://www.nngroup.com/articles/text-scanning-patterns-eyetracking/))
6. **Use one column, short sentences, no walls of text, no marketing adjectives.** Aim for about half the
   words of a README. Multiple columns and long sentences hurt resume screening, and web users dislike
   promotional tone. ([Ladders 2018](https://www.hrdive.com/news/eye-tracking-study-shows-recruiters-look-at-resumes-for-7-seconds/541582/),
   [NN/g 1997 study](https://www.nngroup.com/alertbox/9710a.htm))
7. **Avoid decoration.** No badges wall, icon grids or stock imagery. Every visual should carry data.
   Recruiters called icons, figures and generic text noise.
   ([Oliveira et al. 2024](https://labsoft-ufmg.github.io/publications/2024/ijseke-johnatan-2024.pdf))
8. **Every claim links to its evidence.** Results link to the README table, the model links to Hugging Face,
   and "scored once" links to the lab notebook entry. Verifiable traces are what engineers trust.
   ([Marlow & Dabbish 2013](https://www.cs.cmu.edu/~xia/resources/Documents/Marlow-cscw13.pdf))
9. **For the hiring manager, show a compact results table with confidence intervals and the exact dataset
   and split, plus links to the reproduction command and the released weights.** This is the ML Code
   Completeness Checklist applied to a page.
   ([Papers with Code](https://medium.com/paperswithcode/ml-code-completeness-checklist-e9127b168501))
   The workspace rule says the same: never report a metric without naming its dataset or split.
10. **Show limitations and at least one failure case on purpose.** Include a query where ShopRank gets it
    wrong, and state what the model is not for. Model cards ask for intended use and limitations. Google
    PAIR's explorables deliberately show where models are confidently wrong.
    ([Mitchell et al.](https://arxiv.org/abs/1810.03993), [PAIR explorables](https://pair.withgoogle.com/explorables/))
11. **Explain the metric in one sentence with a concrete picture.** For example: "nDCG rewards putting
    exact matches near the top; 1.0 = perfect order." Also say that the scores rank products within one
    query and are not probabilities. PAIR's calibration explorable is the model for that warning.
    ([PAIR, "Are Model Predictions Probabilities?"](https://pair.withgoogle.com/explorables/uncertainty-calibration/))
12. **Preset examples first, free input second.** The demo opens with a pre-filled, real query whose result
    is already visible. Free text is the second step. Every strong demo below does this.
    ([TensorFlow Playground](https://playground.tensorflow.org/),
    [Transformer Explainer](https://poloclub.github.io/transformer-explainer/),
    [Gradio Examples](https://www.gradio.app/docs/gradio/examples))
13. **Loading feedback follows the 0.1 s / 1 s / 10 s limits.** Under 1 s, show nothing special. Over 1 s,
    show a busy state. Over 10 s (a cold server or a 32 MB model download), show a percent-done bar and a
    way to cancel. Announce each state to screen readers as a status message.
    ([NN/g response times](https://www.nngroup.com/articles/response-times-3-important-limits/),
    [WCAG 4.1.3](https://www.w3.org/WAI/WCAG22/Understanding/status-messages.html))
14. **The page must never depend on a live server to make its point.** The first screen uses precomputed
    examples (`site/data/examples.json` from `shoprank.serve.showcase`). The live demo is an extra, because
    free hosting sleeps (section 3).
    ([HF sleep policy](https://huggingface.co/docs/hub/spaces-gpus#sleep-time))

---

## 2. Example pages to learn from

| # | Page | What it does well | What we take |
|---|---|---|---|
| 1 | **Nerfies project page**, https://nerfies.github.io/ (template: https://github.com/eliahuhorwitz/Academic-project-page-template, CC BY-SA 4.0) | The title, a row of Paper / arXiv / Video / Code / Data buttons, then a one-line teaser ("turns selfie videos from your phone into free-viewpoint portraits") and a hero video before the abstract. An interpolation slider lets you scrub results. This became the de facto academic project template. | The button row under the title; a one-line plain-English teaser; results shown before the method. Weakness: no numbers or comparison table, so we must add both. |
| 2 | **Distill, "Attention and Augmented RNNs"** (2016), https://distill.pub/2016/augmented-rnns/ | Builds each mechanism one figure at a time, with clear captions and everyday analogies. "Code" subsections link to implementations. | Progressive disclosure: start with the simple idea, add detail as you go, link to code for experts. |
| 3 | **Google PAIR, "Are Model Predictions Probabilities?"** (2022), https://pair.withgoogle.com/explorables/uncertainty-calibration/ | Explains calibration to non-experts with one running example, interactive thresholds and an explicit warning that a calibrated model can still be useless. | Our "scores are not probabilities" note, plus showing a failure on purpose. |
| 4 | **TensorFlow Playground**, https://playground.tensorflow.org/ | Preset datasets, controls on one screen, live loss feedback, and short explanations below the tool. | Presets first; the result is visible immediately; the explanation sits under the tool, not before it. |
| 5 | **Transformer Explainer** (Georgia Tech + IBM, AAAI-25), https://poloclub.github.io/transformer-explainer/ (paper https://arxiv.org/abs/2408.04619) | Runs GPT-2 live in the browser with ONNX Runtime: no install, no server. A pre-filled example prompt; links to the paper, video and code. | Evidence that a small ONNX model can run client-side on a public page; a pre-filled example. |
| 6 | **Model card with live widget**: `cross-encoder/ms-marco-MiniLM-L6-v2`, https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2 | A usage snippet first, then a comparison table of NDCG@10 / MRR@10 / Docs/sec across sibling models, then an in-page query + documents ranking widget. | Pair quality with speed in one table (our nDCG vs pairs/s, INT8 vs fp32 vs ensemble). Our HF model card already exists, so the page should link to it, not duplicate it. |
| 7 | **FineWeb report as a static Space**, https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1 (README: `sdk: static`, `app_file: dist/index.html`) | A Distill-inspired, long interactive report hosted as a static HF Space, with ablation plots built from the data. | Shows a static Space as a polished, free, always-on report page (section 3). |
| 8 | **Whisper Web**, https://huggingface.co/spaces/Xenova/whisper-web (README https://github.com/xenova/whisper-web) | Speech recognition running in the browser with Transformers.js, hosted on Spaces. | The pattern for an in-browser live demo; it needs model-download progress and a size warning. |
| 9 | **Papers with Code ML Code Completeness Checklist**, https://medium.com/paperswithcode/ml-code-completeness-checklist-e9127b168501 | Not a page design but a reader's checklist: results table + commands + weights. | The checklist our "For engineers" section must pass. |

Notes:
- I found no public Space with a clear before/after re-ranking view. The closest are model-comparison
  Gradio apps and an Elastic blog table
  (https://www.elastic.co/search-labs/blog/elasticsearch-cross-encoder-reranker-huggingface). A clean
  side-by-side BM25 vs ShopRank list with the correct item marked would be distinctive.
- Meta's Segment Anything demo (https://segment-anything.com/) is a well-known product-quality demo, but it
  now redirects to a Meta AI Demos hub and I could not inspect its structure, so it is not in the table.

---

## 3. Hosting

### Platform facts that changed in 2026 (read first)

- **Hugging Face now requires a paid plan to create Gradio or Docker Spaces.** The official docs say:
  "Static Spaces are free for everyone. Gradio and Docker Spaces run on compute and require a paid plan to
  create: PRO for personal accounts". Free personal accounts in good standing (verified email, account older
  than 30 days) can host up to 2 ZeroGPU Gradio Spaces.
  (https://huggingface.co/docs/hub/spaces-overview, https://huggingface.co/docs/hub/spaces-zerogpu)
  PRO costs $9/month and lists "Host ZeroGPU, Gradio & Docker Spaces" (https://huggingface.co/pricing).
  - Users first reported the change on 8-9 July 2026
    (https://discuss.huggingface.co/t/docker-sdk-now-marked-as-paid-when-creating-a-new-space/177580).
  - Existing free Gradio and Docker Spaces seem to keep running, but users report that they cannot update
    them without billing. No staff statement exists either way
    (https://discuss.huggingface.co/t/future-of-free-docker-spaces-quota-for-older-accounts/180780).
- **CPU Basic hardware:** 2 vCPU, 16 GB RAM, 50 GB non-persistent disk, no hourly cost
  (https://huggingface.co/docs/hub/spaces-gpus).
  - It sleeps after 48 hours without visitors ("currently"), and the sleep time cannot be changed on free
    hardware. Anyone visiting restarts it (https://huggingface.co/docs/hub/spaces-gpus#sleep-time).
  - **No official wake-up time is documented.** Forum reports include Spaces stuck in BUILDING or
    APP_STARTING for a long time after sleeping, and some 500 errors on wake
    (https://discuss.huggingface.co/t/sleeping-spaces-are-not-restarting/174003). These are anecdotes, not
    measurements.
- **Custom domains on Spaces are PRO, Team or Enterprise only**
  (https://huggingface.co/docs/hub/spaces-custom-domain).
- **Embedding:** any public Space can be iframed from `https://<subdomain>.hf.space`. Gradio Spaces can
  also be embedded with the `<gradio-app>` web component. It loads lazily and sizes itself, but the script
  version must match the Space's Gradio version
  (https://huggingface.co/docs/hub/spaces-embed, https://www.gradio.app/guides/sharing-your-app).
- **The cross-origin path broke recently.** From 22 June 2026 a Hub-side CORS change broke `<gradio-app>`
  embeds, and users also saw `@gradio/client` failures. The fix shipped in Gradio 6.20.0 on 8 July 2026, so
  owners had to upgrade and redeploy (https://github.com/gradio-app/gradio/issues/13554,
  https://discuss.huggingface.co/t/creating-a-website-chat-widget-with-gradio-part-iv/177208). This is
  real maintenance risk for options A, B and C.
- **`@gradio/client`** connects with `Client.connect(space)` and supports `submit` with status events
  (queue position, ETA). The docs advise pinning a version rather than loading "latest" from the CDN
  (https://www.gradio.app/guides/getting-started-with-the-js-client). A mirrored client README describes a
  `status_callback` that reports sleeping/building/running while a Space wakes. That README is a fork, not
  the official repo, so test this before relying on it.
- **Gradio theming:** custom `css` and `js` are supported. Gradio warns that "query selectors in custom JS
  and CSS [are] not guaranteed to work across Gradio versions"; use `elem_id` and `elem_classes`
  (https://www.gradio.app/guides/custom-CSS-and-JS). So design control is partial and can break on upgrade.
- **Gradio accessibility:** the long-open issue asking for ARIA landmarks (opened October 2023) was closed
  as completed in January 2026 (https://github.com/gradio-app/gradio/issues/5771). I found no published
  WCAG conformance statement for Gradio.
- **Gradio page weight:** my measurement on 2026-11-21. The HTML of the public `gradio/hello_world` Space
  (`https://gradio-hello-world.hf.space/`) references **65 CSS files and 3 JS entry files**, including
  stylesheets for audio, video and image components that the app does not use. PageSpeed Insights hit its
  quota, so I have no Lighthouse score. Run it during the build stage before quoting a number.
- **GitHub Pages:**
  - 1 GB site limit, 100 GB/month soft bandwidth limit, custom domains supported, not for commercial SaaS
    (https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits,
    https://docs.github.com/en/pages/configuring-a-custom-domain-for-your-github-pages-site/about-custom-domains-and-github-pages).
  - **No custom response headers**, so no COOP/COEP headers for multi-threaded WebAssembly. A service-worker
    workaround exists (https://github.com/orgs/community/discussions/13309,
    https://github.com/gzuidhof/coi-serviceworker).
- **Vercel Hobby:** free, non-commercial personal use only, 100 GB fast data transfer per month, 1M CDN
  requests (https://vercel.com/docs/plans/hobby, https://vercel.com/docs/limits/fair-use-guidelines).
  Custom headers can be set, so COOP/COEP is possible.
- **Render free web service** (the workspace already uses Render):
  - It sleeps after 15 minutes without traffic, and waking takes "about one minute".
  - 750 free instance hours per month; not intended for production (https://render.com/docs/free).
- **In-browser inference:** ONNX Runtime Web runs ONNX models on WebAssembly (CPU) or WebGPU
  (https://onnxruntime.ai/docs/tutorials/web/).
  - It is multi-threaded only when the page is cross-origin isolated; otherwise it falls back to one thread
    (https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html).
  - Transformers.js already ships ONNX MiniLM MS MARCO cross-encoders; the quantized L-6 file is about
    23 MB (https://huggingface.co/Xenova/ms-marco-MiniLM-L-6-v2).

**Repo facts relevant to hosting:**
- The served model `model_int8.onnx` is **33,848,219 bytes (32.3 MiB)**.
- The spell vocabulary `spell_vocab.json` is **2.66 MB**.
- The demo needs the ONNX model, its tokenizer and the spell corrector (`src/shoprank/serve/spell.py`,
  SymSpell-style, about 140 lines).
- `src/shoprank/serve/showcase.py` already exports precomputed BM25-vs-ShopRank examples to
  `site/data/examples.json` and has a parity check.

Download time of the 32.3 MiB model (my arithmetic: bytes x 8 / bandwidth):

| Connection | Time |
|---|---|
| Lighthouse "Slow 4G" (1.6 Mbps) | about 165 s |
| DevTools "Fast 4G" (about 9 Mbps) | about 30 s |
| 50 Mbps broadband | about 5 s |

### Options compared

- **A.** The Gradio Space is the page: link to it, or iframe the whole app.
- **B.** A static page (GitHub Pages or Vercel) with precomputed examples, plus a live demo that calls a
  Gradio Space through `@gradio/client` (or embeds it as `<gradio-app>` or an iframe).
- **C.** A Docker Space serving a custom static page and the existing FastAPI on one origin.
- **D.** (Added because of the 2026 HF change.) A static page with precomputed examples, plus a live demo
  that runs the same INT8 ONNX model in the visitor's browser (onnxruntime-web, or Transformers.js for the
  tokenizer), with the spell corrector ported to JS.

| | A. Gradio Space only | B. Static page + Space via client | C. Docker Space (static + FastAPI) | D. Static page + in-browser ONNX |
|---|---|---|---|---|
| **First load, 30-s recruiter** | Poor. Heavy Gradio front end (65 CSS refs measured). After 48 h idle the visitor gets a cold start of unknown length. | **Best.** Plain HTML/CSS; the first screen is precomputed and never waits on the Space. | Mixed. Custom page, but served by the same container, so a sleeping Space delays even the static page. | **Best.** Same as B; the model loads only after a click. |
| **Live demo latency** | Warm: about 70 ms p50 on a laptop (README load test). CPU Basic has 2 vCPU, so expect slower; cold: unknown. | Same as A, plus a network round trip; cold start visible in a status line. | Same as A. | No server. Pays a one-time 32 MB download (5 s on broadband, about 165 s on Slow 4G), then local inference; speed on phones not measured. |
| **Reliability** | Sleeps after 48 h; anecdotal stuck wakes; owner must upgrade Gradio when the Hub changes (June 2026). | The page always works; the live part shares A's risks plus CORS (broken 22 June to 8 July 2026 for embeds). | Sleep and wake risks as A; one origin, so no CORS. | Highest. Static files on a CDN; no sleep, CORS or server. Risk moves to browser compatibility (old browsers, memory on low-end phones). |
| **Design control** | Low. Theme plus custom CSS that "may break" between versions. | Full for the page; the live widget is ours if we use the client, Gradio's look if embedded. | Full. | Full. |
| **Accessibility** | Depends on Gradio; landmarks added only in 2026; no conformance statement. | Ours to get right (WCAG 2.2 AA achievable). | Ours. | Ours. |
| **Effort** | Lowest: `demo.py` exists. | Low-medium: static page + about 50 lines of client code + status handling. | Medium-high: Dockerfile (UID 1000), static serving + FastAPI, CORS off, sleep handling. | Medium-high: JS tokenizer + ONNX session, port of the spell corrector, **JS-vs-Python score-parity test**, progress UI. |
| **Cost** | **Requires HF PRO ($9/mo) to create a new CPU Gradio Space**, unless one already exists or a free ZeroGPU slot works for a CPU-only ONNX app (unverified). | GitHub Pages / Vercel free + the Space (as A). | **Requires PRO** (Docker). | **Free** (GitHub Pages, Vercel Hobby or a static HF Space). |
| **Maintenance** | Gradio upgrades; watch the Space's sleep and build status. | Pin `@gradio/client` to the Space's Gradio version; the Hub can break the cross-origin path. | Container and dependency upkeep; Hub changes. | Lowest server upkeep; pin onnxruntime-web; rerun the parity test when the model changes. |

### Recommendation

**Build the page as static HTML on GitHub Pages, as in B and D.** The first screen and the results come
from precomputed data (`showcase.py` already does this). Put the live demo below the fold and load it only
when someone clicks "Try it live". **Run the live demo in the browser (D) by default**, with a Hugging Face
Space as an optional extra link.

Why:
1. The recruiter's 30 seconds never touch a server. Nothing can sleep, cold-start or hit CORS errors on the
   first screen (rules 1, 2, 14).
2. Since July 2026 a new Gradio or Docker Space needs HF PRO. D is free and never sleeps. It also runs the
   exact `model_int8.onnx` file that the API serves, so "same model" stays true if the parity test passes.
3. Full control over design, accessibility and page weight. A Lighthouse mobile score of 90+ is realistic
   for a plain static page; this is a target, not a measurement.
4. The 32 MB download is the cost of D. It is acceptable only because it happens after an explicit click
   ("Downloads 32 MB once"), with a percent-done bar (rule 13). Mobile visitors on slow networks still get
   the precomputed examples.

**Fallback.** If the parity test fails, or in-browser speed is poor on a mid-range laptop, use B instead:
`@gradio/client` with a pinned version, talking to a CPU Space. That needs PRO, or an existing Space. Show
"Waking the demo (can take a minute)" while the status is not running.

**Avoid A and C as the public entry point.** A gives the slowest first impression and the least design
control. C has the same sleep problem and the most upkeep.

If multi-threaded WASM turns out to matter, move from GitHub Pages to Vercel. Vercel can send COOP/COEP
headers; GitHub Pages cannot without a service-worker hack.

---

## 4. Accessibility and performance targets

### WCAG 2.2 AA: what matters on this page

WCAG 2.2 became a W3C Recommendation on 5 October 2023 and removed 4.1.1 Parsing
(https://www.w3.org/WAI/standards-guidelines/wcag/new-in-22/; full text https://www.w3.org/TR/WCAG22/).

| SC | Requirement | How it applies to us |
|---|---|---|
| 1.1.1 Non-text content (A) | Text alternatives | Every chart gets a text summary or data table; decorative images get `alt=""`. |
| 1.3.1 Info and relationships (A) | Structure in markup | Real `<h1>`-`<h3>` order; ranked lists as `<ol>` or `<table>` with `<th>`; labelled form fields. |
| 1.4.1 Use of colour (A) | Colour is not the only cue | The E/S/C/I relevance labels need text or icon plus colour, not colour alone. |
| 1.4.3 Contrast (AA) | 4.5:1 text, 3:1 large text | Includes muted grey captions and both themes. |
| 1.4.4 / 1.4.10 / 1.4.12 (AA) | 200% zoom, reflow at 320 CSS px, text spacing | Results tables must reflow or scroll inside their own box, never the whole page. |
| 1.4.11 Non-text contrast (AA) | 3:1 for UI parts and graphics | Focus rings, input borders, chart lines. |
| 2.1.1 Keyboard (A) | All functions by keyboard | Preset chips, the "Try it live" button, the BM25/ShopRank toggle. |
| 2.4.7 Focus visible (AA) | Visible focus | Never remove outlines without a replacement. |
| **2.4.11 Focus not obscured (AA, new)** | Focused item not hidden | A sticky header must not cover focused elements. |
| **2.5.8 Target size (AA, new)** | At least 24x24 CSS px (or spacing) | Preset chips, links in tables. |
| 3.1.1 Language of page (A) | `lang="en"` | Some ESCI titles are Spanish; mark them with `lang` if shown. |
| 3.3.1 / 3.3.2 (A) | Errors identified; labels | Query box errors ("query is empty") in text, tied to the field. |
| 4.1.2 Name, role, value (A) | Custom controls exposed | Toggles and tabs use native elements or the correct ARIA. |
| **4.1.3 Status messages (AA)** | Announced without moving focus | "Downloading model 40%", "Ranked 6 products in 85 ms", "Demo unavailable" use `role="status"` (https://www.w3.org/WAI/WCAG22/Understanding/status-messages.html). |

Also:
- Respect `prefers-reduced-motion` for any animated re-ordering. Strictly this is AAA (2.3.3), but it is
  cheap to do.
- Automated checks (Lighthouse, axe) find only part of the problems: about 57% of issues by volume in
  Deque's 2021 study of more than 2,000 audits, a vendor study
  (https://www.deque.com/blog/automated-testing-study-identifies-57-percent-of-digital-accessibility-issues).
  So a Lighthouse accessibility score of 100 is necessary but not enough. Add one manual keyboard-only pass
  and one screen-reader pass (NVDA on Windows).

### Performance

**Core Web Vitals**, judged at the 75th percentile of page loads
(https://web.dev/articles/vitals, https://web.dev/articles/lcp, https://web.dev/articles/inp,
https://web.dev/articles/cls):

| Metric | Good | Poor |
|---|---|---|
| LCP | ≤ 2.5 s | > 4.0 s |
| INP | ≤ 200 ms | > 500 ms |
| CLS | ≤ 0.1 | > 0.25 |

A low-traffic portfolio page will probably have no field (CrUX) data. That is my inference, not checked,
so we rely on lab tests.

**Lighthouse performance score** (v10+ weights): TBT 30%, LCP 25%, CLS 25%, FCP 10%, Speed Index 10%.
Bands: 0-49 poor, 50-89 needs improvement, 90-100 good. Scores vary from run to run, so take the median of
several runs
(https://developer.chrome.com/docs/lighthouse/performance/performance-scoring). What moves the score for
us:
- **TBT:** no big JS on load. Keep the ONNX runtime and model out of the initial bundle; load them on click.
- **LCP:** the LCP element should be the headline text, not a hero image; preload or self-host the one web
  font, or use system fonts.
- **CLS:** reserve space for the results table and the demo panel; set `width` and `height` on images.

**Our targets** (proposed, not from a source):
- Lighthouse mobile: Performance ≥ 90, Accessibility 100, Best Practices ≥ 95.
- First-screen transfer under about 150 KB with no model bytes.
- LCP under 2.5 s on Lighthouse's Slow 4G profile.

**Testing on slow 4G:**
- Lighthouse mobile by default *simulates* 150 ms RTT, about 1.6 Mbps down and 750 Kbps up, with a 4x CPU
  slowdown. The `devtools` mode applies live throttling instead (562.5 ms RTT, about 1.47 Mbps down).
  (https://github.com/GoogleChrome/lighthouse/blob/main/docs/throttling.md)
- Check the report's `benchmarkIndex` and adjust `--throttling.cpuSlowdownMultiplier` if your laptop is
  much faster than a mid-tier phone (same source).
- In the Chrome DevTools Network panel, the "Slow 4G" preset is about 562 ms latency and 1.4 Mbps, and
  "Fast 4G" is about 165 ms and 9 Mbps (https://www.debugbear.com/blog/chrome-devtools-network-throttling).
  Custom profiles and CPU calibration are under Settings > Throttling
  (https://developer.chrome.com/docs/devtools/settings/throttling).
- Test the "Try it live" path on Slow 4G separately. Check that the progress bar, cancel and error states
  work, and that the page stays usable during the 32 MB download.

---

## 5. Licence finding (Shopping Queries Dataset, ESCI)

- **Licence.** The dataset repo https://github.com/amazon-science/esci-data states "This project is
  licensed under the Apache-2.0 License".
  - The LICENSE file is the Apache License 2.0
    (https://raw.githubusercontent.com/amazon-science/esci-data/main/LICENSE).
  - The NOTICE file reads "Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved."
    (https://raw.githubusercontent.com/amazon-science/esci-data/main/NOTICE).
  - The repo also asks users to cite the paper (https://arxiv.org/abs/2206.06588).
- **What it allows.** Apache 2.0 section 4 permits reproducing and distributing the Work or derivative
  works in any medium. That covers showing product titles, brands, colours, bullets and queries on a public
  page. The conditions are:
  - include or link a copy of the licence;
  - keep the attribution notices, including the NOTICE text, in a place where notices normally appear
    (a footer "Data" line works);
  - mark modified files as changed.

  Section 6 grants **no trademark rights**. So: no Amazon logo or trade dress, and no wording that suggests
  Amazon endorses the project. Product titles contain third-party brand names; show them as plain text data
  only, with no brand logos.
- **Suggested footer line:** "Data: Shopping Queries Dataset (Reddy et al., 2022), © Amazon.com, Inc. or
  its affiliates, Apache-2.0 [link]. Not affiliated with or endorsed by Amazon."
- **The paper is licensed separately.** The arXiv paper itself is CC BY-NC-SA 4.0
  (https://arxiv.org/abs/2206.06588). Do not copy the paper's figures onto the page; draw our own.
- **Images are not in the dataset.** The product table columns are `product_id, product_title,
  product_description, product_bullet_point, product_brand, product_color, product_locale`, with no image
  field (https://github.com/amazon-science/esci-data). Images appear only in a third-party extension, SQID
  by Crossing Minds. Its dataset card says MIT (https://huggingface.co/datasets/crossingminds/shopping-queries-image-dataset),
  but the images were crawled from Amazon by ASIN, per the TREC 2023 Product Search overview
  (https://arxiv.org/pdf/2311.07861). My reading is that the MIT grant cannot cover Amazon's photos. This
  matches our decision to use no product images.
- **Caveat:** this is a reading of the licence text, not legal advice.

---

## 6. Open questions

1. **Hugging Face plan.** Is the account free or PRO, and does a CPU Gradio Space for ShopRank already
   exist? This decides whether the fallback (B) is available. I did not check: no account lookups.
2. **Is $9/month acceptable** for PRO? It enables Gradio and Docker Spaces and Space custom domains.
3. **In-browser parity.** Do onnxruntime-web plus a JS tokenizer give the same scores as Python on the
   showcase queries (same tokenizer, max length 128, pair truncation)? This needs a parity test like
   `showcase check`.
4. **In-browser speed.** How fast is single-threaded WASM inference for about 16 candidates on a mid-range
   laptop and phone? This decides GitHub Pages vs Vercel (COOP/COEP) and whether D is viable on mobile.
5. **Spell corrector in JS.** Port about 140 lines of Python and ship 2.66 MB of vocabulary (gzip size
   unknown). Could the vocabulary be trimmed, for example to words of 4+ letters, without changing
   corrections on the showcase queries?
6. **ZeroGPU as a free CPU host.** Can a free account's ZeroGPU Gradio Space serve a CPU-only ONNX model
   without `@spaces.GPU`? The docs say ZeroGPU targets PyTorch apps. Unverified.
7. **Space wake time.** If B is used, measure the cold start after 48 h of sleep. No official number exists.
8. **Custom domain.** Is one wanted? GitHub Pages and Vercel give one free; HF Spaces need PRO.
9. **Showcase query review.** Real customer queries and titles go on a public page, so review the chosen
   examples by hand for anything awkward or offensive.
10. **Analytics.** Should the page record visits? If so, choose a cookie-free option to avoid a consent
    banner. Not researched here.
