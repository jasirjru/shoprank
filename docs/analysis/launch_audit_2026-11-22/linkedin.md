# LinkedIn launch post plan (22 Nov 2026)

Report only. Nothing was posted. Web searches used only generic topic terms (no names, emails or paths).
Source quality is labelled: **[primary]** = LinkedIn itself or a paper; **[data, vendor]** = a tool vendor's
own analysis of many posts (useful, but they sell LinkedIn tools); **[opinion]** = practitioner advice.
Judgement is marked *(judgement)*.

## 0. Pre-posting findings (ranked)

| # | Rank | Finding | Evidence | Fix |
|---|---|---|---|---|
| 1 | BLOCKER | The 32 MB INT8 model was never scored on the test set. 0.867 belongs to the distilled MiniLM (served recipe); the INT8 copy loses 0.0008 on dev. A post saying "my 32 MB model scores 0.867 on the official test" is slightly false. | README.md:22, README.md:51, README.md:139 | In every post: "the small model scores 0.867 ... compressed to 32 MB with almost no loss (0.001 on the dev set)". The drafts below already do this. |
| 2 | BLOCKER | "Right product at #1" numbers are not final. | Placeholders `[75%]` / `[63%]` in all drafts | Fill from the other audit; check they are on the test split and for the same model as 0.867 (not INT8, not dev). If they are dev numbers, say "on the dev set". |
| 3 | SHOULD | Do not write "beats Amazon's model". 0.854 is the author's re-run of Amazon's recipe; Amazon's own 0.857 is on a different query sample. | README.md:32-39 | Write "my re-run of Amazon's published MiniLM method: 0.854". Drafts do this. |
| 4 | SHOULD | Typo result is on simulated one-edit typos, not real customer typos. | README.md:141-144 | Write "won back 87-90% of the score lost to simulated typos". |
| 5 | SHOULD | The two ready-made rerankers are 568M (BGE-reranker-v2-m3) and 596M (Qwen3-Reranker-0.6B) parameters; served MiniLM is 33M. "600M" for both is loose. | README.md:117-118 | Write "two ready-made rerankers about 17 times bigger (0.57-0.6B parameters)". |
| 6 | SHOULD | The page's examples come from the **dev** split, picked by hand. An image built from them must not be captioned as a test result. | site/data/examples.json (`"split": "dev"`) | Image footer: "Example search from the held-out dev set. Test numbers: 8,956 searches, scored once." |
| 7 | SHOULD | Say "official test split of Amazon's Shopping Queries dataset", not "Amazon's official test", which can read as if Amazon tested it. Do not use Amazon's logo in the image. | *(judgement)*; LinkedIn reduces content that misleads ([LinkedIn, Mar 2026][li-feed]) | Wording used in drafts below. |
| 8 | SHOULD | The page's link-preview description repeats the INT8 issue and says "beats" (this is what shows under the link in the first comment). | src/shoprank/site.py:163 | Old: `A 32 MB re-ranking model that beats keyword search on Amazon's Shopping Queries benchmark: nDCG 0.867 vs 0.820 on the official test set. Real examples, numbers and code.` New: `A small re-ranking model (32 MB on CPU) for Amazon's Shopping Queries dataset: nDCG 0.867 vs 0.820 for keyword search on the official test set, scored once. Real examples, including failures.` (regenerate the page; do not re-score anything). |
| 9 | NICE | The page has no `og:image` / `og:title`, so the first-comment link preview will be a bare card. | site/public/index.html:6 (only `<title>`; no `og:` tags found); src/shoprank/site.py:160-164 | Add after line 163: `<meta property="og:title" content="ShopRank: the right product first">`, `<meta property="og:description" content="...same as description...">`, `<meta property="og:image" content="https://shoprank-jasirjru.vercel.app/og.png">` and ship a 1200x627 `og.png` ([LinkedIn Help: 1200x627, 1.91:1][li-share]). |

## 1. Research findings: how the feed treats posts (2025-2026)

Overall: LinkedIn publishes principles, not weights. Most numbers below come from vendors and disagree with each
other. Treat them as direction, not law.

**What LinkedIn itself says [primary]**
- The feed favours "genuine insight and practical ideas" matched to members' professional interests, and reduces
  engagement bait ("comment to agree"), recycled low-substance posts, comment automation and engagement pods
  ([LinkedIn Newsroom, 12 Mar 2026][li-feed]).
- Avoid: posts that only promote without insight, direct asks for likes/comments/follows, irrelevant hashtags,
  copied content ([LinkedIn blog, content to avoid][li-avoid]; [Social Media Today, Nov 2024][smt-donts]).
- "Hashtags do not impact distribution, so no need to include them" (LinkedIn VP Product Gyanda Sachdeva, via
  [Social Media Today, 21 Dec 2025][smt-2026]). Same source: share your own expertise; AI "is best used to augment
  your expression"; 2-5 posts a week.
- LinkedIn's Laura Lorenzetti: hashtags are "a nice to have, not a need to have"; "put [the most interesting
  part] at the top" (via [Buffer][buffer], secondary).
- Ranking uses long-dwell (time spent) as a signal: [LinkedIn Engineering, 2020][li-dwell] [primary, older];
  2026 paper on the current sequential feed ranker counts "long dwell" events and reports +2.10% time spent
  ([arXiv 2602.12354][arxiv-feed]) [primary].

**External links: body vs first comment** (evidence conflicts)
- Van der Blom *Algorithm Insights 2026* (1.3M posts): one external link in the body cuts median reach by
  18.8% (reported via [Goodman, Substack][goodman]) [data, vendor/consultant]. AuthoredUp says "up to 50%",
  no sample given ([AuthoredUp stats][authoredup-stats]) [vendor].
- The same Substack reports claims that link-comments are also suppressed (no source named), while Buffer says
  links in comments "don't appear to hurt" ([Buffer][buffer]) [vendor]. A Saywhat analysis (~400k posts) found
  multi-link resource posts did well ([Goodman][goodman]).
- Link posts have the lowest engagement rate of all formats: 3.25% (2025) vs 7.0% documents
  ([Socialinsider, 2.45M Page posts][socialinsider]) [data, vendor; company pages, not personal profiles].
- *(judgement)* Keep the body link-free; put links in your own first comment, posted within a minute.

**Post length**
- Hard limit 3,000 characters. Mobile cuts at roughly 140 characters / 2-3 lines; desktop around 210
  ([contentin.io][contentin]; [linkedinpreview][lipreview]) [vendor; check on your own phone].
- AuthoredUp (2025): image posts do best at 800-1,100 characters; "lessons learned" posts at 900-1,100; top 5%
  creators average 780 characters with many short paragraphs ([AuthoredUp UpLogic #4][authoredup-len]) [vendor].
  Other vendors find 1,300-2,500 better ([search summary of AuthoredUp/Ordinal/Taplio][len-search]). Taplio's
  global data (130k posts, Nov 2026) favours >1,500 ([Taplio][taplio]).
- *(judgement)* The brief's 1,300 cap is fine: it matches the image-post range. Make the first 140 characters
  carry the whole hook.

**Format**
- Engagement rate by format, 2025: document 7.0%, multi-image 6.45%, video 6.0%, image 5.3%, text 4.5%,
  link 3.25% ([Socialinsider][socialinsider]) [data, vendor, company pages]. Buffer (1M+ posts): carousels ~3x
  images and video ([Buffer][buffer]). Taplio (130k posts): images get the most impressions
  ([Taplio][taplio]).
- LinkedIn image spec: ratio 3:1 to 4:5, max 5 MB, tallest render 4:5; multi-image layout follows the first
  image ([LinkedIn Help][li-images]) [primary]. 1080x1350 is the common 4:5 size ([Sendible][sendible]).
- *(judgement)* For a first post from a small network, a single clear 4:5 image (or 2 images: win + failure) is
  the best trade-off: easy to read on a phone, quick to make, holds attention. A PDF carousel scores higher in
  vendor data but costs hours and is mostly reported for company pages.

**First hour, comments, editing**
- Replying to comments: ~30% higher engagement (Buffer, 72k posts) ([Buffer][buffer]) [vendor].
- "First 60 minutes" as the test window is widely repeated but has no primary source
  ([beehiiv summary][beehiiv]) [opinion].
- Editing: LinkedIn shows an "Edited" label and does not notify followers; no documented penalty for typo fixes;
  one vendor claims a 42% drop when a link is added after posting, without method
  ([search summary: ConnectSafely, MagicPost, Inc.][edit-search]) [vendor/opinion]. *(judgement)* Fix typos if
  needed; never add or remove links or swap the image after posting.

**Best time for an India audience**
- No rigorous India study exists. SocialPilot (683k posts): Tue-Thu, 10-12 and 13-16, in **EST**
  ([SocialPilot][socialpilot]); Hootsuite (1M posts, 118 countries): Tue/Wed 8-9 local, no India data
  ([Hootsuite, Nov 2025][hootsuite]); India-focused guides converge on Tue-Thu, 9-12 IST
  ([search summary][time-search]) [vendor].
- *(judgement)* Tuesday 24 or Wednesday 25 Nov 2026, **09:30 IST** (Indian recruiters' working morning).
  Today, 22 Nov 2026, is a Sunday: do not post today.

## 2. Research findings: what gets junior ML project posts recruiter attention

- CodePath survey of 200+ engineering leaders (Dec 2025): signals beyond interviews were side projects/portfolios
  38%, internships 35%, public code such as GitHub 34%, degree 23%, school prestige 17%; 38% cut entry-level
  hiring ([Fortune][fortune]) [data, survey].
- LinkedIn (Jan 2025, 8,035 HR pros): 73% say under half of applications meet the criteria; 63% see a skills
  mismatch ([LinkedIn News][li-press]) [primary]. Proof of skill is the differentiator.
- Chip Huyen: for juniors, relevant experience incl. independent projects is the strongest signal; reviewers
  skim fast, so make the strongest signal easy to spot ([ML Interviews Book][huyen]) [opinion, well regarded].
- Practitioner consensus: name the baseline you beat, show failure cases, show trade-offs, make the repo run
  as written; tutorial clones (Titanic, MNIST, chatbot wrappers) hurt ([search summary][portfolio-search];
  [TDS][tds]) [opinion].

**Common mistakes** *(judgement, backed by the LinkedIn guidance above)*: opening with "Thrilled to announce"
(spends the 140-character hook on nothing); jargon before meaning ("nDCG", "cross-encoder" in line 1); walls
of text; numbers without a baseline; "beats Amazon"-style claims; asking for likes/comments/reposts (LinkedIn
names this bait); 5+ hashtags; tagging people who did not take part; AI-sounding filler.

ShopRank already has the things recruiters look for: a baseline (BM25), a published method re-run (0.854), a
single scoring of the test set, a failure shown in public, a deployable artifact, and a live demo.

## 3. Image

*(judgement, based on the format data and LinkedIn image spec above)*

- **Size:** 1080x1350 PNG (4:5), under 5 MB. Large type (min ~36 px at 1080 wide) so it reads on a phone.
- **Content:** before/after for `iphone xs pro` (clearest to any reader; site/data/examples.json, first example):
  - Title: `Search: "iphone xs pro"`
  - Left column `Keyword search (BM25)`: three screen-protector titles, short (e.g. "Screen protector, 2-pack"),
    marked "accessory, not the phone".
  - Right column `ShopRank`: "Apple iPhone XS 64GB (Renewed)", "Apple iPhone XS Max 64GB (Renewed)",
    "Apple iPhone 11 Pro Max 512GB (Renewed)", marked with a tick.
  - Footer line 1: `Official test, 8,956 searches, scored once: 0.867 vs 0.820 (nDCG)`
  - Footer line 2 (small): `Example from the held-out dev set. More examples, including failures: shoprank-jasirjru.vercel.app`
- **For Draft A** use 2 images: image 1 = the `m23 for cars` failure (keyword: two McLaren model cars at #1-2;
  ShopRank: sticker, wheel, then the model cars at #3-4), image 2 = the `iphone xs pro` win. Same size; the
  first image sets the layout ([LinkedIn Help][li-images]).
- No Amazon logo, no face-swap/stock art, no emoji. Alt text: describe both columns in one sentence.
- Easiest way to make it: a 1080x1350 screenshot of the page's example card, cropped, with the footer added.

## 4. Recommended format

Single 4:5 image (two for Draft A), 700-1,300 characters, hook in the first 140 characters, short paragraphs,
no link in the body, links in your own first comment, 0-2 hashtags (they do not affect distribution), one call
to action, posted Tue/Wed 09:30 IST. **Recommended draft: A** *(judgement)*: the failure-led hook is unusual,
honest, and shows the judgement recruiters cannot see on a CV; B is the safer fallback.

## 5. Drafts

Character counts are measured on the text inside each block (limit 1,300). Placeholders: `[75%]`, `[63%]`
(from the other audit), `[N months]`.

### Draft A: story / failure-led (1,238 characters)

```text
My search model gets this one wrong.

Search: "m23 for cars". The shopper wants McLaren M23 model cars. Plain keyword search puts both model cars at #1 and #2. My model puts a car sticker first.

I start with the failure because it is part of the result. ShopRank re-ranks product search results, trained on Amazon's public Shopping Queries dataset. It helps on most searches, but on rare model names it can lose to simple keyword matching.

On the dataset's official test split (8,956 US searches, scored once, at the end):
- Ranking score (nDCG, 0 to 1): 0.867, vs 0.820 for keyword search (BM25)
- Right product at #1: [75%] of searches, vs [63%]
- My re-run of Amazon's published MiniLM method: 0.854

How: I trained 5 large DeBERTa models (0.888), then taught a small model to copy them. Compressed to 32 MB, it ranks 16 products in 69 ms (median) on a laptop CPU, served with FastAPI and Docker.

Two surprises:
- A spell checker built from catalogue words won back 87-90% of the score lost to simulated typos.
- Two ready-made rerankers 17x bigger scored lower on my dev set.

I'm looking for my first ML engineering role. If your team works on search or ranking, I'd be glad to talk. Page, code and model are in the first comment.
```

### Draft B: result-led (1,134 characters)

```text
Search "iphone xs pro" with plain keyword matching and you get screen protectors. ShopRank puts the phone first.

ShopRank is a small model that re-ranks the products a search engine has already found, so the best match moves to the top. I built it on Amazon's public Shopping Queries dataset.

On the dataset's official test split (8,956 US searches, scored once):
- Ranking score (nDCG, 0 to 1): 0.867, vs 0.820 for keyword search (BM25)
- Right product at #1: [75%] of searches, vs [63%]
- My re-run of Amazon's published MiniLM method: 0.854
- A 5-model DeBERTa "teacher": 0.888

The teacher is too slow for a CPU, so I distilled it into a small MiniLM and compressed it to 32 MB (INT8 ONNX) with almost no loss. It ranks 16 products in 69 ms (median) on a laptop CPU, behind FastAPI in Docker.

It is not perfect. Rare model names trip it up, and the page shows one such failure next to the wins. A spell checker built from catalogue words fixes most of the damage from simulated typos.

Try a few searches on the live page (link in the first comment) and judge the ranking yourself. I'm looking for my first ML engineering role.
```

### Draft C: short (607 characters)

```text
Keyword search for "iphone xs pro" shows screen protectors. My model shows the phone.

ShopRank is a 32 MB model that re-ranks product search results on a laptop CPU (69 ms for 16 products). On the official test split of Amazon's Shopping Queries dataset (8,956 searches, scored once), it scores 0.867 vs 0.820 for keyword search, and puts the right product first in [75%] of searches vs [63%].

It also gets some wrong. The live page shows one next to the wins.

I'm looking for my first ML engineering role. If you hire for search or ML, I'd be glad to talk. Page, code and model are in the first comment.
```

## 6. First comment (post it yourself within a minute of publishing)

```text
Links:
Live page (real searches side by side, including a failure): https://shoprank-jasirjru.vercel.app
Code, results and how to run it: https://github.com/jasirjru/shoprank
Model (32 MB, INT8 ONNX): https://huggingface.co/jasirjru/shoprank-reranker-minilm-int8

For the details: the test set was scored once, with the systems chosen in advance, and every number has a 95% confidence interval in the README. The page's examples come from the dev set. Limits: US English only, re-ranking only (no retrieval stage), typos were simulated.
```

## 7. Posting checklist

**Before (Mon 23 Nov)**
- [ ] Fix BLOCKERs 1-2 (fill `[75%]`/`[63%]`, check split and model) and SHOULDs 3-8.
- [ ] Make the image(s): 1080x1350 PNG, under 5 MB, footer says dev example + test numbers. Add alt text.
- [ ] Profile ready for visitors: headline names the role you want ("ML engineer, search and ranking"); add
      ShopRank to Featured (page + GitHub); Open to Work set as you prefer. *(judgement)*
- [ ] Open the live page, GitHub and Hugging Face links in a private window on a phone: all load.
- [ ] Paste the draft into LinkedIn's composer, check where "...see more" falls on your phone: the first
      sentence must make sense alone. Proofread now; edits later show "Edited".

**Posting (Tue 24 or Wed 25 Nov, 09:30 IST)**
- [ ] Post with the image(s). No link in the body. 0-2 hashtags at the end at most (e.g. `#MachineLearning`).
- [ ] Within 1 minute: post the first comment above.
- [ ] Do not tag Amazon or people who did not work on it. Do not ask for likes or reposts.

**First hour**
- [ ] Stay online. Reply to every comment with a real answer (a fact, a number, a question back), not "Thanks!".
- [ ] Message 3-5 people who would genuinely care (classmates, mentors) individually with the link; no pods or
      "engagement groups" (LinkedIn reduces these: [LinkedIn, Mar 2026][li-feed]).
- [ ] Do not edit links, the image, or the text beyond a typo fix.

**After**
- [ ] Reply to later comments for 48 hours. Note impressions and profile views at 24 h and 7 days, to compare
      with the next post.
- [ ] A follow-up post 4-7 days later on one lesson (e.g. the spell checker or distillation), same rules.

## Sources

[li-feed]: https://news.linkedin.com/2026/ImprovingTheFeed
[li-avoid]: https://www.linkedin.com/blog/content-recommendations-avoid
[smt-donts]: https://www.socialmediatoday.com/news/linkedin-shares-notes-posting-dos-donts/733562/
[smt-2026]: https://www.socialmediatoday.com/news/linkedin-shares-tips-on-how-to-boost-your-in-app-performance-in-2026/808448/
[buffer]: https://buffer.com/resources/linkedin-algorithm/
[li-dwell]: https://engineering.linkedin.com/blog/2020/understanding-feed-dwell-time
[arxiv-feed]: https://arxiv.org/html/2602.12354
[goodman]: https://melaniegoodmanlinkedinconsultant.substack.com/p/linkedin-algorithm-2026-reach-topic-authority
[authoredup-stats]: https://authoredup.com/blog/linkedin-post-statistics
[authoredup-len]: https://authoredup.com/newsletter/uplogic-issue-4-finding-the-right-balance
[socialinsider]: https://www.socialinsider.io/blog/linkedin-study/
[contentin]: https://contentin.io/blog/how-to-preview-a-linkedin-post/
[lipreview]: https://linkedinpreview.com/blog/linkedin-hook-preview
[len-search]: https://www.tryordinal.com/blog/linkedin-post-size
[taplio]: https://taplio.com/what-and-when-to-post/India
[li-images]: https://www.linkedin.com/help/linkedin/answer/100983
[li-share]: https://www.linkedin.com/help/linkedin/answer/46687
[sendible]: https://sendible.com/insights/linkedin-post-size
[beehiiv]: https://box.beehiiv.com/p/87-of-linkedin-posts-die-in-the-first-hour
[edit-search]: https://magicpost.in/blog/edit-a-linkedin-post
[socialpilot]: https://www.socialpilot.co/blog/best-time-to-post-on-linkedin
[hootsuite]: https://blog.hootsuite.com/best-time-to-post-on-linkedin/
[time-search]: https://recurpost.com/blog/best-and-worst-times-to-post-on-linkedin/
[fortune]: https://fortune.com/2025/12/18/tech-hiring-slow-but-not-fully-stalled-exclusive-codepath-data-new-secret-to-land-tech-role/
[li-press]: https://news.linkedin.com/2025/talent-campaign-b2b-press-release
[huyen]: https://huyenchip.com/ml-interviews-book/contents/2.1.3-what-signals-companies-look-for-in-candidates.html
[portfolio-search]: https://mentorcruise.com/blog/career-roadmap-how-to-build-an-ai-portfolio-that-gets-you-hired/
[tds]: https://towardsdatascience.com/the-machine-learning-projects-employers-want-to-see/

Further sources read (not quoted above): Inc. on editing (https://www.inc.com/joel-schwartzberg/the-linkedin-advice-youve-been-hearing-might-be-wrong/91341128),
ConnectSafely on editing (https://connectsafely.ai/articles/does-editing-linkedin-post-affect-reach-2026),
Superprofile India timing guide (https://superprofile.bio/guides/best-time-to-post-on-linkedin-india).
