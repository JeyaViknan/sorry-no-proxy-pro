# Enrollment — plugging in the new dataset

Everything in the pipeline is built and tested. This is the one part waiting
on data.

---

## TL;DR — what you do when the photos arrive

```bash
# 1. Drop images here, named {REGNO}_{NN}_{variant}.jpg
#    e.g. gallery/images/25BCE1276_01_frontal.jpg
#    (a bare 25BCE1276.jpg also works — anything before the first
#     underscore is read as the registration number)

# 2. Build the gallery. Every image is quality-gated; failures are
#    reported per student and copied to gallery/rejected/ so you can
#    see exactly what needs retaking.
npm run build:gallery

# 3. Audit it, and calibrate the threshold on real data.
npm run verify:gallery

# 4. Put the recommended thresholds in .env, then restart the server.
```

No code changes. The verifier picks up the new `gallery/face_db.npz` on its
next restart.

---

## Why the current gallery is the problem

`npm run verify:gallery` against the existing data reports:

```
identities: 67   embeddings: 67   images_per_identity: min 1, max 1, mean 1.0

genuine  (same student, different image): (none)
impostor (different students) (n=2211)
  min=-0.135  mean=0.129  p99=0.345  max=0.502

** 1 PAIR(S) COLLIDE AT THE CONFIGURED 0.5 THRESHOLD **
     25BRS1169 <-> 25BRS1286 : 0.5016
```

Two findings, both real:

**1. One image per student.** This is the largest single cause of false
rejections. A single reference vector captures zero intra-class variance, so
glasses (±0.05–0.15 cosine), a new beard (±0.10–0.20), or a lighting change
has nothing to fall back on. It exactly matches the spectacles/facial-hair
failures observed in class.

**2. A live false-accept exists today.** `25BRS1169` and `25BRS1286` score
0.5016 against each other, above the configured `0.50` — on clean enrollment
photographs, before any camera degradation. Those two students can currently
mark each other present.

Interim mitigation until re-enrollment: raise `FACE_THRESHOLD_ACCEPT` to
`0.55` (zero colliding pairs in the current gallery) and accept a higher
false-rejection rate, which the review band absorbs.

---

## Capture protocol

**5 images** per student, **7** for spectacle wearers.

| # | Variation | Purpose |
|---|---|---|
| 1 | Frontal, ~40 cm (arm's length) | Primary. Matches probe geometry, including selfie-lens distortion. |
| 2 | Frontal, ~70 cm | Different perspective profile. |
| 3 | Head turned ~15–20° right | Pose robustness. |
| 4 | Head turned ~15–20° left | Same, opposite side. |
| 5 | Frontal, ~40 cm, different lighting | Illumination robustness. |
| 6\* | Frontal, **glasses off** | The failure mode actually observed. |
| 7\* | Turned ~15–20°, **glasses off** | Glasses-off × pose. |

\* spectacle wearers only.

**Not worth an image slot:** expression (ArcFace is largely
expression-invariant — smiling moves the embedding an order of magnitude less
than glasses do) and hairstyle (`norm_crop` aligns to the five facial
landmarks, so most hair falls outside the 112×112 recognition input).

**Cannot be enrolled prospectively:** facial hair. You cannot ask a
clean-shaven student to grow a beard. Record the state at enrollment and
handle drift via re-enrollment.

### Camera

Capture on **the student's own phone, front camera, at a proctored station**.

This is deliberately not the highest-quality option. It gives *per-student
sensor match* — the exact lens, ISP tuning and noise profile that will produce
every future probe for that student. The existing gallery was shot on a Canon
EOS M50 II at 740×1024 to 1740×2631; the probe path is a phone front camera.
That domain gap is a cause of the false rejections, not a virtue.

Proctoring is non-negotiable: verify the face against a student ID at capture
time. An unproctored enrollment lets someone enroll another face under a
registration number and permanently defeat the system for that student.

| Parameter | Target |
|---|---|
| Stream | ≥ 720×960 portrait, `getUserMedia` path (matches probes) |
| Inter-ocular distance | **≥ 90 px** (ISO/IEC 19794-5 high quality) |
| Framing | Head and shoulders, face ≈ 45–55% of frame height |
| Background | Plain mid-tone grey/beige — **not white** (fools exposure metering) |
| Format | JPEG q0.92 |
| Beauty mode | **Off** — ISP skin smoothing removes the high-frequency texture ArcFace uses |

### Lighting

Soft, diffuse, frontal, from two sides. 500–1000 lux, consistent colour
temperature. Practical version with no equipment: face a large window with
indirect daylight, light wall behind the photographer for bounce fill.

Avoid, in order of how often it ruins a photo:
backlighting (window behind subject) · overhead-only downlight (shadowed eye
sockets, and ArcFace weights the periocular region heavily) · single hard
source · flash (glasses glare) · mixed colour temperature.

---

## Automatic validation

`build_gallery.py` gates every image. Enrollment thresholds are **stricter**
than the probe thresholds in `face_pipeline/config.py` — an enrollment error
is permanent and silent, a verification error is visible and retryable.

| Check | Enrollment | Probe |
|---|---|---|
| `det_score` | ≥ 0.85 | ≥ 0.60 |
| Inter-ocular distance | ≥ 90 px | ≥ 55 px |
| Blur (Laplacian variance, aligned crop) | ≥ 90 | ≥ 45 |
| Brightness | 80–180 | 55–205 |
| Contrast std | ≥ 32 | ≥ 22 |
| Clipped pixels | < 3% | < 8% |
| Yaw / pitch / roll | 30° / 15° / 12° | 28° / 25° / 20° |
| Second face | rejected outright | rejected if > 55% of primary |

Blur is measured on the **aligned 112×112 crop**, not the raw frame —
variance-of-Laplacian scales with resolution, so a raw-image threshold would
mean something different on every phone.

> **Calibrate, don't inherit.** The blur numbers above are starting points.
> Capture ~50 images you judge acceptable, compute their variance, and set the
> enrollment gate at the 10th percentile and the probe gate at the 2nd.

### The two checks that catch what thresholds miss

**Self-consistency.** Each image is compared against the mean of that
student's others. Below 0.55 it is flagged — almost always a bystander's face
selected instead of the student's, a mislabelled file, or the wrong person at
the station. This is the failure that produces "this one student never
verifies, no matter what", and the old pipeline had no way to notice it.

> The old `generate_embeddings.py` took `faces[0]` (arbitrary detector order)
> while verification took the **largest** face. Any enrollment photo with a
> bystander could therefore have stored the wrong person's embedding. Both
> sides now use the same rule.

**Cross-identity uniqueness.** Each student's centroid is compared to every
other. Above 0.45 is flagged as a duplicate enrollment, a sibling, or a twin —
which is how the `25BRS1169`/`25BRS1286` collision above was found.

---

## Threshold calibration

`verify_gallery.py` **refuses to recommend a threshold when there are no
genuine pairs**, and that refusal is deliberate.

Impostor data alone can only tell you how *low* you must not go. It says
nothing about how many legitimate students a given threshold would reject, so
any recommendation derived from it is one-sided — which is precisely the
mistake that produced the uncalibrated `0.50` in the first place.

Once every student has 5–7 images, both distributions exist and the tool
recommends a real operating point:

```
Recommended accept : 0.XX   -> FAR 0.0XXX%   FRR X.XX%
Recommended review : 0.XX
```

**Target FAR ≤ 0.1%.** A false acceptance *is* a successful proxy — the exact
thing this system exists to prevent — and it is undetectable after the fact. A
false rejection is visible, recoverable in seconds by retrying, and absorbed
by the review band. The costs are not symmetric, so the operating point should
not be either.

---

## Future-proofing

| Change | Impact on ArcFace | Strategy |
|---|---|---|
| New / shaved beard | **Large** (0.10–0.20) | Not enrollable in advance. Re-enrollment or template update. |
| Haircut | Small — mostly outside the aligned crop | Ignore |
| Glasses | **Large**, and solvable upfront | Enroll both states (images 6–7) |
| Aging | Modest over a 4-year degree | Annual re-enrollment |
| Masks | **Severe** (0.30–0.50) — ArcFace effectively fails | Needs a mask-aware or periocular model. Keep originals so you can re-embed. |

### Adaptive template update — not yet

Appending verified probes to a student's gallery absorbs gradual drift. It is
**not implemented**, and that is deliberate sequencing:

> Template update on a system with uncontrolled FAR does not merely risk
> poisoning — it *compounds* it. One false accept becomes a permanent gallery
> entry, making the next easier. It must not be enabled before liveness
> detection exists.

When it is added, the safe shape is: append only at similarity ≥ 0.65, cap the
gallery at 8 vectors, **keep the original enrollment vectors immutable and
never evictable** (so recovery is one delete), log every update, and stop
auto-updating a student whose added vectors drift below 0.55 from their
originals — flag them for human re-enrollment instead.

---

## Storage and retention

Raw photographs are **required** long-term, despite the privacy cost: a model
upgrade needs re-embedding, and without originals that means re-enrolling the
entire university.

The mitigation is access control, not deletion:

- `gallery/images/` is **gitignored** and excluded from the Docker build
  context — only the derived `face_db.npz` enters the image.
- Store originals in encrypted private object storage, access-logged, with an
  automated deletion job on the retention date.
- Never inside the web-served directory. (`GET /gallery/images/…` returns 404;
  this is asserted in `scripts/integration-test.mjs`.)

### Still outstanding

The 67 existing photographs remain in **git history**. Removing them from the
working tree does not remove them from history — that needs `git filter-repo`,
which rewrites commit hashes and is disruptive to anyone who has cloned. Your
call on timing, but it should happen before the repository is shared further.

### Metadata

Store per student: `enrollment_date`, `wears_glasses`,
`facial_hair_at_enrollment`, `capture_device`, quality scores, and the
`model` + `embedding_version` the gallery was built with.

**Do not store name, department or section alongside the biometrics.** They
buy nothing for recognition and materially increase the blast radius of a
leak. Join on registration number from the student information system when
needed.
