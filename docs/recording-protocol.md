# Recording protocol — RHpE PoC project

*Version 2.0 — 2026-05-06. Worth reading before recording — it will save us both time.*

## Animal welfare comes first

Welfare > PoC. No exceptions. This is the project's first principle.

- If, during recording, the horse shows signs of pain or strong discomfort — the session stops and the horse goes to a vet.
- The cues used during recording are natural elements of normal day-to-day work with the horse. We do **not** induce fear, **not** provoke pain, and **not** force discomfort.
- If anything in this protocol seems inconsistent with your horse's welfare — don't do it, let me know, and we'll adjust.

## What we're actually trying to do

**Single-behavior classification from RHpE** (Ridden Horse Pain Ethogram, Sue Dyson, 24 behaviors) — currently ear position (ear forward / sideways / back).

This is **not** a pain detector. RHpE requires ≥8 of 24 behaviors together before musculoskeletal-pain inference is appropriate. What we're building here is the foundation — single per-behavior classifiers that, much further down the road, could be combined into an assistive scoring tool. The current MVP = one of the 24 building blocks.

## Who scores what the model sees in a clip

Not you. Your role is to **record**. The ground truth (i.e. "this clip shows ears forward", "this clip shows ears back") is assigned **by a certified RHpE assessor after the fact**. Don't try to predict what the model "should" see in a clip, and don't provoke specific behaviors — that distorts the data and shifts the ethical decisions onto you.

The whole ask is simpler than it sounds: **record a normal, varied training session of the kind you do anyway**.

## Why a separate protocol

The first PoC iteration collected 53 clips with no specific recording rules (mixed venues, phone, YouTube, mixed formats). A sanity check revealed that the model was learning to recognize the recording session (phone in Lesznowola vs YouTube documentary), not the horse's behavior. LOSO drop −34 pp for head_position, −49 pp for tail_movement. Full analysis: [`lessons_learned.md`](lessons_learned.md).

Conclusion: 30 clips collected by intuition ≠ 30 clips that train a model. This document explains what works and why, so our combined time is spent well.

## What works technically (and why)

| dimension | recommendation | why |
| --- | --- | --- |
| Camera | smartphone on a tripod | hand shake constantly changes the background — the model learns the noise instead of the horse |
| Angle | one chosen angle for the whole session; preferred 90° side or ¾ front | a consistent view lets the model compare clips instead of learning angle differences |
| Distance | full silhouette of the horse + ~30% margin | close-ups drop context (posture, tail); too wide loses facial expression |
| **Single subject in frame** | **only one horse visible; no other moving subjects (a second horse in the next stall, an instructor walking, a swinging door, swinging equipment)** | **verified empirically**: a second moving horse or human in the background is mistaken by the model for ear movement. In Read My Ears, one of the recording sources had two horses in the frame — the model failed dramatically (LOSO AUC 0.633 vs ~0.90 on clean sources). See Lesson 10 in `lessons_learned.md`. |
| Resolution | 1080p | reasonable upload size; the model downsamples internally anyway |
| Frame rate | 25–30 fps (smartphone default) | painless default, no need to fiddle |
| Length | ~30 s per clip | shorter loses temporal context; longer mixes multiple behaviors |
| Format | .mp4 or .mov | both work fine |
| Lighting | consistent within a session | mixing arena / sunlight / shade in one session = the model learns lighting again |

## Procedure: 1 session = 5 clips from a normal training session

Goal: capture different moments from a normal, day-to-day training session. The point is not to stage anything special — it's enough to record 5 different moments from what you'd be doing anyway.

| # | typical moment | typical duration |
| --- | --- | --- |
| 1 | A moment before or after the session — the horse standing relaxed | ~30 s |
| 2 | Warm-up — the first minutes of walk | ~30 s |
| 3 | Main part — walk on a loose rein or a light trot | ~30 s |
| 4 | Direction changes, walk-trot-walk transitions | ~30 s |
| 5 | Cool-down — end of the session, long reins | ~30 s |

These are only suggested moments. You can adapt them to your session — the key is **variety of moments**, not this exact list. If your session looks different (groundwork, lunging, hippotherapy), pick 5 sensibly varied moments from what you do.

**While recording:**

- don't provoke specific reactions from the horse (backing up, spooking, stress responses)
- don't stage situations that aren't normally part of your training
- don't change how you work with the horse for the sake of the recording

The more "naturally as always" — the better for the project.

## What won't work (and why it's worth knowing)

- A single 30-minute continuous video — skips the moment-level structure and is harder to analyze for temporal context. Better to record 5 separate ~30 s clips.
- Hand-held camera — the shake changes the background; the model learns the shake, not the horse.
- Mixing angles / lighting / locations within a single session — breaks consistency and effectively turns one session into several.
- Clips from YouTube or other sources — their session leakage is unverifiable; they can't be included in a balanced LOSO.
- Close-ups on the head only — we lose posture and tail; the model loses context.
- **A second horse or a person in the frame** (verified empirically) — any independently moving second subject introduces background motion that the model confuses with ear movement. If the stable has two horses in adjacent stalls visible to the camera, find a different angle, move the camera back, or separate the subjects in time (the second horse led out for the duration of the recording).

## GDPR and intellectual property

In short: riders / owners / anyone in the frame signs a written consent before the session. This is a hard legal requirement, independent of me. Template below.

What happens to the raw clips:

- stored locally on an encrypted drive, not in any public cloud
- used solely for model training and validation
- never published as video in any form
- if a write-up is published, it will contain only embeddings and accuracy metrics — no identifiable footage

Consent can be withdrawn at any time. Contact: <piotr.pawluk@gmail.com>.

## How to send the files

After a session, please send the 5 clips + a short description. A helpful naming scheme:

```
{horse_name}_{session_id}_{moment}_{date}.mp4

e.g. miszka_s01_relaxed_2026-05-12.mp4
     miszka_s01_warmup_2026-05-12.mp4
     miszka_s01_main_2026-05-12.mp4
     miszka_s01_changes_2026-05-12.mp4
     miszka_s01_cooldown_2026-05-12.mp4
```

In the email / form a brief description helps: horse (breed, age, experience, any historical veterinary issues), session (where, when, who rode, what natural cues happened during the session).

How to send: WeTransfer / Google Drive / Dropbox link to me (<piotr.pawluk@gmail.com>), or a USB stick at a meeting. Messenger compresses video and degrades quality — better avoided.

## Consent template (to print / sign)

### English version

```
CONSENT TO USE OF RECORDINGS FOR A RESEARCH PROJECT

I, [first and last name], consent to the recording and use of video
footage featuring me and/or my horse [horse name] for the research
project "horse-pain-poc" run by Piotr Pawluk (piotr.pawluk@gmail.com).

I acknowledge that:
- The recordings will be used solely for AI model training and validation
- Raw recordings will not be published in any form
- I may withdraw consent at any time (by email)
- Scientific publications may include anonymous metrics / embeddings,
  with no possibility of identifying the horse or any person

Date: ______________

Signature: ______________
```

### Polish version (for Polish-speaking participants)

```
ZGODA NA WYKORZYSTANIE NAGRAŃ DO PROJEKTU BADAWCZEGO

Ja, [imię i nazwisko], wyrażam zgodę na nagrywanie i wykorzystanie
nagrań wideo z udziałem mojej osoby i/lub mojego konia [imię konia]
w ramach projektu badawczego "horse-pain-poc" prowadzonego przez
Piotra Pawluk (piotr.pawluk@gmail.com).

Przyjmuję do wiadomości, że:
- Nagrania będą używane wyłącznie do treningu i walidacji modelu AI
- Surowe nagrania nie będą publikowane w żadnej formie
- Mogę wycofać zgodę w dowolnym momencie (kontakt mailowy)
- W publikacjach naukowych mogą pojawić się anonimowe metryki / embeddingi,
  bez możliwości identyfikacji konia ani osoby

Data: ______________

Podpis: ______________
```

## What this looks like from your side

1. Read this document (~5 min). Questions via DM / email / Issues in the repo.
2. Schedule a session (alone, with me, whichever is convenient).
3. Record 5 clips per the guidance above, from a normal training session.
4. Send the files + a short description.

If something here doesn't fit, or you have a better idea — let me know. The protocol is versioned in the repo; I can update it when sensible feedback comes in.

---

*Protocol updates are versioned in the repo. If I change it, I'll let collaborators know before the next session.*
