# PEP 541 Outreach — email drafts

Two messages to send before filing the PEP 541 takeover issue. Send
both, wait roughly two weeks, then file at
<https://github.com/pypi/support/issues/new/choose> with the
correspondence noted (no reply / declined / consented).

**Send from:** `vital@snu.ac.kr` (the address registered on your
existing PyPI account `vitaldb`).

The two recipients are the registered authors of the existing
`openEEG 0.1.1` (per the PKG-INFO inside the 2021 tarball):

- **Jackie Li** — `lijiaqi199609@sina.com`
- **Seth Zhao** — `sethzhao506@berkeley.edu` (Berkeley undergrad
  account; he is now at UCLA, where his current address is
  `sethzhao506@g.ucla.edu`). Send to **both** addresses to maximise
  the chance of delivery.

Replace `[YOUR NAME]` and `[DATE]` placeholders before sending.

**Status:** Sent on **2026-05-31** to both Jackie Li
(`lijiaqi199609@sina.com`) and Seth Zhao
(`sethzhao506@berkeley.edu`, `sethzhao506@g.ucla.edu`). 14-day
window to **2026-06-14** before filing PEP 541 absent a reply.

---

## Email 1 — Jackie Li

**To:** lijiaqi199609@sina.com
**Subject:** Reaching out about your `openEEG` PyPI package

Hi Jackie,

I hope this finds you well. My name is Hyung-Chul Lee; I am an
anaesthesiologist at Seoul National University Hospital and one of
the maintainers of the [VitalDB](https://vitaldb.net) open clinical
biosignal database (~6,000 surgical cases, used in 80+ published
studies). We also maintain a small family of open-source Python
libraries for working with VitalDB-style biosignals — for example
[`openecg`](https://pypi.org/project/openecg/) for ECG processing.

I am writing because I noticed you and Seth Zhao published a project
called `openEEG` on PyPI back in July 2021, and we are currently
preparing a sibling library for EEG that we would also like to
publish under the name `openeeg` (matching the `openecg` naming
pattern in our family).

Our project is an open-source reimplementation of the BIS-mimic
algorithms (Connor 2022/2024) for depth of anaesthesia, validated
against the VitalDB BIS cohort. Source is at
<https://github.com/vitaldb/openeeg>.

I noticed that your `openEEG` package has not received an update
since the initial 0.1.1 release and that the source archive contains
metadata only (no Python modules), so I wanted to ask:

1. Are you still actively planning to develop the package?
2. If not, would you be open to letting us take over the PyPI
   name? We are happy to coordinate the transfer in whatever form
   works for you — either directly (you adding `vitaldb` as
   maintainer on the existing release) or through the official
   PyPI PEP 541 process.

Either answer is completely fine; if you would prefer to keep the
name we will of course publish under a different one. I just wanted
to check with you directly first.

Thank you for your time, and apologies for the unsolicited email.

Best regards,
[YOUR NAME]
Department of Anaesthesiology
Seoul National University Hospital
vital@snu.ac.kr

---

## Email 2 — Seth Zhao

**To:** sethzhao506@berkeley.edu, sethzhao506@g.ucla.edu
**Subject:** Reaching out about your `openEEG` PyPI package

Hi Seth,

I hope this finds you well. My name is Hyung-Chul Lee; I am an
anaesthesiologist at Seoul National University Hospital and one of
the maintainers of the [VitalDB](https://vitaldb.net) open clinical
biosignal database (~6,000 surgical cases, used in 80+ published
studies). We also maintain a small family of open-source Python
libraries for working with VitalDB-style biosignals — for example
[`openecg`](https://pypi.org/project/openecg/) for ECG processing.
(Apologies for emailing both your Berkeley and UCLA addresses —
I wasn't sure which is current.)

I am writing because I noticed you and Jackie Li published a project
called `openEEG` on PyPI back in July 2021, and we are currently
preparing a sibling library for EEG that we would also like to
publish under the name `openeeg` (matching the `openecg` naming
pattern in our family).

Our project is an open-source reimplementation of the BIS-mimic
algorithms (Connor 2022/2024) for depth of anaesthesia, validated
against the VitalDB BIS cohort. Source is at
<https://github.com/vitaldb/openeeg>.

I noticed that your `openEEG` package has not received an update
since the initial 0.1.1 release and that the source archive contains
metadata only (no Python modules), so I wanted to ask:

1. Are you still actively planning to develop the package?
2. If not, would you be open to letting us take over the PyPI
   name? We are happy to coordinate the transfer in whatever form
   works for you — either directly (you adding `vitaldb` as
   maintainer on the existing release) or through the official
   PyPI PEP 541 process.

Either answer is completely fine; if you would prefer to keep the
name we will of course publish under a different one. I just
wanted to check with you directly first.

Thank you for your time, and apologies for the unsolicited email.

Best regards,
[YOUR NAME]
Department of Anaesthesiology
Seoul National University Hospital
vital@snu.ac.kr

---

## Notes on tone / strategy

* Both messages are intentionally short, polite, and give the
  current owner an easy "yes" or "no" — the goal is to make a
  reply feel low-effort.
* We explicitly do **not** frame their package as "abandoned" —
  that's an inference for the PyPI staff. To the authors we just
  note "no update since the initial release".
* We acknowledge the "empty package" issue (metadata only, no
  modules) factually but without judgement.
* The offer to add `vitaldb` as a co-maintainer is the easiest
  path for the current owner if they consent — no PEP 541
  formalities needed.
* The email cc / from address is `vital@snu.ac.kr` (the PyPI
  account), which makes the request traceable for PyPI staff if
  PEP 541 is later needed.

## After sending — checklist

- [x] Note the send date in `docs/pep541-openeeg-takeover.md`
      (sent 2026-05-31; 14-day window closes 2026-06-14).
- [ ] Wait 14 days for a reply.
- [ ] If consent received: **prefer official transfer over
      collaborator-add**. See "PyPI permission revocability" below
      — a granted Maintainer/Owner role can be revoked at any time
      by the existing Owners, so the durable path is PEP 541
      with their consent letter attached, which PyPI staff
      processes as a clean transfer that removes the previous
      owners. Reply suggesting that route in the same email
      thread.
- [ ] If no reply or declined: file the PEP 541 issue per
      `docs/pep541-openeeg-takeover.md`.

## PyPI permission revocability — why we don't just take a
collaborator role

PyPI has two collaborator roles per project:

| Role | What they can do |
|---|---|
| **Owner** | Upload releases · add/remove collaborators · delete project · **remove other Owners** |
| **Maintainer** | Upload releases only |

Consequences for our request:

- If the current owners add `vitaldb` as a **Maintainer**, they
  retain Owner role and can revoke at any time.
- If they add `vitaldb` as an **Owner** but stay on themselves,
  either side can remove the other. A future account compromise
  (e.g. their old Berkeley address is taken over) puts the package
  at risk.
- The only durable arrangement is for the previous owners to
  step off the project entirely — either by them voluntarily
  removing themselves after we are added as Owner, or by PyPI
  staff executing a transfer through the PEP 541 process.

PEP 541 explicitly accepts **explicit consent of current
maintainers** as the strongest evidence and processes those
requests quickly. The recommended sequence is therefore:

1. Receive their consent email.
2. Reply: "Thank you. Would you mind if we file the official
   PyPI transfer request (PEP 541) and include this email as
   evidence of your consent? That way PyPI staff handles the
   formal transfer."
3. File the PEP 541 issue per
   `docs/pep541-openeeg-takeover.md`, attaching the consent
   correspondence.
