# PEP 541 Request — Takeover of `openeeg` on PyPI

**This is a draft.** Review and edit before filing at
<https://github.com/pypi/support/issues/new/choose> using the
**"PyPI Project Name (PEP 541)"** template.

---

## Project on PyPI

[`openeeg`](https://pypi.org/project/openeeg/) (uploaded as `openEEG`,
case-insensitive on PyPI) — version 0.1.1, last and only release
uploaded **2021-07-02**.

## Requester

- **PyPI account:** `vitaldb`
- **Maintainer:** Hyung-Chul Lee (vital@snu.ac.kr), Department of
  Anaesthesiology, Seoul National University Hospital, on behalf of
  the VitalDB consortium (<https://vitaldb.net>).
- **Other PyPI packages we maintain:** [`openecg`](https://pypi.org/project/openecg/)
  (currently v0.4.0, actively maintained — part of the same
  single-modality `open*` biosignal family this request concerns).
- **GitHub:** <https://github.com/vitaldb/openeeg> (active development
  of the proposed replacement).

## Summary of the request

We respectfully request transfer of the `openeeg` project name
to the `vitaldb` PyPI account under PEP 541's "abandoned project"
provision. The existing release is an **empty package** (contains no
Python modules) and has had no activity since its single upload nearly
five years ago.

## Evidence of abandonment

### 1. The 0.1.1 release is an empty package — no Python code shipped

We downloaded the only release from PyPI
(`openEEG-0.1.1.tar.gz`, 1,742 bytes, SHA-256
`bb89dd3f60a8950c756a03b0896ab337e5baec85566dbcf1f5c0c58c2ff47a89`)
and inspected its contents:

```
openEEG-0.1.1/
  LICENSE
  PKG-INFO
  README.md             (2 lines of text)
  setup.cfg
  setup.py              (uses find_packages(), which found 0 packages)
  openEEG.egg-info/
    PKG-INFO, SOURCES.txt, dependency_links.txt, requires.txt,
    top_level.txt       (EMPTY — confirming no top-level module ships)
```

The shipped `top_level.txt` is empty, meaning the wheel installs **no
importable module**. `pip install openeeg` followed by `import openeeg`
fails. No `__init__.py` or any `.py` source file is present in the
archive.

### 2. README is a 2-line placeholder pointing to a nonexistent repository

The full content of `README.md`:

> This is a package for EEG data analysis and deep learning.
> Details of this package is available on GitHub.

No GitHub URL is given. The package's `home_page` field on PyPI is set
to `https://github.com/pypa/sampleproject` — the **Python Packaging
Authority's sample project template**, not a real source repository.
This is consistent with a learning exercise that was published as part
of a packaging tutorial and never followed up on.

### 3. Single release in ≈ 5 years

PyPI shows exactly one release (v0.1.1, 2021-07-02). No subsequent
uploads, yanks, or metadata updates have been made. PEP 541's
guidance on "Project not Maintained" expressly contemplates this
case (no activity for >6 months and no public source repository).

### 4. No public source repository exists

Searches for `openeeg` repositories owned by the listed authors on
GitHub return no maintained source for the PyPI distribution. The
existing GitHub repositories matching "openeeg"
(`Mor-Li/OpenEEG`, `bagabont/OpenEEG`, `TobiasKaiser/openeeg-tools`,
the `openeeg` GitHub org) are unrelated and are not owned by the
authors registered on PyPI.

### 5. Generic metadata, no maintenance signal

`keywords`, `classifiers`, and `requires_python` are all unset.
Description is "package for EEG data analysis and deep learning"
with no technical scope, no API, no examples, no clinical or
research use described.

## Attempts to contact the current maintainers

Per PEP 541, we have attempted to reach the registered maintainers
through public channels prior to filing this request:

- **Jackie Li** — author email `lijiaqi199609@sina.com` per PKG-INFO.
  We have sent a message on 2026-05-31 explaining the situation and
  asking whether the project name can be transferred. *Response: [TODO]*.
- **Seth Zhao** — author email `sethzhao506@berkeley.edu` per
  PKG-INFO. This appears to be a Berkeley undergraduate account; the
  author's current affiliation (per <https://sethzhao506.github.io/>)
  is UCLA. We have also sent a message to `sethzhao506@g.ucla.edu`
  on 2026-05-31 requesting their permission to transfer the name.
  *Response: [TODO]*.

We respectfully request that PyPI staff additionally attempt outreach
via the registered maintainer emails on file, and include any
correspondence received in their evaluation.

## Intended use of the name

`openeeg` will be a maintained, Apache-2.0-licensed Python library
for depth-of-anesthesia processing of raw EEG, distributed by the
[VitalDB consortium](https://vitaldb.net). Initial scope:

- Paper-faithful reimplementation of the **openibis** BIS-mimic
  algorithm (Connor 2022, *Anesthesia & Analgesia* 135(4):855–864).
- Paper-faithful reimplementation of **OpenBSR** (Connor 2024) for
  frequency-domain burst-suppression detection.
- Empirically-tuned QUAZI-style BSR detector for matching the BIS
  Vista's published `SR` track on the VitalDB cohort.
- Validation against the VitalDB BIS subset (≈5,000 surgical cases
  with paired raw EEG and commercial BIS values).

The library is a sibling of our already-published
[`openecg`](https://pypi.org/project/openecg/) and will follow the
same release cadence, license (Apache-2.0), and quality bar
(typed API, smoke tests, reproducible benchmarks). Active
development is at <https://github.com/vitaldb/openeeg>.

## Why takeover, not a renamed distribution?

The `open<modality>` naming pattern (`openvital`, `openecg`,
`openeeg`) is a deliberate family convention for our biosignal
libraries and is already published under that pattern for ECG. A
prefix-fork (e.g. `vitaldb-openeeg` or `pyopeneeg`) would split
discovery for users searching for "openeeg" between an abandoned
empty package and the maintained library, which is worse for users
than either current option.

## Acknowledgements

Thank you for your time reviewing this request. We are happy to
provide any additional documentation or to defer the timeline if PyPI
staff need to complete outreach. If the current maintainers respond
and prefer to retain the name, we will of course withdraw the
request.

---

### Filing checklist

Before submitting, fill in / verify:

- [x] Outreach sent 2026-05-31 (Jackie Li, Seth Zhao).
- [ ] Forward any reply received from Jackie Li or Seth Zhao to PyPI
      staff as a follow-up comment.
- [ ] Confirm `vitaldb` PyPI account is logged in and 2FA enabled
      (PEP 541 transfers require this).
- [ ] Use the official template at
      <https://github.com/pypi/support/issues/new/choose> →
      "PyPI Project Name (PEP 541)".
