---
name: Bug report
about: Report a bug you hit running these tools on real hardware
title: "[bug] "
labels: bug
---

**Which tool?**
(e.g. train_cpt.py, expand_model.py, oom_guard.sh)

**What happened?**
A clear description of the bug, including any error output.

**What did you expect?**

**Your setup**
- AMD GPU model + ROCm version:
- PyTorch version:
- transformers version:
- OS:

**Reproduction**
The exact command(s) you ran, and the smallest dataset/config that reproduces it.

**Did the --selftest pass?**
Run `python3 <tool>.py --selftest` and report whether it passes.
