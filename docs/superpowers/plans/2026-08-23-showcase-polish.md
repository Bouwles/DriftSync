# DriftSync Showcase Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make DriftSync portfolio-ready through 50 small, pushed improvements covering tests, correctness, docs, media, UX copy, and repo hygiene.

**Architecture:** Preserve the current Python package and Pygame application. Add tests around pure core modules, add lightweight helper modules/scripts for showcase assets and metadata, and update README/docs to make the project impressive and reproducible.

**Tech Stack:** Python 3.11+, PyTorch, NumPy, pandas, scikit-learn, matplotlib, pygame, pytest, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-08-23-showcase-polish-design.md`

## Global Constraints

- Keep changes compatible with Python 3.11+.
- Do not remove the existing Pygame app or ML pipeline.
- Do not depend on unavailable external services.
- Use real project artifacts for README media.
- Keep each improvement small enough to commit and push independently.
- Run fresh verification before claiming completion.

---

## Task List

Implement the 50 improvements in order. For behavior changes, write a focused pytest before production code and verify it fails for the expected reason. For documentation and media-only commits, verify with path checks or script dry-runs. After each item:

```bash
git status --short
git add <changed files>
git commit -m "<type>: <specific improvement>"
git push origin main
```

1. Add pytest configuration.
2. Add task-engine correctness tests.
3. Add stimulus-generation tests.
4. Add preprocessing feature tests.
5. Add target-label horizon tests.
6. Add sequence extraction tests.
7. Add data split tests.
8. Add metric edge-case tests.
9. Fix realtime inference event log duplication.
10. Document realtime log event schema.
11. Align `DataConfig.feature_columns` with the 11-feature pipeline.
12. Add a checkpoint resolver helper.
13. Improve missing-checkpoint error messages.
14. Add tests for checkpoint resolution.
15. Add a command-line smoke test module.
16. Add README title, badges, and summary.
17. Fix README encoding artifacts.
18. Add polished system-flow diagram.
19. Add screenshot placeholders and asset paths.
20. Add generated static screenshots.
21. Add generated working GIF.
22. Add asset-generation script.
23. Add quick-start section.
24. Add reproducible experiment section.
25. Add live inference section.
26. Add results table from checked-in summary.
27. Add architecture section.
28. Add “why it matters” showcase section.
29. Add repository structure section.
30. Add development commands.
31. Add CI workflow.
32. Add dev requirements.
33. Add Makefile shortcuts.
34. Improve launcher copy.
35. Make launcher auto-install opt-in.
36. Add launcher tests for dependency detection.
37. Improve simulator intro copy.
38. Improve simulator outro saved-path display.
39. Improve live mode missing-model UX copy.
40. Improve results empty state copy.
41. Add changelog.
42. Add contributing guide.
43. Add project metadata file.
44. Add release/demo checklist.
45. Remove unused imports.
46. Add generated-artifact policy doc.
47. Add sample realtime log fixture.
48. Add package smoke import test.
49. Run formatting/import sanity checks and update docs if needed.
50. Final verification pass and final showcase README polish.
