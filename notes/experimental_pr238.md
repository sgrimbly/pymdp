# experimental_pr238 Tracking

- Branch tip: `722196b`
- Created: 2025-09-22

## Workflow Checklist
- [ ] Each session: `git fetch upstream` then review `git rev-list --left-right upstream/v1.0.0_alpha...experimental_pr238`
- [ ] Note any upstream commits to pull or cherry-pick and capture them below.
- [ ] After integrating changes: run targeted tests/notebooks and update the validation log.
- [ ] Push updates when the branch is stable: `git push --set-upstream origin experimental_pr238`

## TODO
- [ ] Re-run automated tests (e.g. `pytest` or focused suites) after the latest merges/cherry-picks. *(last run: pending)*
- [ ] Execute `examples/envs/simplest_demo.ipynb` with the new SimplestEnv to confirm visuals/actions. *(last run: pending)*
- [ ] Monitor upstream PR #238 discussion for review feedback and capture new tasks here.
- [ ] Watch for `pymdp/envs` or `pymdp/agent.py` updates in upstream branches that may need integration.

## Validation Log
- 2025-09-22 – Added select_probs ordering fix (cherry-pick `55fb84c`); tests not yet rerun post-change.

## Integrated Upstream Commits
- 2025-09-22 – Cherry-picked `55fb84c` (select_probs dependency ordering fix).
