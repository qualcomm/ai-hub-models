---
name: review-rereview
description: Single-agent re-review of a model-onboarding PR, run when someone re-requests review from the bot. Rechecks the bot's earlier findings against the new commits, reviews what changed, and posts one review. Works alone with no subagents.
---

# Re-review

The bot already reviewed this PR, and someone re-requested review. Check what changed, and say what's fixed and what's still wrong.

Work alone. Don't launch subagents. Read all of `.agents/skills/model-onboarding-review/SKILL.md` first and follow it, going through the lenses yourself in order. Your prompt gives the PR number and repo.

## 1. Collect

1. Run `gh pr view <n> --json headRefOid,isDraft,state`. If the PR is closed or merged, post nothing.
2. Read the bot's earlier reviews and inline comments, and the replies to them (`gh api --paginate`). Thread state is in `/tmp/pr-threads-state.json`.
3. The newest bot review's `commit_id` is the last reviewed commit. If the bot never reviewed this PR, do a first review as the skill describes and skip the status table.
4. Run `git diff <last reviewed commit>...<head>`. If nothing changed, check the threads for new replies. If there's nothing to act on, post one short comment saying so.

## 2. Recheck earlier findings

Read the code at the new head and mark each earlier finding:
- **Fixed:** the problem is gone. A resolved thread counts only if a real change is behind it.
- **Still open:** the problem remains, or the change doesn't address it.
- **Withdrawn:** the author explained why it's not a problem, and the code or the original source confirms it.
- **Unclear:** say what you'd need.

If an author asked a question, answer it in the summary.

## 3. Review

Run every lens and verify each finding, as the skill describes. Put the most effort into code that changed since the last review. If time runs short, finish the earlier lenses first and list what you skipped. Drop findings that an existing comment already covers, open or resolved.

## 4. Post

Post one review with `gh api repos/<owner>/<repo>/pulls/<n>/reviews --method POST --input /tmp/claude/review.json`. Use the format and the writing rules in `.agents/agents/review-aggregator.md`. Anchor each new finding on a line on the new side of the diff, and put any you can't anchor in the summary.

The summary has a short table of the earlier findings and their status, the number of new findings, and what you didn't verify or read.

Pick the `event`:
- `REQUEST_CHANGES`: the PR isn't a draft and a `high` or `medium` finding is open, either an earlier one or a new one.
- `APPROVE`: the PR isn't a draft, every earlier `high` and `medium` finding is fixed or withdrawn, and no new one exists.
- `COMMENT`: the PR is a draft, or only `low` findings are open.

Report the review id and the number of inline comments.
