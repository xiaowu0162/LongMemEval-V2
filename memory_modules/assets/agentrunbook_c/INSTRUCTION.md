# Task Overview

You are acting as a quick memory retrieval module to provide contexts from agent trajectories collected from a customized web environment for a downstream reader to answer questions specific to that environment.  

The question is in `question.json`. You need to aggregate information from the local `trajectories/` directory. Strictly follow the workflow (detailed below) to collect the information and provide to the reader module. Follow the navigation hints and tools (if any) in the workflow and do not attempt to re-verify/re-explore/rebuild maps because this task has latency constraints. 

Be quick and do not over-explore unless necessary. Work inside the current directory and never explore outside of it. 


# Output Requirement

Write your final result to `memory_module_output.json` as valid JSON with this exact schema:

```json
{
  "memory_markdown": "## Support Analysis\n...\n\n## Relevant Procedure and Hint Notes\n...",
  "trajectory_spans": [
    {
      "trajectory_id": "<trajectory id>",
      "start_state_index": 0,
      "end_state_index": 0
    }
  ]
}
```

Requirements:

- `memory_markdown` should contain the two narrative sections only:
  - `## Support Analysis`: a brief plain language description on where the supporting evidence can be found, pointing to the Relevant Procedure and the spans. If the evidence contradicts with the premise of the question, you should clearly say that the question's premise is wrong (and where it is wrong) and thus cannot be answered. This serves as the hint to the downstream reader worker to abstain from answering that question.
  - `## Relevant Procedure and Hint Notes`: relevant task procedures and observations found in the sessions, taken from the procedure_notes/ directory. If procedure_notes/ has not been populated yet, skip this section. 
- `trajectory_spans` must use zero-based inclusive indices.
- Preserve span order by importance.
- If you find no useful evidence, still write valid JSON with minimal `memory_markdown` and `trajectory_spans`.


# Workflow

First, `trajectories/` is already organized in a fixed way. Because this layout is fixed, do not spend time rediscovering the directory tree.

- `trajectories/<trajectory_id>/` contains one full session.
- `trajectories/<trajectory_id>/trajectory.json` is the main file for that session.
- `trajectories/<trajectory_id>/screenshots/` contains the screenshots referenced by that session.
- Inside each `trajectory.json`, the top-level fields are typically:
  - `id`: the trajectory id
  - `goal`: the task goal for that session
  - `start_url`: the initial page
  - `actions`: action trace metadata
  - `outcome`: final result metadata
  - `states`: the ordered state sequence for that trajectory
- Each item in `states` is one step in order and typically includes:
  - `state_index`: the zero-based state number
  - `url`: page URL at that state
  - `text`: the main current state dump axtree
  - `screenshot`: the screenshot for the current state corresponding to the axtree like `screenshots/0007.png`
  - `thoughts`: short reasoning / note text
  - `action`: next action taken after observing the state

Next, you should be aware of what tasks have been done to our knowledge. Luckily I have rendered a summary of each of the existing trajectories in `trajectories/TRAJECTORY_SUMMARY_CONCISE.md` (This one has a quick high-level overview of each trajectory so you can get oriented fast and later you can use `trajectories/TRAJECTORY_SUMMARY_FULL.md` which has the detailed thought/action sequence for shortlist selection and exact verification). 

Finally go ahead to prepare the memory context. 

## Step 1: do a quick triage of the query so that you have an expectation of what to do. 

- First classify the question quickly before opening any trajectory in detail. Do not blindly browse.
- If `question.json` contains an image, inspect that image. For screenshot-grounded questions, do not generalize across loosely similar sessions until you have aligned the screenshot with the matching surface/state.
- For direct lookup questions, find one exact state showing the requested field/value/button/page and prefer a single clean supporting span. For comparison questions, Find the supporting state from one trajectory per side in the comparison. As long as the state contains the support, stop and return do not need to verify further. 
- For procedure questions, stay within one workflow family unless the question explicitly asks for a shared pattern across workflows. Do not import a plausible step from a different task just because it looks analogous.

## Step 2: inspecting and collecting trajectories. 

- Start from `trajectories/TRAJECTORY_SUMMARY_FULL.md` and shortlist only a few likely trajectories using the goal, start URL, action sequence, and final reward. Prefer trajectories on the exact same product/page/workflow family over merely related ones.
- After shortlisting, do not read raw `trajectory.json` unless necessary. Prefer the helper script for quick inspection:
  - `python scripts/inspect_trajectory.py <trajectory_id>` for a compact trajectory summary
  - `python scripts/inspect_trajectory.py <trajectory_id> --state 7` for one exact state
  - `python scripts/inspect_trajectory.py <trajectory_id> --span 6:8` for a short contiguous span
  - `python scripts/inspect_trajectory.py <trajectory_id> --match "Delete Review|Previous"` to find matching states within one candidate trajectory
- Use the helper only on shortlisted trajectories. It is for exact verification, not for broad rediscovery.
  - IMPORTANT: avoid using rg or find over the raw `trajectory.json`. This is extremely time consuming and will give you too much context to consume, defeating the purpose of a fast retrieval.
- If the evidence contradicts with the question, this might mean that the question's premise is wrong. For example, if you see questions referring to nonexistent static/dynamic features or nonexistent steps in the procedure or even nonexistent procedure, say so clearly. You need to switch to the abstention style in the response and hint at the downstream reader. You can still include the contradicting evidence in the retrieval result. 
- If the exact evidence is missing, incomplete, or contradictory, do not extrapolate from numeric progressions, nearby rows, similar buttons, or similar workflows. In those cases, your job is to preserve the contradiction or uncertainty for the downstream reader, not to guess.
- Keep the final evidence package small. You only need one span to prove one point. The span size should not be too large either, usually no more than 3 states in a span would suffice. If you need to mention something like a procedure across a lot of states, It might be better to mention it in the analysis in the response. 

# Final Reminder - Important Rules:

- Move fast and prefer targeted exploration. Your job is to deliver the relevant evidence as fast as possible.
- Put the most important evidence first.
- Avoid redundant trajectories when multiple trajectories support the same important information.
- Reject nearby-but-not-exact matches. Do not replace the asked field, row, tab, header, button, or state with a similar neighbor.
- You may emit any number of spans, but the total number of states across all spans must be at most `20`.
- Count span size inclusively. For example, states `3-5` count as `3` states toward the budget.
- You may write scratch files in the current directory if needed.
- Do not copy screenshots or AXTree blocks into the output JSON.
