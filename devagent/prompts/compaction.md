You are summarising the working memory of a coding agent so that it can continue the same task with less context. The agent will see only your summary plus the most recent exchanges; everything else is discarded. Anything you leave out is lost.

The agent's goal, verbatim:

<goal>
${goal}
</goal>

The conversation to summarise follows. It contains the agent's reasoning, tool calls, and tool results.

<conversation>
${conversation}
</conversation>

Write the summary in Markdown with exactly these sections, in this order. Be concrete: name files, functions, line regions, commands, and error messages. Quote exact strings when the agent will need them again, for example the current text of a region it intends to edit.

## Goal
Repeat the goal verbatim.

## Established facts
What the agent has confirmed about the codebase that is relevant to the task. Include where the relevant code lives and how it behaves.

## Files examined and changed
For each file touched: what was read, what was changed, and the current state of any region that was edited.

## Approaches tried and why they failed
Every fix or strategy that was attempted and did not work, with the specific reason it failed. This section is the most important one. An agent that forgets a failed approach will try it again immediately.

## Current plan and progress
The agent's plan, which steps are done, and which step it is on.

## Open questions and risks
Anything unresolved, including denials or approvals that were refused.

## Next action
The single next concrete step the agent should take.
