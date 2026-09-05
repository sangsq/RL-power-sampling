# Post-hoc grading sensitivity

Added on 2026-09-04 after inspecting development outputs and before inspecting
any test-set results. This does not change training, checkpoint selection,
generation settings, or the predeclared primary boxed-answer metric.

Development evaluation showed that the final policy largely replaced boxed
answers with `#### <number>` and `The answer is: <number>`. Several manually
inspected responses had correct numeric answers but failed the strict grader.

Apply the same additional deterministic rule to every saved condition:

1. If a boxed answer exists, retain it, including incorrect/non-numeric boxes.
2. Otherwise, use the final `####` marker and its following numeric answer.
3. Otherwise, use the final explicit `answer is`, `answer:`, or `answer =` marker.
4. Parse a signed number, decimal, comma-separated number or numeric fraction
   immediately after that marker (allow punctuation/currency before it).
5. Require agreement with the reference number under the existing 1e-9 tolerance.
   No unmarked last-number guessing, ground-truth-dependent extraction or LLM judging.

Report numeric accuracy, parse coverage and pass@k separately as a post-hoc
sensitivity analysis. These do not establish valid reasoning traces and must
not be presented as the original preregistered outcome.
