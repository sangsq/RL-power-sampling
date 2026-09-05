# Paired output analysis

These examples compare low-temperature and power-sampling outputs for the same
question and seed. They are illustrative, not evidence that the sampler always
causes the same local behavior.

## Corrections made by power sampling

### Boundary counting (question 468)

For `2 < sqrt(t) < 3.5`, low-temperature sampling listed `5` through `11` and
answered 7. Power sampling included the valid endpoint integer `12` (because
`12 < 12.25`) and answered 8. Here the changed trajectory repairs an off-by-one
error after both outputs derive the same interval.

### Algebraic identity (question 58)

Low-temperature sampling incorrectly rewrote `99^2 + 99 + 1` as `100^2` and
answered 10,000. Power sampling instead used `(99 + 1)^2 - 99` and obtained
9,901. The difference is substantive reasoning, not formatting.

### Required answer format (question 457)

Both outputs correctly state that `54 mod 6 = 0`, but the low-temperature output
omits `\boxed{}` and is unparseable under the paper protocol. Power sampling
ends with `\boxed{0}`. This gain is format compliance rather than mathematical
reasoning.

## Regressions under power sampling

### Missing final box (questions 433, 262, and 12)

For `57/160`, `10^x - 10 = 9990`, and the amicable-number divisor problem, both
low-temperature and power outputs derive the correct numerical answer. The
low-temperature outputs put it in `\boxed{}`, while the power outputs stop after
an unboxed final sentence. All three are therefore scored as power-sampling
losses. These cases explain why power sampling can improve accuracy while its
overall parse rate is lower.

## Aggregate interpretation

On seed 0, power sampling changes 111 of 500 correctness decisions relative to
low-temperature sampling: 67 gains and 44 losses, a net gain of 23 questions.
Of the gains, 20 convert an unparseable low-temperature response into a correct
boxed answer. Of the losses, 19 are correct low-temperature answers replaced by
unparseable power outputs. The remaining changes reflect different mathematical
trajectories. The sampler therefore shifts both reasoning content and termination
or formatting behavior; accuracy alone cannot distinguish those mechanisms.

Seed 1 confirms the aggregate direction with 60 gains and 36 losses. Across
both seeds there are 127 gains and 80 losses, a net improvement of 47 answers
out of 1,000 paired generations. Forty-five gains start from an unparseable
low-temperature response, while 37 losses end in an unparseable power response.
The examples above remain seed-0 illustrations; the two-seed counts are the
stronger evidence for the population-level description.
