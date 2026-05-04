# Open Questions for Thomas / Supervisor

Questions about FDDC algorithm parameters that are inherited from Thomas's implementation
but lack documentation on their origin or rationale.

## 1. Default sigma partition count: why 6?

Thomas's `min_maxV2` initializes `sigma = [1, 1, 1, 1, 1, 1]` -- six partitions of the
experimental distribution. The number 6 is never explained in his thesis or code comments.
Is it empirically chosen? Based on the original Paredis FDDC paper? Or arbitrary?

**Why we need to know:** The sigma partition count directly affects the cost landscape.
A different number of partitions could change which solutions the optimizer considers "good."
If 6 was arbitrary, this should be documented. If it was tuned, the tuning rationale matters
for reproducibility.

**Where it appears:** `program code/distributionComparison.py:312`, our `min_max_v2.py:41`

## 2. Sigma points to distribute: why 20% of sigma length?

Thomas's `fddc.py:79`: `self.points_to_distribute = int(0.2 * len(self.original_sigma))`.
Each pop2 individual starts as all-ones sigma, then 20% of positions get boosted by 4.
The 20% and the boost factor of 4 are never explained.

**Why we need to know:** These values control the initial diversity of the problem population.
If they're from the Paredis paper, we should cite it. If Thomas chose them empirically,
we should document that.

**Where it appears:** `program code/fddc.py:79,29`, our `fddc.py` FDDCConfig defaults

## 3. Memory size: 10 (halved from 20)

Thomas's thesis (line 444): "The original paper chose 20 but this took too much time so
this value was halved." The original paper is Paredis (1995).

**Status:** Documented. Thomas explicitly explains the halving in his thesis. We use 10
as the default, matching Thomas. The Paredis reference should be cited.

## 4. Number of encounters: 10

Same as memory_size. Thomas uses 10 encounters per generation. His thesis doesn't explain
why 10 specifically, only that it matches memory_size when co-evolution is enabled
(`self.number_of_encounters = 10 if self.co_evolution else 1`).

**Why we need to know:** More encounters = more accurate fitness estimates but more compute.
The relationship between encounters and memory_size may matter for convergence properties.

**Where it appears:** `program code/fddc.py:32`, our `fddc.py` FDDCConfig defaults

## 5. Rank selection power: 1.5

Thomas's `fddc.py:62`: `self.probabilities.append(((i + 1) / self.populationSize) ** 1.5)`.
The exponent 1.5 controls selection pressure. Higher = stronger bias toward top-ranked
individuals. 1.5 is never explained.

**Why we need to know:** Selection pressure directly affects exploration vs exploitation
tradeoff. If 1.5 was tuned, the tuning setup matters. If it's from literature, cite it.

**Where it appears:** `program code/fddc.py:62`, our `fddc.py` FDDCConfig defaults

## 6. Transfac (translation factor): 1.0

Thomas's `min_maxV2.__init__`: `self.transfac = transfac` with default 1. This controls
the exponential penalty for peak misalignment: `cost *= exp(percentage / transfac)`.
With transfac=1, even small peak offsets get amplified exponentially.

**Why we need to know:** This is the most sensitive parameter in the cost function.
Changing it from 1.0 to 0.5 would make the penalty much harsher. Was 1.0 tuned or arbitrary?

**Where it appears:** `program code/distributionComparison.py:315`, our `min_max_v2.py:36`
