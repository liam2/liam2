Status of this branch
=====================

Apart for the issues in the issue tracker, here is a list of things which
need to happen before this branch can be released:

* fix the failing matching.yml test. This is probably related to the use of
  prepare_simple_expr + jit expressions.

* find (and probably fix) the source of difference in results. For example,
  the demo08.yml simulation with a fixed seed of 0, does not result in the
  same number of individuals:
  "16879 individuals on average" in 0.12 (64bit)
  but
  "16858 individuals on average" in 0.13 (64bit)

* fix show("yada", groupby). There is no | anymore, which is okay to me, but it lacks
  a newline at the start to ensure table column names are aligned with the
  data

* fix a few of the worst FIXMEs (e.g. the one in expand_with_defaults)

* use LArray to transform labels to indices, instead of the awful code
  in expr.py:GlobalVariable.evaluate, otherwise the whole point of
  moving to LArray is moot

* massive rebase/cleanup of history

* Ideally, we should also progress in the prepare_simple_expr front. The goal
  was for that to reclaim the performance lost by the transition to LArray, but
  that is not strictly necessary, as it is better to have a usable-but-slow
  version than no version at all. Besides, I *think* that the overhead for
  simulations with more individuals (300K or 2M) should be *much* less
  (possibly even negligible) compared to the overhead for the test simulation
  (9K) or demo simulation (16K). For reference, the current overhead is around
  20% slower for the demo simulations (102K -> 83K individuals/s/period on
  average).

  IIRC, the slowdown for the test "simulation.yml" is much worse, but this is
  an apples-to-oranges comparison given there are more tests now than before.
    - use it in groupby (I started the faster_groupby branch for this, but
      it did not work yet. Might be the same issue as for matching?)
    - use it more generally so that the cost of traversing & variable checking
      and numexpr compiling is only paid once per expression, not once
      per expression*period. This should be done either in the various Process
      subclasses, or (probably better) in Expr.evaluate() itself *and* cache
      the resulting "jitted expr"
    - if we could precompute resulting axes statically (possible in >90% of the
      cases), we could do that in the "prepare"/precompile" step, which would
      almost eliminate LArray overhead. We must still support axes determined
      dynamically, so that will likely make the code more complex
