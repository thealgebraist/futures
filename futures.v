From Coq Require Import ZArith.
Open Scope Z_scope.

Definition profit (wins losses win_amt loss_amt : Z) : Z :=
  wins * win_amt - losses * loss_amt.

Theorem positive_profit_exists : forall (w l wa la : Z),
  w * wa > l * la ->
  profit w l wa la > 0.
Proof.
  intros w l wa la Hgt.
  apply Z.lt_gt.
  apply Z.lt_0_sub.
  apply Z.gt_lt.
  exact Hgt.
Qed.
