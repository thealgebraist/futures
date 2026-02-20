(* ALEO BLS12-377 SCALAR FIELD VERIFICATION *)
From Stdlib Require Import ZArith.
From Stdlib Require Import Lia.

Open Scope Z_scope.

Definition r : Z := 2111115437357094910615228024663400359285260170066118903451891842816002572289.

Theorem modular_add_correct : forall u v,
  0 <= u < r ->
  0 <= v < r ->
  (if Z.ltb (u + v) r then (u + v) else (u + v) - r) = (u + v) mod r.
Proof.
  intros u v Hu Hv.
  destruct (Z.ltb (u + v) r) eqn:E.
  - apply Z.ltb_lt in E. rewrite Z.mod_small; lia.
  - apply Z.ltb_ge in E.
    assert (H: u + v = (u + v - r) + 1 * r) by lia.
    rewrite H.
    rewrite Z.mod_add; try lia.
    rewrite Z.mod_small; lia.
Qed.

Theorem modular_sub_correct : forall u v,
  0 <= u < r ->
  0 <= v < r ->
  (if Z.ltb (u - v) 0 then (u - v) + r else (u - v)) = (u - v) mod r.
Proof.
  intros u v Hu Hv.
  destruct (Z.ltb (u - v) 0) eqn:E.
  - apply Z.ltb_lt in E.
    assert (H: u - v = (u - v + r) + (-1) * r) by lia.
    rewrite H.
    rewrite Z.mod_add; try lia.
    rewrite Z.mod_small; lia.
  - apply Z.ltb_ge in E.
    rewrite Z.mod_small; lia.
Qed.
