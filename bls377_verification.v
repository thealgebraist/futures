(* BLS12-377 Prime Field Formal Model *)
From Stdlib Require Import ZArith.
Require Import Lia.

Open Scope Z_scope.

Definition P : Z := 8444461749428379642460912098653601437217047122565453817293671142103010304001.

Theorem modular_add_logic : forall a b,
  0 <= a < P ->
  0 <= b < P ->
  (a + b < P -> (a + b) mod P = a + b) /\
  (a + b >= P -> (a + b) mod P = a + b - P).
Proof.
  intros a b Ha Hb.
  split; intros H.
  - rewrite Z.mod_small; lia.
  - assert (a + b < 2 * P) by lia.
    rewrite <- Z.mod_small with (a := a + b - P) (n := P); lia.
Qed.
