Require Import Reals.
Require Import Lra.

Open Scope R_scope.

(* Define Resource as a Real number representing percentage (0.0 to 1.0) *)
Definition Resource := R.

(* Safety condition: Mainline + Filler tasks do not exceed total capacity *)
Definition safe_allocation (main : Resource) (filler : Resource) (total : Resource) : Prop :=
  main + filler <= total.

(* Theorem: If mainline is bounded by 75% and filler by 24%, 
   the allocation is safe for total=100% *)
Theorem utilization_safety_bound : 
  forall (m f t : Resource),
  t = 1 ->
  m <= 0.75 * t ->
  f <= 0.24 * t ->
  safe_allocation m f t.
Proof.
  intros m f t Ht Hm Hf.
  unfold safe_allocation.
  rewrite Ht in *.
  lra.
Qed.

(* Robustness with respect to FFNN prediction error epsilon *)
Theorem safety_with_epsilon :
  forall (m f t epsilon delta : Resource),
  t = 1 ->
  m <= 0.75 * t ->
  f <= (1.0 - 0.75 - delta) * t ->
  delta > epsilon ->
  m + f + epsilon < t.
Proof.
  intros m f t eps del Ht Hm Hf Hd.
  rewrite Ht in *.
  lra.
Qed.
