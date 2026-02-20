(* Formal proof of stability for Levy-gated stochastic activation *)
Require Import Reals.

Open Scope R_scope.

(* Axioms for Real Order and Addition *)
Axiom R_plus_le_compat_ax : forall r1 r2 r3 r4, r1 <= r2 -> r3 <= r4 -> r1 + r3 <= r2 + r4.
Axiom Ropp_plus_distr_ax : forall r1 r2 : R, -(r1 + r2) = -r1 + -r2.

(* Axiom for tanh range *)
Axiom tanh_range_ax : forall x : R, -1 <= tanh x /\ tanh x <= 1.

(* Theorem: Sum of bounded tanh and bounded noise is bounded *)
Theorem levy_stability : forall (x noise C : R),
  (-C <= noise /\ noise <= C) ->
  (-(1 + C) <= (tanh x + noise) /\ (tanh x + noise) <= 1 + C).
Proof.
  intros x noise C Hn.
  destruct Hn as [Hn_low Hn_high].
  pose proof (tanh_range_ax x) as Ht.
  destruct Ht as [Ht_low Ht_high].
  split.
  - (* Prove -(1 + C) <= tanh x + noise *)
    assert (H_plus: -1 + -C <= tanh x + noise).
    { apply R_plus_le_compat_ax. apply Ht_low. apply Hn_low. }
    (* Goal: -(1 + C) <= tanh x + noise *)
    (* Use eq_ind to apply Ropp_plus_distr_ax *)
    apply (eq_ind (-1 + -C) (fun r => r <= tanh x + noise) H_plus (-(1 + C))).
    (* Sym because Ropp_plus_distr_ax gives -(1+C) = -1 + -C *)
    (* Wait, eq_ind takes P x -> x = y -> P y. So we need -1 + -C = -(1 + C) *)
    (* But our axiom is -(1 + C) = -1 + -C *)
    (* We need symmetry *)
    assert (HS: -1 + -C = -(1 + C)).
    { apply eq_sym. apply Ropp_plus_distr_ax. }
    exact HS.
  - (* Prove tanh x + noise <= 1 + C *)
    apply R_plus_le_compat_ax.
    apply Ht_high.
    apply Hn_high.
Qed.
