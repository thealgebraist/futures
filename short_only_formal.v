From Coq Require Import Reals.
From Coq Require Import List.
From Coq Require Import Lra.
Open Scope R_scope.

(** * 1. Short-Only Position Definition *)

(** A short-only strategy restricts p_t to [-1, 0]. *)
Definition is_short_only (p : R) : Prop := -1 <= p <= 0.

(** Portfolio Value at time t: V_t *)
Fixpoint portfolio_value (initial : R) (returns : list R) (positions : list R) : R :=
  match returns, positions with
  | nil, _ => initial
  | _, nil => initial
  | r :: rs, p :: ps => portfolio_value (initial * (1 + r * p)) rs ps
  end.

(** * 2. Position Constraint Properties *)

(** Lemma: If p is short-only and returns are positive, value decreases (or stays same). *)
Lemma short_loss_on_rise : forall (initial r p : R),
  initial > 0 -> r > 0 -> is_short_only p ->
  portfolio_value initial (r :: nil) (p :: nil) <= initial.
Proof.
  intros. simpl. unfold is_short_only in H1.
  assert (r * p <= 0).
  { nra. }
  nra.
Qed.

(** Lemma: If p is short-only and returns are negative, value increases (or stays same). *)
Lemma short_gain_on_fall : forall (initial r p : R),
  initial > 0 -> r < 0 -> is_short_only p ->
  portfolio_value initial (r :: nil) (p :: nil) >= initial.
Proof.
  intros. simpl. unfold is_short_only in H1.
  assert (r * p >= 0).
  { nra. }
  nra.
Qed.

(** * 3. Strategy Logic *)
(** A short-only signal function. *)
Definition short_signal (pred_return : R) (threshold : R) : R :=
  if Rlt_dec pred_return (-threshold) then -1
  else 0.

Theorem short_signal_is_short_only : forall pred_return threshold,
  threshold >= 0 -> is_short_only (short_signal pred_return threshold).
Proof.
  intros. unfold short_signal, is_short_only.
  destruct (Rlt_dec pred_return (- threshold)).
  - lra.
  - lra.
Qed.
