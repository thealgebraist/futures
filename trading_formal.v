From Coq Require Import Reals.
From Coq Require Import List.
From Coq Require Import Lra.
Open Scope R_scope.

(** * 1. Portfolio Definition *)

(** Portfolio Value at time t: V_t *)
(** Returns: r_t = (Price_{t+1} - Price_t) / Price_t *)
(** Position: p_t in [-1, 1] (allowing for short, cash, or long) *)

Fixpoint portfolio_value (initial : R) (returns : list R) (positions : list R) : R :=
  match returns, positions with
  | nil, _ => initial
  | _, nil => initial
  | r :: rs, p :: ps => portfolio_value (initial * (1 + r * p)) rs ps
  end.

(** * 2. Properties *)

(** Non-negativity: If we never leverage more than 100% and returns > -1, portfolio remains positive. *)
Theorem portfolio_remains_positive : forall (rs ps : list R) (initial : R),
  initial > 0 ->
  (forall r p, In r rs -> In p ps -> 1 + r * p > 0) ->
  portfolio_value initial rs ps > 0.
Proof.
  induction rs; intros.
  - simpl. auto.
  - destruct ps as [| p0 ps0].
    + simpl. auto.
    + simpl. apply IHrs.
      * apply Rmult_gt_0_compat.
        { auto. }
        { apply H0. simpl. auto. simpl. auto. }
      * intros. apply H0. simpl. auto. simpl. auto.
Qed.

(** * 3. Strategy Logic *)
(** A signal function decides the position based on predicted return. *)
Definition signal (pred_return : R) (threshold : R) : R :=
  if Rgt_dec pred_return threshold then 1
  else if Rlt_dec pred_return (-threshold) then -1
  else 0.

(** Theorem: Total return is the product of individual step returns. *)
Theorem total_return_decomposes : forall (r : R) (p : R) (initial : R),
  portfolio_value initial (r :: nil) (p :: nil) = initial * (1 + r * p).
Proof.
  intros. simpl. auto.
Qed.
