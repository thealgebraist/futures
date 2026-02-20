(* Aleo Profitability Formal Model *)
From Coq Require Import Reals.
From Coq Require Import Lra.

Open Scope R_scope.

Parameter P : R. (* Price of Aleo *)
Parameter S : R. (* Staking requirement for proving *)
Parameter APR : R. (* Pure Staking APR *)
Parameter Rnet : R. (* Daily network rewards *)
Parameter Hnet : R. (* Total network hashrate *)

(* Constraints *)
Axiom P_pos : P > 0.
Axiom S_pos : S > 0.
Axiom Rnet_pos : Rnet > 0.
Axiom Hnet_pos : Hnet > 0.
Axiom APR_pos : APR > 0.

Definition OppCost : R := S * P * (APR / 365.0).
Definition Revenue (Hi : R) : R := (Hi / Hnet) * Rnet * P.

(* Break-even condition: Revenue must exceed Opportunity Cost *)
Theorem break_even_hashrate : 
  forall Hi, 
  Hi > (S * APR / 365.0) * (Hnet / Rnet) -> 
  Revenue Hi > OppCost.
Proof.
  intros Hi H.
  unfold Revenue, OppCost.
  
  (* We want to show (Hi / Hnet) * Rnet * P > S * P * (APR / 365.0) *)
  (* Since P > 0, this is equivalent to (Hi / Hnet) * Rnet > S * (APR / 365.0) *)
  (* Since Hnet > 0 and Rnet > 0, this is equivalent to Hi > (S * APR / 365.0) * (Hnet / Rnet) *)
  
  assert (Hnet_gt0: Hnet > 0) by apply Hnet_pos.
  assert (Rnet_gt0: Rnet > 0) by apply Rnet_pos.
  assert (P_gt0: P > 0) by apply P_pos.
  
  lra.
Qed.
