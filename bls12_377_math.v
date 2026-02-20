Require Import ZArith.
Open Scope Z_scope.

Section BLS12_377_Math.
  Variable p : Z.
  Hypothesis Hp : p > 0.

  Definition add_mod (a b : Z) : Z := (a + b) mod p.

  Lemma add_mod_comm : forall a b : Z, add_mod a b = add_mod b a.
  Proof.
    intros a b. unfold add_mod. rewrite Z.add_comm. reflexivity.
  Qed.

  Lemma add_mod_assoc : forall a b c : Z, add_mod (add_mod a b) c = add_mod a (add_mod b c).
  Proof.
    intros a b c. unfold add_mod.
    rewrite Zplus_mod_idemp_l.
    rewrite Zplus_mod_idemp_r.
    rewrite Z.add_assoc.
    reflexivity.
  Qed.

End BLS12_377_Math.
