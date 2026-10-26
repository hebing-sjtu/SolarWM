"""Ref2VA proxy conditioning for Wan2.2 TI2V-5B and I2V-A14B.

The MiniMax-H3 proxy recipe ported to Wan: the anchor picture, an optional OC identity
sheet and one proxy video enter the transformer as clean in-context latent frames ahead of
the target, each on its own RoPE grid and a sequential temporal clock. Nothing is added to
the network; umT5 only reads text. Stage0.5 trains the bidirectional model, Stage1 a W6
AnyFlow causal model and Stage2 a four-step SGF student, mirroring the H3 proxy stages.
"""
