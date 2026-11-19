# edu-ai-suite

The package computes interpretable image features: each feature in the catalogue turns a
picture into one number, among them `mean_intensity`, `edge_density`, `blue_minus_red_mean`,
`coarse_texture`, `left_right_symmetry` and `centre_of_mass_x`. It also trains a tiny Keras
language model whose `vocab_size`, `context_width`, `embedding_dim` and `hidden_units` the
caller sets and whose rows come from `make_pairs`, and it fine-tunes an existing model with
`train_lora`, reporting `iters_completed` while `split_for_training` holds back a validation
share.
