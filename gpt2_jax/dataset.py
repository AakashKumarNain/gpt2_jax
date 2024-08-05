import jax.numpy as jnp

class SimpleDataLoader:
    def __init__(
        self,
        batch_size,
        seqlen,
        tokenizer,
        split="train",
        file_path=None,
    ):
        self.B = batch_size
        self.T = seqlen
        self.current_position = 0
        assert split in ("train", "val", "test"), "Expected split to be one of `train`, `val`, or `test`."
        self.split = split

        if file_path is not None:
            self.text = text = self.read_text_file(file_path)
            self.tokens = jnp.array(tokenizer.encode(text))
            print("Total number of tokens            :",len(self.tokens))
            print(f"Number of mini-batches in 1 epoch : {len(self.tokens) // (self.B * self.T)}")

    def read_text_file(self, file_path):
        # Load the file
        with open(file_path, "r") as f:
            text = f.read()
        return text

    def next_batch(self):
        B, T = self.B, self.T
        batch_tokens = self.tokens[self.current_position : self.current_position + B * T + 1]
        inputs = jnp.reshape(batch_tokens[:-1], (B, T))

        if self.split in ("train", "val"):
            targets = jnp.reshape(batch_tokens[1:], (B, T))
        else:
            targets = None

        # Advance the pointer to the current position by the number of
        # tokens consumed
        self.current_position +=  B * T

        # Check if we already processed the last batch
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0

        return inputs, targets

    def reset(self):
        self.current_position = 0
