"""Contributed HPC-02 encoders: one module per encoder, registering itself with `bittrellis.hpc02.register`.

    from bittrellis.hpc02 import Encoder, register

    def _encode(ctx, fmt):                 # ctx.rows float32 [rows, cols]; ctx.unit; ctx.params; ctx.calibration
        ...                                # return the tensor's GGUF block bytes for `fmt`
    register(Encoder("my_kq", 1, "regenerable", _encode))

See docs/hpc02_encoder_contract.md.
"""
