# Adapted from FluidInference/mobius parakeet-redux converter, Apache-2.0.
import re

def hf_to_nemo_name(name: str) -> str:
    """Map moondream/HF-transformers parameter names to NeMo EncDecRNNTBPEModel names."""
    n = name
    n = re.sub(r"^encoder\.subsampling\.layers\.(\d+)\.", r"encoder.pre_encode.conv.\1.", n)
    n = n.replace("encoder.subsampling.linear.", "encoder.pre_encode.out.")
    n = n.replace(".self_attn.q_proj.", ".self_attn.linear_q.")
    n = n.replace(".self_attn.k_proj.", ".self_attn.linear_k.")
    n = n.replace(".self_attn.v_proj.", ".self_attn.linear_v.")
    n = n.replace(".self_attn.o_proj.", ".self_attn.linear_out.")
    n = n.replace(".self_attn.relative_k_proj.", ".self_attn.linear_pos.")
    n = n.replace(".self_attn.bias_u", ".self_attn.pos_bias_u")
    n = n.replace(".self_attn.bias_v", ".self_attn.pos_bias_v")
    n = n.replace(".conv.norm.", ".conv.batch_norm.")
    n = n.replace("decoder.embedding.weight", "decoder.prediction.embed.weight")
    n = n.replace("decoder.lstm.", "decoder.prediction.dec_rnn.lstm.")
    n = n.replace("decoder.decoder_projector.", "joint.pred.")
    n = n.replace("encoder_projector.", "joint.enc.")
    n = n.replace("joint.head.", "joint.joint_net.2.")
    return n
