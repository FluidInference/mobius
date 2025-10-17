[Machine Learning Research](https://machinelearning.apple.com/)

[Open Menu](https://machinelearning.apple.com/research/core-ml-on-device-llama#localnav-menustate) [Close Menu](https://machinelearning.apple.com/research/core-ml-on-device-llama#)

- [Overview](https://machinelearning.apple.com/)
- [Research Highlights](https://machinelearning.apple.com/highlights)
- [Publications](https://machinelearning.apple.com/research/)
- [Events](https://machinelearning.apple.com/updates/)
- [Work with us](https://machinelearning.apple.com/work-with-us)

[content type highlight](https://machinelearning.apple.com/highlights) \| published November 1, 2024

# On Device Llama 3.1 with Core ML

![Hero](https://mlr.cdn-apple.com/media/Core_ML_blue_5e24d89886.png)

Many app developers are interested in building on device experiences that integrate increasingly capable large language models (LLMs). Running these models locally on Apple silicon enables developers to leverage the capabilities of the user’s device for cost-effective inference, without sending data to and from third party servers, which also helps protect user privacy. In order to do this, the models must be carefully optimized to effectively utilize the available system resources, because LLMs often have high demands for both memory and processing power.

This technical post details how to optimize and deploy an LLM to Apple silicon, achieving the performance required for real time use cases. In this example we use Llama-3.1-8B-Instruct, a popular mid-size LLM, and we show how using Apple’s [Core ML framework](https://developer.apple.com/documentation/coreml) and the optimizations described here, this model can be run locally on a Mac with M1 Max with about ~33 tokens/s decoding speed. While this post focuses on a particular Llama model, the principles outlined here apply generally to other transformer-based LLMs of different sizes.

We take the official definition and trained weights of the Llama-3.1-8B-Instruct model [hosted](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) on Hugging Face. We outline the steps to convert the model to the Core ML format using [Core ML Tools](https://apple.github.io/coremltools/docs-guides/), optimize it for on-device inference on a Mac, and benchmark its performance. We use a Mac with M1 Max and specifically target the GPU, as the models like the Llama-3.1-8B-Instruct are usually constrained by memory bandwidth, and the GPU offers the best combination of compute FLOPS and memory bandwidth on the device of our interest.

## Baseline Model Export and Performance

It is easiest to begin by exporting a version of the Llama model with the most basic options (e.g. no KV cache, static input shapes etc). This allows us to learn the export process, how the model generates tokens, how its performance is measured and the metrics used to report it. We will also use this model to establish a baseline performance and analyze it to understand why it is poor. This will then lead to a better understanding of the optimizations that we introduce in the following sections to improve the performance.

## Exporting PyTorch Model into Core ML

To make the model exportable we define a thin wrapper on top of the [LlamaForCausalLM](https://huggingface.co/docs/transformers/main/en/model_doc/llama#transformers.LlamaForCausalLM) class. This wrapped model uses fixed input shapes and no key-value caching (we go over that in subsequent sections). While this version of the model is not optimal for export, it serves as a good starting point. It requires only a slight modification to the [`LlamaForCausalLM`](https://huggingface.co/docs/transformers/main/en/model_doc/llama#transformers.LlamaForCausalLM) module, as shown below:

```hljs python
# !pip install transformers==4.44.2
import torch
from transformers.models.llama.modeling_llama import LlamaForCausalLM

class BaselineLlamaForCausalLM(LlamaForCausalLM):
    """Baseline LlamaForCausalLM model without key/value caching."""

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
    ) -> torch.Tensor:
        out = super().forward(
            input_ids,
            attention_mask,
            use_cache=False,
        )
        return out.logits

model_id: str = "meta-llama/Llama-3.1-8B-Instruct"
torch_model = BaselineLlamaForCausalLM.from_pretrained(model_id).eval()

```

To export, we will first trace the PyTorch model and then use [Core ML Tools](https://apple.github.io/coremltools/docs-guides/), both of these steps require the shapes of the input tensors to be provided.

```hljs python
# !pip install coremltools numpy
import coremltools as ct
import numpy as np

batch_size, context_size = 1, 2048
input_shape = (batch_size, context_size)

# trace the PyTorch model
example_inputs: tuple[torch.Tensor] = (
    torch.zeros(input_shape, dtype=torch.int32),
    torch.zeros(input_shape, dtype=torch.int32),
)
traced_model: torch.jit.ScriptModule = torch.jit.trace(
    torch_model,
    example_inputs=example_inputs,
)

# convert to Core ML format
inputs: list[ct.TensorType] = [\
    ct.TensorType(shape=input_shape, dtype=np.int32, name="inputIds"),\
    ct.TensorType(shape=input_shape, dtype=np.int32, name="attentionMask"),\
]

outputs: list[ct.TensorType] = [ct.TensorType(dtype=np.float16, name="logits")]
mlmodel: ct.models.MLModel = ct.convert(
    traced_model,
    inputs=inputs,
    outputs=outputs,
    minimum_deployment_target=ct.target.macOS13,
    skip_model_load=True,
)

```

Core ML by default produces a `Float16` [precision](https://apple.github.io/coremltools/docs-guides/source/typed-execution.html#choosing-the-appropriate-precision) model. Hence for this 8B model, the generated Core ML model will be about 16GB in size `((BitWidth / 2) x #ModelParameters)`. We verify that the outputs of the Core ML and PyTorch models (which is in `Float32` precision) match within a low tolerance.

## Model Inputs and Outputs

The `context_size` refers to the maximum amount of tokens that the model can process. We set it to 2048 (later we will vary it and see the impact in performance).

The model takes two inputs, both are statically shaped, i.e., the shape is a constant irrespective of the length of the input text.

- `inputIds <shape=(batch_size, context_size)>`: This represents the tokenized text input sequence. That is, each element is an integer corresponding to a token ID from the model’s vocabulary. This is generated by tokenizing the input text using the model’s associated tokenizer. The tokens beyond the input text are padded with zero up to the context size.

- `attentionMask <shape=(batch_size, context_size)>`: this is a binary tensor that is set to 1 where text tokens are present, and 0 otherwise in the padded portion. The model takes this and internally transforms this into a causal mask, of shape `(1, 1, context_size, context_size)`, which is added to the [self-attention](https://en.wikipedia.org/wiki/Attention_(machine_learning)) matrix. It ensures that the values in the padded region are ignored, and that each input token only attends to the previous tokens, thereby preserving the autoregressive nature of the language model.


There is one output returned by the model:

- `logits <shape=(batch_size, context_size, vocab_size)>`: raw unnormalized probability scores for each token in the sequence, for every entry in the vocabulary. `vocab_size` refers to the total number of unique tokens in the model’s vocabulary (its value is `128,256` for the Llama 3.1 family of models).

## Execution

The model is run in two stages: “prompt” and “extend.” Let’s understand this with an example, where we invoke the model with the prompt “What is generative AI?” This prompt has 7 tokens.

- **Prompt**: The first 7 elements of `inputIds` `inputIds` are set to the integer tokens generated by the tokenizer from the prompt string. The rest are all `0` s. Similarly, the first 7 elements of `attentionMask` are set to `1` s and rest to `0` s. With these inputs, the model will produce logits, of shape `(1, context_size, vocab_size)`. To determine the next token in the sequence, we will employ the basic greedy sampling strategy, which suffices for our benchmarking code, it also has the benefit of being deterministic (in practice a more sophisticated sampling strategy would be typically used). This approach selects the token with the highest probability, hence in this case, it will pick the token at the 7th location, i.e., `argmax(logits[0, 6, :])`.

- **Extend**: The selected token is appended to the non-zero `inputIds`, `1` is appended to `attentionMask`, and model prediction is invoked. Therefore, in this prediction, both `inputIds` and `attentionMask` will have 8 non-zeros values. The next prediction will have 9 non-zero values, and so on. Hence we will keep producing one token at a time, one with each prediction call. This process continues in a loop until one of two conditions is met: either we reach the token limit (as specified by the user or the max context size) or the model generates an end-of-sequence token.


With the exported Core ML model, we get the following output:

```hljs python
# Float16 Core ML Llama-3.1-8B-Instruct model

Prompt: "What is generative AI?"
Response: "Generative AI is a type of artificial intelligence that uses algorithms to generate new, original content, such as images, music, videos, or text. Unlike traditional AI, which is designed to analyze and process existing data, generative AI creates new data from scratch. This technology has the potential to revolutionize various industries, including art, music, and entertainment..."

```

## Understanding Execution Performance

To evaluate the model’s performance, we calculate two metrics: prompt latency and extend throughput.

- **Prompt latency**: This metric evaluates the model’s initial responsiveness. It is measured by the time taken by the model to produce its first token, after the prompt is processed. We measure it in milliseconds, lower is better. It is also often referred to as TTFT (time to first token), that’s the term we will use as well for the rest of this article.
- **Extend throughput**: This metric measures the model’s token generation efficiency. It is computed by dividing the total number of tokens generated by the total time taken for generation. Extend throughput reflects the model’s speed in producing a continuous stream of output tokens. We typically let the model generate ~100 tokens to calculate this metric. This is the main performance metric we will use to compare the various versions of the models. It will be reported in number of tokens generated per second (tokens/s). Higher is better.

All the TTFT and extend throughput numbers reported in this article have been measured by using the Swift runner using [Core ML framework](https://developer.apple.com/documentation/coreml) with newly released [MLTensor APIs](https://developer.apple.com/documentation/coreml/MLTensor), using a Mac with M1 Max running macOS Sequoia 15.2 Beta.

## Performance of the Baseline Model

For the baseline model, which is statically shaped (inputs zero-padded up to max context size), and does not employ key-value cache, we obtained the following prompt and extend throughputs with context size 2048.

```hljs python
# context_size: 2048
[Prompt]  => 7 tokens, latency (TTFT): 5374.15 ms
[Extend]  => 100 tokens, throughput: 0.19 tokens/s

```

As can be seen the extend throughput is very low, even lower than 1 token/s. This is expected with how this model is constructed and executed. There are two main reasons contributing to this slow inference:

- (a) The model does an attention computation (i.e., all the matrix multiplications) for the full sequence length of `context_size` (2048 above), even though the actual tokens needed to be processed are much lower (<=107, for 7 prompt + 100 tokens generated). This is because a static shaped padded inputs are used.
- (b) For each token that is produced, there is a lot of re-compute happening, for values that have already been evaluated while processing previous tokens. The transformer architecture uses the attention mechanism which involves 3 tensors: “query,” “key” and “value,” which keep growing with the number of tokens been processed. As explained in the next section in more detail, the “key” and “value” tensors, for previous tokens, can be _cached_ (this is referred to as KV cache) and only computed for the new tokens, while updating the cache. This is not happening for the baseline run of the model.

In the subsequent sections, we address both these issues, by using flexible shaped inputs and a stateful key-value cache mechanism to drastically improve the performance.

For this baseline model, the effect of both (a) and (b) can be reduced by decreasing the `context_size` with which the model is exported, resulting in increased throughput, as seen in the table below. Obviously, this limits the practical usability of the model if it’s forced to limit to a smaller text length window.

| **Maximum Context Size** | **Extend Throughput (tokens/s)** |
| --- | --- |
| 8192 | 0.01 |
| 4096 | 0.09 |
| 2048 | 0.19 |
| 1024 | 0.39 |
| 512 | 0.8 |
| 256 | 1.67 |
| 128 | 3.23 |

**Table 1:** On device performance with baseline model (M1 Max, macOS Sequoia 15.2 beta).

## Model Optimizations

Now that we have established the baseline, we will look at the key optimizations to improve the performance of it. In addition to addressing the two shortcomings that we identified in the previous section, we will also learn how to incorporate a more optimized version of the attention computation (via fused SDPA op) and quantize the weights of the model to get a significant bump in the decoding speed. We will cover all these optimizations in the following three sections:

- **Fused Scaled Dot Product Attention (SDPA)**
- **Key-value cache and flexible shaped inputs**
- **Block-wise int4 weight quantization**

## Fused Scaled Dot Product Attention (SDPA)

Transformer models use what is referred to as the Scaled Dot-Product Attention (SDPA), within the multi-head attention blocks. The SDPA op takes the key, value, query tensors along with the mask and computes the attention matrix, for all tokens, and updates the representation that is passed on to the next block. It is computationally intensive, involving multiple matrix multiplications, softmax, addition, and multiplication operations on high-dimensional query, key, and value tensors.

Starting with macOS Sequoia, the Core ML model format has the [`scaled_dot_product_attention`](https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html#module-coremltools.converters.mil.mil.ops.defs.iOS18.transformers) available as a high level operation [(see Figure 1)](https://machinelearning.apple.com/research/core-ml-on-device-llama#figure1). This gets mapped to a single fused GPU kernel, which executes more efficiently. For instance, with a fused kernel, the “attention” tensor (result of multiplication of the “key” and “query” tensors) that can be very large in size `((1, #attn_heads, #token_length, #token_length))` need not be fully materialized.

[![](https://mlr.cdn-apple.com/media/Fig1_SDPA_afe646f591.png)](https://mlr.cdn-apple.com/media/Fig1_SDPA_afe646f591.png)

Figure 1: Before macOS 15 Sequoia, `SDPA` operation is decomposed into several operations. Conversion using Core ML Tools with macOS 15 Sequoia as the target, uses a fused `SDPA` representation that is accelerated on the GPU.

While the Core ML-GPU compiler tries to automatically detect the pattern and fuse the SPDA op, using the PyTorch op `torch.nn.functional.scaled_dot_product_attention` (which the Hugging Face’s Llama implementation already uses), along with setting the minimum deployment target to macOS 15+ in the [Core ML Tools](https://apple.github.io/coremltools/docs-guides/) `convert` API, will automatically ensure the resulting model to have the fused SDPA op [(see Figure 1)](https://machinelearning.apple.com/research/core-ml-on-device-llama#figure1).

```hljs python
mlmodel: ct.models.MLModel = ct.convert(
    traced_model,
    inputs=inputs,
    outputs=outputs,
    # Set target to macOS15+ to use the fused SDPA operation.
    minimum_deployment_target=ct.target.macOS15,
    skip_model_load=True,
)

```

## Key-Value Cache and Flexible Shaped Inputs

Transformer architecture consists of multiple attention blocks and each such block generates tensors that are referred to as the “query,” “key” and “value.” These tensors are generated for every token that the model processes. When a new token arrives, its query projection needs to be processed, via the SDPA operation, in combination with the key and value projections for all the previous tokens. In the baseline model, for each new token, the key-value projections are being recomputed for all the previous tokens. Instead of doing that, we now create a cache for both keys and values and initialize it with zeros. For the Llama-31-8B-Instruct model, for a context size of 2048, this cache, for each of `Key` and `Value`, would be of the shape `(32, 1, 8, 2048, 128)`. This is because there are 32 attention blocks, each with its own key/value tensors, there are 8 key/value heads in each block, and the size of the projections are 128. Now, when the model has processed, say, `t` tokens, the key-value cache will be updated, i.e. `Key[:, :, :, 0:t, :]` will have the computed values and rest will be zeros (same for the `Value` cache). When the `t+1`-th token is to be processed, its key/value tensor will be computed and appended to the cache (making the non-zero values span `Key[:, :, :, 0:t+1, :]`) and then used with the query tensor. For the next token, cache will be updated for the `t+1`-th slice and so on.

Introduction of the key-value cache allows us to update the inputs that the model consumes, now we can make them flexible shaped, as follows:

- `inputIds <shape=(batch_size, [1, context_size])>`: for the prompt stage, in our example, the `inputIds` will be of shape `(1, 7)`. This will result in updating the key-value cache for 7 tokens. Thereafter, in the extend stage, where tokens are fed one at a time, `inputIds` will take the shape `(1, 1)`, and each decoding step will result in updating the cache by 1 token.

- `causalMask <shape=(batch_size, 1, [1, context_size], [1, context_size])>`: unlike the baseline model, now we directly feed the causal mask to the model, instead of the binary attention mask and the model computing the causal mask from it internally. For the prompt stage, in our running example, the shape of the causal mask will be `(1, 1, 7, 7)`, and the values will be `-inf` in the upper triangular region and 0 elsewhere (that’s how “causality” is encoded, preventing tokens from attending to their future tokens). This mask will be added as is to the “attention” tensor in the SDPA operation. Presence of `-inf` makes sure that when the `softmax` operation is applied, the dot products of current token with future tokens become 0. In the extend mode, for the first token, this input will be set to shape `(1, 1, 1, 8)`, for the next one to `(1, 1, 1, 9)`, and so on; with all values set to 0. In the decoding stage since there is only one token being fed, there is no future token to mask, hence the value of `casualMask` is all 0s.


A simple back-of-the-envelope calculation shows how with this input change, the `matmul` s in the SDPA op become much smaller, compared to the baseline model:

- In the baseline model, the query, key, value tensors are always of shape `(1, 32, 2048, 128)`, for both prompt and extend. The `query x key matmul` operation, will produce the attention tensor of shape `(1, 32, 2048, 2048)`, and will have complexity of `O(32*128*2048^2)`.

- In the key-value cache and flexible shaped model:
  - during the prompt stage the query, key, value tensors are of shape `(1, 32, prompt_length, 128)`, hence the matmul complexity of `O(32*128*prompt_length^2)`.
  - And in the decoding stage, the query tensor is always `(1, 32, 1, 128)` and key value tensors of shape, `(1, 32, current_token_id, 128)`, hence complexity of `O(32*128*current_token_id)`.

We note that the number of operations are significantly less compared to the baseline model. Before we take a look at how this translates in improving the performance metrics, we need to decide how to implement the key-value cache. There are several ways to do it (static, dynamic, etc). We will consider a static cache and two mechanisms to implement it, as described next.

## Key-Value Cache as Model I/O (inputs and outputs)

This is the basic version. In this case, the model is [“pure”](https://en.wikipedia.org/wiki/Pure_function) and the key-value cache is implemented via model inputs and outputs [(see Figure 2)](https://machinelearning.apple.com/research/core-ml-on-device-llama#figure2). That is, the pre-allocated cache tensor is fed as an input to the model, during token processing the model generates an updated cache tensor and returns that as an output. The driver code then takes the output, and passes it as the input for the next iteration [(see Figure 2)](https://machinelearning.apple.com/research/core-ml-on-device-llama#figure2).

Figure 2: Key-value cache implemented as model I/O (inputs and outputs). This is possible to do prior to macOS Sequoia as well. For details, please refer to [this video](https://developer.apple.com/videos/play/wwdc2024/10161/?time=586) on deploying models with Core ML.

With this approach, these are the performance numbers we get:

```hljs python
# context_size: 2048
[Prompt]  => 7 tokens, latency (TTFT): 933.89 ms
[Extend]  => 100 tokens, throughput: 1.25 tokens/s

```

Extend throughput is about an order of magnitude faster than the baseline model. However it’s still very slow. If we look at the performance as a function of the context size, we get an idea of what’s going on:

| **Maximum Context Size** | **Extend Throughput (tokens/s)** |
| --- | --- |
| 8192 | 0.1 |
| 4096 | 0.17 |
| 2048 | 1.25 |
| 1024 | 8.36 |
| 512 | 10.92 |
| 256 | 13.07 |
| 128 | 14.69 |

**Table 2:** On device performance with key-value cache implemented as model I/O (M1 Max, macOS Sequoia 15.2 beta).

We notice that the performance improves quite rapidly with lower context size. In this model, multiple data copies of the key/value tensor are happening, at the time of updating it inside the model for each attention block, then between the iterations when copying it from the output to the next input. Since the size of the key-value cache are `2 (Key/Value) * 2 (#BytesInFP16DataType) * 32 (#Layers) * 8 (#KeyValueHeads) * 128 (AttentionHeadDim) * ContextSize` growing with the `context_size`, the larger it is, the more time is spent in memory copies. With Llama3.1 8B model, this can go up to ~1GB when using 8192 context size, which would result in huge overhead to copy. With the stateful key-value cache, we avoid these costs. Let’s see how we do so next.

## Key-Value Cache as State

Starting with macOS Sequoia, Core ML introduced a new type of inputs called [“states”](https://developer.apple.com/documentation/coreml/mlstate) [(see Figure 3)](https://machinelearning.apple.com/research/core-ml-on-device-llama#figure3). A prediction can now be stateful, in that the values of the state tensors will get updated at the end of the prediction call, without returning them explicitly. Depending on the compute backend and the operations around state tensors in the model graph, the complier may be able to perform the update “in place.” If that happens, the computational overhead associated with transferring states in and out of the model is significantly reduced. This is what happens when we implement key-value cache for the Llama model via Core ML states.

```hljs python
# context_size: 2048
[Prompt]  => 7 tokens, latency (TTFT): 128.32 ms
[Extend]  => 99 tokens, throughput: 16.26 tokens/s

```

We now see that the performance is improved ~13 times, compared to key-value cache as I/O for 2048 context size. It is also much more consistent across different context sizes now (several computations in the model still scale up with context size, hence a slight monotonic pattern is still observed). It’s worth noting that beyond the 2048 context size, the key-value cache becomes too large to consistently fit within the GPU cache. This leads to an increased frequency of cache misses, resulting in a decrease in the decoding speed. This effect is not evident up to the context size of 1024, as the key-value cache fit within the GPU cache boundaries.

| **Maximum Context Size** | **Extend Throughput (tokens/s)** |
| --- | --- |
| 8192 | 12.31 |
| 4096 | 14.78 |
| 2048 | 16.26 |
| 1024 | 17.41 |
| 512 | 17.59 |
| 256 | 17.61 |
| 128 | 17.89 |

**Table 3:** On device performance with key-value cache as state (M1 Max, macOS Sequoia 15.2 beta).Figure 3: Key-value cache as model state. This is possible with the macOS Sequoia and is more efficient than implementing a cache with model inputs and outputs (Figure 2). For more details, please refer to [this video](https://developer.apple.com/videos/play/wwdc2024/10161/?time=624) on deploying models with Core ML.

## Model Export

We now show how to implement the stateful key-value cache and flexible input features.

We implement our own static cache implementation that is passed to the `transformers` API. This is done via the class `SliceUpdateKeyValueCache`, that extends the `Cache` class. It essentially implements a simple update logic via the slicing operation, these op patterns are then detected by the Core ML-GPU compiler and allows it to perform in-place updates.

```hljs python
from typing import Any, Optional, Sequence

import torch
from transformers.cache_utils import Cache

class SliceUpdateKeyValueCache(Cache):
    """Helper class for in-place slice updating key/value caches."""

    def __init__(
        self,
        *,
        shape: Sequence[int],
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Create key/value cache of shape:
        (#layers, batch_size, #kv_heads, context_size, head_dim)."""
        super().__init__()
        self.past_seen_tokens: int = 0
        self.k: torch.Tensor = torch.zeros(shape, dtype=dtype)
        self.v: torch.Tensor = torch.zeros(shape, dtype=dtype)

    def update(
        self,
        k_state: torch.Tensor,
        v_state: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update key / value cache tensors for slice [begin, end).\
        Return slice of key / value cache tensors from [0, end)."""\
        position = cache_kwargs.get("cache_position", None)\
        assert position is not None, "cache_position required to update cache."\
        begin, end = self.past_seen_tokens, self.past_seen_tokens + position.shape[-1]\
        self.k[layer_idx, :, : k_state.shape[1], begin:end, :] = k_state\
        self.v[layer_idx, :, : v_state.shape[1], begin:end, :] = v_state\
        k_state = self.k[layer_idx, :, :, :end, :]\
        v_state = self.v[layer_idx, :, :, :end, :]\
        return k_state, v_state\
\
    def get_seq_length(self, _: int = 0) -> int:\
        """Get the sequence length of the cache."""\
        return self.past_seen_tokens\
\
```\
\
We define a wrapper `KVCacheStateLlamaForCausalLM` on top of the [LlamaForCausalLM](https://huggingface.co/docs/transformers/main/en/model_doc/llama#transformers.LlamaForCausalLM) class to use this custom cache class. To make sure that Core ML conversion process is able to detect and generate a Core ML model with key-value cache as state inputs, we register them using PyTorch’s `register_buffer` API:\
\
```hljs python\
import torch\
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaForCausalLM\
\
class KvCacheStateLlamaForCausalLM(torch.nn.Module):\
    """Model wrapper to swap cache implementation and register as buffers."""\
\
    def __init__(\
        self,\
        model_path: str, *,\
        batch_size: int = 1, context_size: int = 4096\
    ) -> None:\
        super().__init__()\
        self.model = LlamaForCausalLM.from_pretrained(model_path)\
        config: LlamaConfig = self.model.config\
        self.kv_cache_shape: tuple[int, ...] = (\
            config.num_hidden_layers,\
            batch_size,\
            config.num_key_value_heads,\
            context_size,\
            config.hidden_size // config.num_attention_heads,\
        )\
        # Register KV cache buffers to be recognized as Core ML states\
        self.kv_cache = SliceUpdateKeyValueCache(shape=self.kv_cache_shape)\
        self.register_buffer("keyCache", self.kv_cache.k)\
        self.register_buffer("valueCache", self.kv_cache.v)\
\
    @torch.no_grad()\
    def forward(\
        self,\
        input_ids: torch.LongTensor,\
        causal_mask: torch.Tensor,\
    ) -> torch.Tensor:\
        # Compute past seen tokens used for updating key/value cache slices\
        self.kv_cache.past_seen_tokens = causal_mask.shape[-1] - input_ids.shape[-1]\
        return self.model(\
            input_ids,\
            attention_mask=causal_mask,\
            past_key_values=self.kv_cache,\
            use_cache=True,\
        ).logits\
\
```\
\
In the export code, we use the `coremltools.RangeDim` class to denote the model inputs as [flexible shape](https://apple.github.io/coremltools/docs-guides/source/flexible-inputs.html), and the `coremltools.StateType` class to ensure that `kv_cache.k` and `kv_cache.v` are recognized as [state inputs](https://apple.github.io/coremltools/docs-guides/source/stateful-models.html). The rest is the same as before.\
\
```hljs python\
import coremltools as ct\
\
batch_size, context_size = 1, 2048\
\
# Initialize and trace PyTorch model\
loaded_model: torch.nn.Module = KvCacheStateLlamaForCausalLM(\
    model_id, batch_size=batch_size, context_size=context_size\
).eval()\
example_inputs: tuple[torch.Tensor, ...] = (\
    torch.zeros((1, 2), dtype=torch.int32),\
    torch.zeros((1, 1, 2, 5), dtype=torch.float32)\
)\
traced_model: torch.jit.ScriptModule = torch.jit.trace(\
    loaded_model.eval(), example_inputs=example_inputs\
)\
\
# Convert to Core ML\
query_size = ct.RangeDim(lower_bound=1, upper_bound=context_size, default=1)\
final_step = ct.RangeDim(lower_bound=1, upper_bound=context_size, default=1)\
inputs: list[ct.TensorType] = [\
    ct.TensorType(shape=(batch_size, query_size), dtype=np.int32, name="inputIds"),\
    ct.TensorType(\
        shape=(batch_size, 1, query_size, final_step),\
        dtype=np.float16,\
        name="causalMask",\
    ),\
]\
states: list[ct.StateType] = [\
    ct.StateType(\
        wrapped_type=ct.TensorType(shape=loaded_model.kv_cache_shape, dtype=np.float16),\
        name="keyCache",\
    ),\
    ct.StateType(\
        wrapped_type=ct.TensorType(shape=loaded_model.kv_cache_shape, dtype=np.float16),\
        name="valueCache",\
    ),\
]\
outputs: list[ct.TensorType] = [ct.TensorType(dtype=np.float16, name="logits")]\
mlmodel: ct.models.MLModel = ct.convert(\
    traced_model,\
    inputs=inputs,\
    outputs=outputs,\
    states=states,\
    minimum_deployment_target=ct.target.macOS15,\
    skip_model_load=True,\
)\
\
```\
\
To run the model, we don’t need to manage the cache as part of the model I/O; the only required change is to pass the state when calling `predict`.\
\
```hljs python\
kv_cache = mlmodel.make_state()\
logits = mlmodel.predict(data=inputs, state=kv_cache)["logits"]\
\
```\
\
## Block-Wise Int4 Quantization\
\
macOS Sequoia [introduced](https://apple.github.io/coremltools/docs-guides/source/opt-whats-new.html) several low-bit quantization methods supported by Core ML, including 4-bit block-wise linear quantization, channel group-wise palettization etc, to enhance model compression and accuracy [(see Figure 4)](https://machinelearning.apple.com/research/core-ml-on-device-llama#figure4). These techniques are essential for optimizing memory usage and performance for on-device inference. For example, low-bit palettization greatly reduces the model’s memory footprint and improves latency on the neural engine, block-wise quantization minimizes accuracy loss by applying quantization at a higher granularity and is optimized for the GPU. You can find additional information [here](https://apple.github.io/coremltools/docs-guides/source/opt-overview.html).\
\
[![](https://mlr.cdn-apple.com/media/Fig4_compression_fc6629cb62.png)](https://mlr.cdn-apple.com/media/Fig4_compression_fc6629cb62.png)\
\
Figure 4: Weight compression features introduced in macOS Sequoia. For details, check out this [WWDC 2024 session video.](https://developer.apple.com/videos/play/wwdc2024/10159/)\
\
To further improve the model performance, we will quantize the model to `Int4` format using block-wise quantization (block size = 32). We will use a simple data-free Post-Training Quantization (PTQ) approach. Since our main focus here is to evaluate latency/throughput, we do not evaluate the quality of the model on, say, datasets that are common in the literature to qualify accuracy. That said, we observed that the quantized model produces very similar outputs to that of the `Float16` precision model on a few prompts we tried. Based on the application and its testing requirements, there may be an accuracy loss with PTQ quantization and some calibration or fine tuning based quantization may be required. However, doing that first on the PyTorch model (e.g., with [coremltools.optimize.torch](https://apple.github.io/coremltools/docs-guides/source/opt-workflow.html) APIs) and then converting, will not alter the performance.\
\
The following code snippet quantizes the `Float16` model to `Int4` format.\
\
```hljs python\
op_config = ct.optimize.coreml.OpLinearQuantizerConfig(\
    mode="linear_symmetric",\
    dtype="int4",\
    granularity="per_block",\
    block_size=32,\
)\
config = ct.optimize.coreml.OptimizationConfig(global_config=op_config)\
mlmodel_int4 = ct.optimize.coreml.linear_quantize_weights(\
    mlmodel, config=config\
)\
\
```\
\
```hljs python\
# Int4 Core ML Llama-3.1-8B-Instruct model\
\
Prompt: "What is generative AI?"\
Response: "Generative AI refers to a type of artificial intelligence (AI) that can create new content, such as images, videos, music, or text, based on a given prompt or input. This technology uses machine learning algorithms to generate novel and often surprising outputs that can be used for a variety of applications, including art, design, entertainment, and even education..."\
\
```\
\
With this simple change, the extend throughput for the default context size of `2048` increases to `~33` tokens per second, twice as fast compared to the Float16 model. Model size also reduced from `16 GB` to `4.2 GB`, a ~4x reduction.\
\
```hljs python\
# context_size: 2048\
[Prompt] => 7 tokens, latency (TTFT): 51.91 ms\
[Extend] => 100 tokens, throughput: 33.67 tokens/s\
\
```\
\
The following table shows the impact of context size on the extend throughput.\
\
| **Maximum Context Size** | **Extend Throughput (tokens/s)** |\
| --- | --- |\
| 8192 | 21.46 |\
| 4096 | 29.39 |\
| 2048 | 33.67 |\
| 1024 | 37.58 |\
| 512 | 38.07 |\
| 256 | 39.19 |\
| 128 | 39.44 |\
\
**Table 4:** On device performance with block-wise Int4 quantization (M1 Max, macOS Sequoia 15.2 beta).\
\
## Model Export via `torch.export` and Inference via ExecuTorch\
\
In this article we used the `torch.jit.trace` method for capturing the PyTorch graph, to then pass on to Core ML Tools for conversion to the Core ML format. The newer `torch.export` path is still in beta and possibly needs minor changes to the Llama model definition to export. Core ML Tools also supports the `torch.export` path (also in beta mode), both directly and via a backend to ExecuTorch.\
\
Llama can be executed on ExecuTorch through the Core ML backend by utilizing `torch.export` with a custom export path. [ExecuTorch](https://pytorch.org/executorch-overview) is part of the PyTorch ecosystem and focuses on deploying machine learning models on mobile and edge devices with an end to end PyTorch experience. It features a [Core ML backend](https://pytorch.org/executorch/stable/build-run-coreml.html) that utilizes Core ML Tools for model export and the [Core ML framework](https://developer.apple.com/documentation/coreml) to efficiently run machine learning models within the ExecuTorch runtime on Apple devices. In addition, ExecuTorch has implemented a custom export path for the Llama family of models. Learn more [here](https://github.com/pytorch/executorch/tree/v0.4.0/examples/models/llama2) to get started using the ExecuTorch Core ML backend to export the Llama models and deploy on a Mac.\
\
## Conclusion\
\
Using the Core ML framework and the optimizations described in this post, app developers can deploy LLMs to run locally on Apple silicon, leveraging the capabilities of the user’s hardware for cost-effective inference on device, which also helps protect user privacy.\
\
In this post, we detailed the process of optimizing a popular LLM, Llama-3.1-8B-Instruct, and deploying it to a Mac with M1 Max running macOS Sequoia to achieve a decoding rate of ~33 tokens/s. To do this, we applied two key optimizations to address the major bottlenecks of large attention matrix computations and model weight memory: quantization to `Int4` to reduce the model weight size, and stateful key-value cache to reuse compute and reduce the amount of data copying in each decoding iteration.\
\
The principles described here apply generally to other transformer-based LLMs, and as increasingly powerful LLMs are being trained with smaller parameter counts, their on-device deployments to Apple silicon via Core ML should become even faster and more efficient.\
\
## Related readings and updates.\
\
[**KV-Runahead: Scalable Causal LLM Inference by Parallel Key-Value Cache Generation**](https://machinelearning.apple.com/research/kv-runahead)\
\
May 14, 2024 \| [research area Methods and Algorithms](https://machinelearning.apple.com/research/?domain=Methods%20and%20Algorithms), [research area Speech and Natural Language Processing](https://machinelearning.apple.com/research/?domain=Speech%20and%20Natural%20Language%20Processing) \| [conferenceICML](https://machinelearning.apple.com/research/?event=ICML)\
\
Large Language Model or LLM inference has two phases, the prompt (or prefill) phase to output the first token and the extension (or decoding) phase to the generate subsequent tokens. In this work, we propose an efficient parallelization scheme, KV-Runahead to accelerate the prompt phase. The key observation is that the extension phase generates tokens faster than the prompt phase because of key-value cache (KV-cache). Hence, KV-Runahead…\
\
[Read more](https://machinelearning.apple.com/research/kv-runahead)\
\
[**Deploying Transformers on the Apple Neural Engine**](https://machinelearning.apple.com/research/neural-engine-transformers)\
\
June 6, 2022 \| [research area Computer Vision](https://machinelearning.apple.com/highlights?domain=Computer%20Vision), [research area Speech and Natural Language Processing](https://machinelearning.apple.com/highlights?domain=Speech%20and%20Natural%20Language%20Processing)\
\
An increasing number of the machine learning (ML) models we build at Apple each year are either partly or fully adopting the [Transformer architecture](https://arxiv.org/abs/1706.03762). This architecture helps enable experiences such as [panoptic segmentation in Camera with HyperDETR](https://machinelearning.apple.com/research/panoptic-segmentation), [on-device scene analysis in Photos](https://machinelearning.apple.com/research/on-device-scene-analysis), [image captioning for accessibility](https://support.apple.com/guide/iphone/use-voiceover-for-images-and-videos-iph37e6b3844/ios), [machine translation](https://apps.apple.com/us/app/translate/id1514844618), and many others. This year at WWDC 2022, Apple is making available an open-source reference [PyTorch](https://pytorch.org/) implementation of the Transformer architecture, giving developers worldwide a way to seamlessly deploy their state-of-the-art Transformer models on Apple devices.\
\
[Read more](https://machinelearning.apple.com/research/neural-engine-transformers)\
\
![Bottom banner](https://mlr.cdn-apple.com/media/Discover_1440x420_2x_9c465d585e.jpg)\
\
## Discover opportunities in Machine Learning.\
\
Our research in machine learning breaks new ground every day.\
\
[Work with us](https://machinelearning.apple.com/work-with-us)\
\
1. [Machine Learning Research](https://machinelearning.apple.com/)\
2. [Publications](https://machinelearning.apple.com/research)\
3. On Device Llama 3.1 with Core ML\
\
\
- [Privacy Policy](https://www.apple.com/legal/privacy/)\
- [Terms of Use](https://www.apple.com/legal/internet-services/terms/site.html)\
- [Legal](https://www.apple.com/legal/)\
\
Copyright © 2025 [Apple Inc.](https://www.apple.com/) All rights reserved.
