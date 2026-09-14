import time, torch
from vllm import LLM, SamplingParams

if __name__ == "__main__":
    FILL = "The quick brown fox jumps over the lazy dog while the server processes tokens at a steady pace and the network forwards every packet without delay. "
    prompt = "abc123xyz Count from one to three hundred, one number per line. Do not stop early.\n\n" + FILL * 145
    sp = SamplingParams(temperature=0, max_tokens=128, ignore_eos=True)
    
    llm = LLM(model="/models/hy-mt2-7b-awq2-mxfp4", kv_cache_dtype="fp8",
              max_model_len=8192, max_num_seqs=8, max_num_batched_tokens=2048,
              gpu_memory_utilization=0.92, enforce_eager=False, disable_log_stats=True)
    t0 = time.time()
    llm.generate([prompt], sp)
    print(f"warmup gen: {time.time()-t0:.1f}s")
    
    t0 = time.time()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
        outs = llm.generate([prompt], sp)
    gen_t = time.time() - t0
    ntok = len(outs[0].outputs[0].token_ids)
    print(f"profiled: {ntok} tokens in {gen_t:.2f}s -> {ntok/gen_t:.1f} t/s (incl overhead)")
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20, max_name_column_width=70))
    