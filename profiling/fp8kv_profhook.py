"""In-process profiler for EngineCore: wraps both step methods, profiles a window."""
import os

def _install():
    if not os.environ.get("FP8KV_PROF_START"):
        return
    import threading, time, sys

    def _poll():
        for i in range(1500):
            try:
                from vllm.v1.engine.core import EngineCore
                break
            except Exception:
                time.sleep(0.2)
        else:
            sys.stderr.write("[profhook] EngineCore never appeared\n")
            return
        _hook(EngineCore)

    def _hook(EngineCore):
        import torch
        START = int(os.environ["FP8KV_PROF_START"])
        NSTEPS = int(os.environ.get("FP8KV_PROF_STEPS", "20"))
        OUT = os.environ.get("FP8KV_PROF_OUT", "/work/prof_engine.txt")
        n_box = [0]
        prof_box = [None]

        def _make_wrap(orig, name):
            def _step(self):
                n_box[0] += 1
                n = n_box[0]
                if n == START:
                    from torch.profiler import profile, ProfilerActivity
                    prof_box[0] = profile(activities=[ProfilerActivity.CPU,
                                                      ProfilerActivity.CUDA])
                    prof_box[0].start()
                    sys.stderr.write(f"[profhook] start at step {n} ({name})\n")
                r = orig(self)
                if n == START + NSTEPS:
                    prof_box[0].stop()
                    ka = prof_box[0].key_averages()
                    with open(OUT, "w") as f:
                        f.write(ka.table(sort_by="self_cuda_time_total", row_limit=25,
                                         max_name_column_width=90))
                        tot = sum(getattr(e, "self_device_time_total", 0) for e in ka)
                        f.write(f"\nTOTAL self GPU time across {NSTEPS} steps: {tot/1000:.1f} ms "
                                f"({tot/NSTEPS/1000:.2f} ms/step GPU-busy)\n")
                    sys.stderr.write(f"[profhook] wrote {OUT}\n")
                return r
            return _step

        EngineCore.step = _make_wrap(EngineCore.step, "step")
        if hasattr(EngineCore, "step_with_batch_queue"):
            EngineCore.step_with_batch_queue = _make_wrap(
                EngineCore.step_with_batch_queue, "bq")
        sys.stderr.write("[profhook] armed\n")

    threading.Thread(target=_poll, daemon=True).start()

_install()
