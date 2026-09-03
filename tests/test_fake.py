#!/usr/bin/env python3
if __name__ == "__main__":
    import LlamaClient as L, io, time
s = L.LlamaServer()
t=time.time(); n=0
for _ in s.stream_raw(L.Request(user='Count to 20', model='Qwen3-Coder-30B-A3B-Instruct-Q4_K_M', max_tokens=60)):
    n+=1; print(f"\r{n} frames {time.time()-t:.1f}s", end="")
print()
class Fake(io.BytesIO):
    def __enter__(self): return self
    def __exit__(self,*a): pass
s._open = lambda path, body=None: Fake(
b'data: {"choices":[{"delta":{"content":"hi"},"finish_reason":null}],"id":"x","model":"m","created":1,"system_fingerprint":"f"}\n'
b'data: {"error":{"code":500,"message":"context shift failed","type":"server_error"}}\n')
got=[]
try:
    for f in s.stream_raw(L.Request(user='x')): got.append(f)
except L.LlamaServerError as e: print("mid-stream raise after", len(got), "frame(s):", e)
