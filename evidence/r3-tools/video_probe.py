#!/usr/bin/env python3
"""Send a short native mp4 (evidence/r3/research/bugs.md item 2) and report the HTTP status and /health after it.
Without GLM53_VIDEO_FRAMES_FIX the expected result is a dead engine, so run it as the last step of a boot.
Usage: video_probe.py CLIP.mp4 [OUT.json]"""
import base64
import json
import sys
import time
import urllib.error
import urllib.request

u = "http://127.0.0.1:8000"
clip = sys.argv[1]
m = json.load(urllib.request.urlopen(u + "/v1/models"))["data"][0]["id"]
v = "data:video/mp4;base64," + base64.b64encode(open(clip, "rb").read()).decode()
body = {"model": m, "max_tokens": 64, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": [{"type": "video_url", "video_url": {"url": v}},
                                                  {"type": "text", "text": "Describe the video in one sentence."}]}]}
req = urllib.request.Request(u + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
res = {"clip": clip}
try:
    r = urllib.request.urlopen(req, timeout=600)
    res["status"] = r.status
    res["content"] = json.loads(r.read())["choices"][0]["message"].get("content", "")[:200]
except urllib.error.HTTPError as e:
    res["status"] = e.code
    res["body"] = e.read()[:300].decode(errors="replace")
except Exception as e:
    res["status"] = None
    res["error"] = f"{type(e).__name__}: {e}"
time.sleep(5)
try:
    res["health_after"] = urllib.request.urlopen(u + "/health", timeout=30).status
except Exception as e:
    res["health_after"] = f"{type(e).__name__}"
print("VIDEO " + json.dumps(res))
if len(sys.argv) > 2:
    open(sys.argv[2], "w").write(json.dumps(res, indent=1) + "\n")
