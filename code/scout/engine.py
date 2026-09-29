# SPDX-License-Identifier: Apache-2.0
"""Stdlib HTTP client for an OpenAI-compatible vLLM server, with SuperScout-7B's sampling pinned.

Sampling is part of the policy. Every field is set client-side on every request; the served
model's generation_config is never relied on. The resolved dict is exposed as `engine.sampling`
and written onto every episode record.
"""
from __future__ import annotations
import http.client
import json
import socket
import time
import urllib.error
import urllib.request

PINNED = {"top_p": 1.0, "top_k": -1, "repetition_penalty": 1.0,
          "presence_penalty": 0.0, "frequency_penalty": 0.0}
TEMPERATURE = 0.9           # single draw at temperature 0.9
MAX_TOKENS = 2048
STOP = ["<|im_end|>"]

TRANSIENT = (ConnectionError, TimeoutError, socket.timeout,
             http.client.RemoteDisconnected, http.client.IncompleteRead)


def resolved_sampling(temperature=TEMPERATURE):
    """The exact sampling dict logged with every result."""
    return {"temperature": temperature, "max_tokens": MAX_TOKENS, "stop": list(STOP), **PINNED}


class HttpEngine:
    """POST /v1/chat/completions with the pinned sampling and a per-episode cache_salt."""

    def __init__(self, endpoint, model, temperature=TEMPERATURE, timeout=180, max_retries=8):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.sampling = resolved_sampling(temperature)

    def _payload(self, messages, cache_salt):
        p = {"model": self.model, "messages": messages,
             "temperature": self.temperature, "max_tokens": MAX_TOKENS,
             "stop": list(STOP), **PINNED}
        if cache_salt is not None:      # isolates this episode's prefix-cache entries
            p["cache_salt"] = cache_salt
        return p

    def _once(self, messages, cache_salt):
        body = json.dumps(self._payload(messages, cache_salt)).encode()
        req = urllib.request.Request(self.endpoint + "/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        r = json.loads(urllib.request.urlopen(req, timeout=self.timeout).read())
        return r["choices"][0]["message"]["content"]

    def generate(self, messages, cache_salt=None):
        """messages -> assistant text. Retries transient network errors and HTTP 5xx with
        exponential backoff. HTTP 4xx (including a 400 context overflow) is raised to the caller."""
        delay, last = 1.0, None
        for attempt in range(self.max_retries + 1):
            try:
                return self._once(messages, cache_salt)
            except urllib.error.HTTPError as e:      # subclass of URLError, so it must come first
                last = e
                if e.code < 500:
                    raise
            except (urllib.error.URLError,) + TRANSIENT as e:
                last = e
            if attempt < self.max_retries:
                time.sleep(delay); delay = min(delay * 2, 15.0)
        raise last


class ReplayEngine:
    """Returns pre-recorded assistant turns in order instead of calling a model. Used to check a
    local sandbox setup against a recorded episode, and in tests."""

    def __init__(self, raws, sampling=None):
        self.raws = list(raws)
        self.i = 0
        self.sampling = sampling or resolved_sampling()
        self.requests = []              # (messages snapshot, cache_salt) per call

    def generate(self, messages, cache_salt=None):
        self.requests.append(([dict(m) for m in messages], cache_salt))
        if self.i >= len(self.raws):
            raise RuntimeError("replay exhausted after %d turns" % len(self.raws))
        text = self.raws[self.i]
        self.i += 1
        return text
