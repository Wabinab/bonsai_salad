#!/usr/bin/env python3
"""
Patches xyz-llama so the xyz drafter and the multimodal projector can be loaded together.

usage: patch_xyz.py <src_dir> [apply_B: 0|1]

Every anchor must match EXACTLY ONCE, otherwise the build fails. That is on purpose:
if you bump REF and upstream moved the code, you want a loud failure, not a silently
unpatched binary.

server-context.cpp
  0-guard : lets the projector load when a drafter is configured, but only when env
            XYZ_ALLOW_MMPROJ_WITH_DRAFTER is set (default behaviour unchanged).
  A       : get_n_draft_max() returns 0 for a slot whose context contains media, so that
            slot runs plain decode. Text-only slots keep drafting.
  B1/B2   : (optional, APPLY_B=1) stop treating "projector loaded" as "slot has media"
            in prepare_speculative_draft(): skips the per-round copy of the whole text
            token list and re-enables the early-verify fast path for text-only requests.
            Needs patch C (B1 calls get_tokens()).

server-common.cpp
  C       : REQUIRED. get_tokens() and set_token() asserted !has_mtmd, and has_mtmd is
            true for EVERY slot once a projector is loaded (even text-only requests).
            That made prepare_speculative_draft() abort on the first text request.
            The assertions now check for actual media (map_idx_to_media.empty()).
"""
import pathlib
import sys

src_dir = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/src")
apply_b = (sys.argv[2] if len(sys.argv) > 2 else "0") == "1"


class Patcher:
    def __init__(self, rel: str):
        self.path = src_dir / rel
        self.text = self.path.read_text()

    def sub(self, old: str, new: str, name: str) -> None:
        n = self.text.count(old)
        if n != 1:
            sys.exit(f"PATCH FAILED [{name}] in {self.path.name}: expected exactly 1 match, found {n}")
        self.text = self.text.replace(old, new)
        print(f"patch [{name}]: applied")

    def save(self) -> None:
        self.path.write_text(self.text)


# ============================ server-common.cpp ==============================
common = Patcher("tools/server/server-common.cpp")

# C1: get_tokens(): assert on actual media instead of the "projector loaded" flag
common.sub(
    "const llama_tokens & server_tokens::get_tokens() const {\n"
    "    GGML_ASSERT(!has_mtmd);",
    "const llama_tokens & server_tokens::get_tokens() const {\n"
    "    GGML_ASSERT(map_idx_to_media.empty()); // xyz fork: no media chunks, instead of 'no projector loaded'",
    "C1-get_tokens",
)

# C2: the other speculative-decoding accessor (set_token), same reasoning
common.sub(
    "GGML_ASSERT(!has_mtmd); // only allow this if mtmd is disabled",
    "GGML_ASSERT(map_idx_to_media.empty()); // xyz fork: only allow this if there is no media",
    "C2-set_token",
)
common.save()

# ============================ server-context.cpp ============================
ctx = Patcher("tools/server/server-context.cpp")

# 0: the load-time guard
ctx.sub(
    "if (has_spec && !params_base.mmproj.path.empty()) {",
    'if (has_spec && !params_base.mmproj.path.empty() && !getenv("XYZ_ALLOW_MMPROJ_WITH_DRAFTER")) {',
    "0-guard",
)

# A: no drafting for slots that hold media.
# find_next_media_chunk() uses upper_bound(idx), i.e. strictly AFTER idx, so a chunk
# starting at index 0 is invisible to it. tokens[0] == LLAMA_TOKEN_NULL covers that case.
ctx.sub(
    "// determine the max draft that fits the current slot state",
    "// xyz fork: a slot whose context holds media (image/audio) runs as plain decode, no drafting\n"
    "        if (prompt.tokens.has_mtmd &&\n"
    "            (prompt.tokens.find_next_media_chunk(0).first != nullptr ||\n"
    "             (!prompt.tokens.empty() && prompt.tokens[0] == LLAMA_TOKEN_NULL))) {\n"
    "            return 0;\n"
    "        }\n"
    "\n"
    "        // determine the max draft that fits the current slot state",
    "A-no-draft-with-media",
)

# B: optional text-only fast path
if apply_b:
    ctx.sub(
        "if (slot.prompt.tokens.has_mtmd) {\n"
        "            slot.spec_prompt = slot.prompt.tokens.get_text_tokens();",
        "if (false) {\n"
        "            slot.spec_prompt = slot.prompt.tokens.get_text_tokens();",
        "B1-spec-prompt",
    )
    ctx.sub(
        "if (spec_early_verify_ok && !slot.prompt.tokens.has_mtmd) {",
        "if (spec_early_verify_ok) {",
        "B2-early-verify",
    )
else:
    print("patch [B]: skipped (APPLY_B=0)")

ctx.save()
print("done")