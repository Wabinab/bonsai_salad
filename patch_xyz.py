#!/usr/bin/env python3
"""
Patches xyz-llama's tools/server/server-context.cpp so the xyz drafter and the
multimodal projector can be loaded together.

usage: patch_xyz.py <src_dir> [apply_B: 0|1]

Every anchor must match EXACTLY ONCE, otherwise the build fails. That is on
purpose: if you bump REF and upstream moved the code, you want a loud failure,
not a silently unpatched binary.

Patch 0 : lets the projector load when a drafter is configured, but only when
          env XYZ_ALLOW_MMPROJ_WITH_DRAFTER is set (default behaviour unchanged).
Patch A : get_n_draft_max() returns 0 for a slot whose context contains media,
          so that slot runs plain decode. Text-only slots keep drafting.
Patch B : (optional, off by default) stop treating "projector loaded" as "slot has
          media" in prepare_speculative_draft(), recovering the fast path for
          text-only requests. Safe only because patch A guarantees anything
          reaching those lines has no media.
"""
import pathlib
import sys

src_dir = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/src")
apply_b = (sys.argv[2] if len(sys.argv) > 2 else "0") == "1"

path = src_dir / "tools/server/server-context.cpp"
text = path.read_text()


def sub(old: str, new: str, name: str) -> None:
    global text
    n = text.count(old)
    if n != 1:
        sys.exit(f"PATCH FAILED [{name}]: expected exactly 1 match, found {n}")
    text = text.replace(old, new)
    print(f"patch [{name}]: applied")


# ---- Patch 0: the load-time guard -------------------------------------------
sub(
    "if (has_spec && !params_base.mmproj.path.empty()) {",
    'if (has_spec && !params_base.mmproj.path.empty() && !getenv("XYZ_ALLOW_MMPROJ_WITH_DRAFTER")) {',
    "0-guard",
)

# ---- Patch A: no drafting for slots that hold media -------------------------
# find_next_media_chunk() uses upper_bound(idx), i.e. strictly AFTER idx, so a
# chunk starting at index 0 is invisible to it. tokens[0] == LLAMA_TOKEN_NULL
# covers that case (media positions are stored as LLAMA_TOKEN_NULL).
sub(
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

# ---- Patch B: optional text-only fast path ----------------------------------
if apply_b:
    sub(
        "if (slot.prompt.tokens.has_mtmd) {\n"
        "            slot.spec_prompt = slot.prompt.tokens.get_text_tokens();",
        "if (false) {\n"
        "            slot.spec_prompt = slot.prompt.tokens.get_text_tokens();",
        "B1-spec-prompt",
    )
    sub(
        "if (spec_early_verify_ok && !slot.prompt.tokens.has_mtmd) {",
        "if (spec_early_verify_ok) {",
        "B2-early-verify",
    )
else:
    print("patch [B]: skipped (APPLY_B=0)")

path.write_text(text)
print("done")