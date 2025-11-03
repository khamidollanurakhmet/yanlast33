#!/usr/bin/env python3
import argparse
import io
import json
import pickle
import re
import sys
from typing import Dict, List, Tuple, Any, Optional

try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False


Option = Tuple[str, str]  # (letter, text)


def load_image(image_blob: Any) -> Optional["Image.Image"]:
    """
    Best-effort conversion of the blob to a PIL Image.
    Returns None if PIL is unavailable or conversion fails (baseline does not use the image).
    """
    if not PIL_AVAILABLE:
        return None

    try:
        # Case 1: bytes-like (common for pickled image bytes)
        if isinstance(image_blob, (bytes, bytearray, memoryview)):
            bio = io.BytesIO(image_blob)
            img = Image.open(bio).convert("RGB")
            return img

        # Case 2: a file path string
        if isinstance(image_blob, str):
            img = Image.open(image_blob).convert("RGB")
            return img

        # Case 3: numpy array-like
        try:
            import numpy as np  # local import
            if isinstance(image_blob, np.ndarray):
                # shape HWC or CHW
                arr = image_blob
                if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):  # CHW -> HWC
                    arr = arr.transpose(1, 2, 0)
                if arr.ndim == 2:
                    mode = "L"
                elif arr.shape[2] == 3:
                    mode = "RGB"
                elif arr.shape[2] == 4:
                    mode = "RGBA"
                else:
                    return None
                img = Image.fromarray(arr.astype("uint8"), mode=mode)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                return img
        except Exception:
            pass
    except Exception:
        return None

    return None


def parse_options(question_text: str) -> List[Option]:
    """
    Parse options from the question text. Supports A., B., C., ... on separate lines or inline.
    Returns a list of (letter, text).
    """
    # Normalize line endings
    text = question_text.replace("\r\n", "\n").replace("\r", "\n")

    # Try regex to capture segments starting with "A. ...", "B. ...", etc.
    # This regex finds a letter dot, captures until next letter-dot or end.
    # Supports Latin and Cyrillic uppercase letters for robustness.
    pattern = re.compile(r'([A-ZА-Я])\.\s*((?:(?!\n[A-ZА-Я]\.).)*)', re.DOTALL)
    matches = pattern.findall(text)

    options: List[Option] = []
    if matches:
        for letter, opt_text in matches:
            cleaned = opt_text.strip()
            options.append((letter, cleaned))
        return options

    # Fallback: split lines, detect ones that start with "X. "
    options = []
    for line in text.split("\n"):
        m = re.match(r'^\s*([A-ZА-Я])\.\s*(.+)$', line.strip())
        if m:
            options.append((m.group(1), m.group(2).strip()))
    return options


def normalize_equation(s: str) -> str:
    """
    Normalize common subscript/superscript and whitespace for simple equality/inequality checks.
    """
    s = s.lower()
    # Replace common unicode subscripts for 1 and 2
    s = s.replace("₁", "1").replace("₂", "2")
    # Remove spaces and unicode spaces
    s = re.sub(r'\s+', '', s)
    # Normalize comparison symbols
    s = s.replace("＝", "=").replace("≠", "!=")
    s = s.replace("≤", "<=").replace("≥", ">=")
    s = s.replace("⩽", "<=").replace("⩾", ">=")
    s = s.replace("⩾", ">=").replace("⩽", "<=")
    return s


def baseline_heuristic(question_text: str, options: List[Option]) -> str:
    """
    A very simple heuristic to select answers:
    - If the question looks like liquid levels h1/h2 and there's an option stating h1 = h2, choose it (usually true for connected vessels in equilibrium).
    - Otherwise, default to picking a single option 'A' to produce a valid output format.
    This is a placeholder to be replaced by a proper vision-language model.
    """
    # Liquid levels heuristic
    normalized_q = normalize_equation(question_text)
    looks_like_liquid = any(tag in normalized_q for tag in ["h1", "h2"]) and "level" in question_text.lower()

    # Scan options for equality/inequality patterns
    norm_opts = [(ltr, normalize_equation(txt)) for (ltr, txt) in options]
    has_h1_gt_h2 = any(("h1>h2" in t) for _, t in norm_opts)
    has_h2_gt_h1 = any(("h2>h1" in t) for _, t in norm_opts)
    has_h1_eq_h2 = [(ltr, t) for (ltr, t) in norm_opts if ("h1=h2" in t)]

    if looks_like_liquid and has_h1_eq_h2:
        # Choose the equality option (often labeled 'C' in examples)
        return "".join(sorted({ltr for (ltr, _) in has_h1_eq_h2}))

    # If "cannot be determined" option exists and both inequalities exist but no equality, prefer "cannot be determined".
    has_cannot_determine = [ltr for (ltr, txt) in options if "cannot be determined" in txt.lower()]
    if has_cannot_determine and has_h1_gt_h2 and has_h2_gt_h1 and not has_h1_eq_h2:
        return "".join(sorted(set(has_cannot_determine)))

    # Fallback: return the first option to keep output valid
    if options:
        return options[0][0]

    # If no options parsed, return empty (should not happen with valid input)
    return ""


def process_item(item: Dict[str, Any]) -> Dict[str, Any]:
    rid = item.get("rid")
    question = item.get("question", "")
    image_blob = item.get("image", None)

    # Load image if needed in future (currently unused by baseline)
    _ = load_image(image_blob)

    options = parse_options(question)
    answer = baseline_heuristic(question, options)
    return {"rid": rid, "answer": answer}


def main():
    parser = argparse.ArgumentParser(description="Baseline solution for ML Olympiad multimodal Q/A (geometry/physics diagrams).")
    parser.add_argument("--input", "-i", type=str, default="input.pickle", help="Path to input.pickle")
    parser.add_argument("--output", "-o", type=str, default="output.json", help="Path to output.json")
    args = parser.parse_args()

    try:
        with open(args.input, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Failed to read input pickle {args.input}: {e}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(data, list):
        print("Input pickle must contain a list of dicts.", file=sys.stderr)
        sys.exit(1)

    results: List[Dict[str, Any]] = []
    for item in data:
        try:
            res = process_item(item)
            # Ensure required keys
            if "rid" not in res or "answer" not in res:
                raise ValueError("Result missing 'rid' or 'answer'.")
            results.append(res)
        except Exception as e:
            # In case of a single item failure, attempt to return an empty answer for that rid
            rid = item.get("rid", None)
            print(f"Error processing rid={rid}: {e}", file=sys.stderr)
            if rid is not None:
                results.append({"rid": rid, "answer": ""})

    try:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False)
    except Exception as e:
        print(f"Failed to write output JSON {args.output}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()