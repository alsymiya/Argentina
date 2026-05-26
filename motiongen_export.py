"""
motiongen_export.py — Convert a saved linkage config to MotionGen JSON.

Takes the .txt produced by the visualizer's "Save Config" button, strips out
the kinematic configuration, and produces the JSON blob that MotionGen
accepts when pasted into its "Load From JSON" dialog.

Usage
-----
    python motiongen_export.py config.txt              # copy to clipboard + print
    python motiongen_export.py config.txt --out file.json
    python motiongen_export.py config.txt --no-clip    # only print

On Windows, the JSON text is copied to the clipboard automatically (uses
``clip.exe``). On macOS / Linux the script falls back to ``pbcopy`` /
``xclip`` / ``xsel`` if available; otherwise it just prints the text and the
user copies it manually.

Format
------
The reference MotionGen JSON from the wishlist is reproduced exactly: same
top-level keys (``name``, ``version``, ``mechanism``, ``motionSynthesis``,
``geometry``, ``shapes``, ``images``, ``settings``); same per-joint and
per-link fields; ground link is colored grey with ``isGround=true``; non-
ground links are colored green; coupler triangles are merged into a single
link with multiple ``jointIds``.

Actuator
--------
The Argentina demo's Python simulator uses a minimal motor descriptor
``motor = [pivot, tip]``: joint 1 (the tip) is driven around joint 0 (the
pivot) by the global theta angle, measured from the positive x-axis. In
MotionGen the same idea takes three joints — the actuator measures the
angle between vectors ``<at, from>`` and ``<at, to>``:

    at   = the pivot (motor[0])
    from = a reference ground joint (the "other" fixed pivot)
    to   = the driven tip (motor[1])

Because the Python simulator already pins ``from`` along the world +x axis
by construction (``cos(theta)``, ``sin(theta)``), we drop that field in the
Python data model but reconstruct it on export by picking the first
``fixed_nodes`` entry that isn't ``motor[0]``. The Argentina demo is a
four-bar — exactly one actuator and one such reference joint — so this
mapping is unambiguous.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import uuid

import numpy as np


# ---------------------------------------------------------------------------
# Config I/O — matches the format the visualizer writes
# ---------------------------------------------------------------------------

def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = [ln for ln in f.readlines() if not ln.lstrip().startswith('#')]
    cfg = json.loads("".join(lines))
    return (
        np.asarray(cfg["JJ"], dtype=int),
        np.asarray(cfg["PSlice"], dtype=float),
        list(cfg["motor"]),
        list(cfg["fixed_nodes"]),
        int(cfg["path_node"]),
    )


# ---------------------------------------------------------------------------
# JJ-matrix → MotionGen link list
# ---------------------------------------------------------------------------

def find_rigid_bodies(JJ):
    """Return a list of joint-index lists, one per rigid body link.

    A "link" in MotionGen can carry 2+ joints (the coupler body of a 4-bar
    with a moving point is the canonical example). The rule here:

      - Any maximal clique in JJ of size >= 3 is one rigid body.
      - Edges not covered by any such clique are binary links of their own.

    Cliques that share an edge (i.e., share two joints) are merged before
    being emitted — that way a planar triangle plus a fourth coplanar
    joint connected to two of its vertices becomes a single 4-joint body.
    """
    JJ = np.asarray(JJ, dtype=int)
    N = JJ.shape[0]

    edges = set()
    for i in range(N):
        for j in range(i + 1, N):
            if JJ[i, j]:
                edges.add((i, j))

    # All triangles.
    bodies = []
    for i in range(N):
        for j in range(i + 1, N):
            if not JJ[i, j]:
                continue
            for k in range(j + 1, N):
                if JJ[i, k] and JJ[j, k]:
                    bodies.append({i, j, k})

    # Merge bodies that share an edge (2 common joints).
    changed = True
    while changed:
        changed = False
        out = []
        used = [False] * len(bodies)
        for i in range(len(bodies)):
            if used[i]:
                continue
            cur = set(bodies[i])
            used[i] = True
            for j in range(i + 1, len(bodies)):
                if used[j]:
                    continue
                if len(cur & bodies[j]) >= 2:
                    cur |= bodies[j]
                    used[j] = True
                    changed = True
            out.append(cur)
        bodies = out

    # Which edges are already covered?
    covered = set()
    for body in bodies:
        bl = sorted(body)
        for ii in range(len(bl)):
            for jj in range(ii + 1, len(bl)):
                covered.add((bl[ii], bl[jj]))

    # Add remaining binary links.
    links = [sorted(body) for body in bodies]
    for e in edges:
        if e not in covered:
            links.append(list(e))
    return links


# ---------------------------------------------------------------------------
# MotionGen JSON builder
# ---------------------------------------------------------------------------

NORMAL_LINK_COLOR = "rgba(104, 211, 145, 0.5)"
GROUND_LINK_COLOR = "rgba(220, 220, 220, 0.5)"
CURVE_COLOR       = "rgba(99, 179, 237, 1)"
ACTUATOR_COLOR    = "rgba(183, 148, 244, 0.5)"


def _joint_dict(jid, x, y):
    return {
        "id": jid,
        "x": float(x),
        "y": float(y),
        "curveColor": CURVE_COLOR,
        "isShowCurve": True,
        "isShowArrows": False,
        "isWelded": False,
        "isSelected": True,
        "isLocked": False,
    }


def _link_dict(lid, joint_ids, is_ground):
    return {
        "id": lid,
        "paths": None,
        "plane": 0,
        "jointIds": list(joint_ids),
        "slotIds": [],
        "color": GROUND_LINK_COLOR if is_ground else NORMAL_LINK_COLOR,
        "trace": 0,
        "isGround": bool(is_ground),
        "isSelected": True,
        "isHidden": False,
    }


def _actuator_dict(aid, at_jid, from_jid, to_jid,
                   vmin=0, vmax=360, velocity=10):
    """One rotary MotionGen actuator. See module docstring for the mapping."""
    return {
        "id": aid,
        "type": "rotary",
        "at": at_jid,
        "from": from_jid,
        "to": to_jid,
        "min": vmin,
        "max": vmax,
        "velocity": velocity,
        "color": ACTUATOR_COLOR,
        "isSelected": False,
    }


def build_motiongen_json(JJ, PSlice, motor, fixed_nodes, path_node):
    """Return the dict that gets serialized as the MotionGen JSON."""
    JJ = np.asarray(JJ, dtype=int)
    PSlice = np.asarray(PSlice, dtype=float)
    N = JJ.shape[0]

    joint_ids = [str(uuid.uuid4()) for _ in range(N)]
    joints = [_joint_dict(joint_ids[k], PSlice[k, 0], PSlice[k, 1]) for k in range(N)]

    links = []
    for body in find_rigid_bodies(JJ):
        links.append(_link_dict(str(uuid.uuid4()),
                                [joint_ids[k] for k in body],
                                is_ground=False))
    # Ground link connects all fixed pivots (if 2+).
    if len(fixed_nodes) >= 2:
        links.append(_link_dict(str(uuid.uuid4()),
                                [joint_ids[k] for k in fixed_nodes],
                                is_ground=True))

    # Single rotary actuator. The Python simulator's motor=[pivot, tip]
    # collapses MotionGen's three-joint "<at, from> vs <at, to>" angle into
    # an absolute world-frame theta around joint 0; on export we reinstate
    # the missing `from` by picking another fixed pivot.
    actuators = []
    other_fixed = [k for k in fixed_nodes if k != motor[0]]
    if other_fixed:
        actuators.append(_actuator_dict(
            str(uuid.uuid4()),
            at_jid=joint_ids[motor[0]],
            from_jid=joint_ids[other_fixed[0]],
            to_jid=joint_ids[motor[1]],
        ))

    return {
        "name": "motiongen",
        "version": "1.1.5",
        "mechanism": {
            "mode": "standard",
            "joints": joints,
            "slots": [],
            "links": links,
            "cylinders": [],
            "actuators": actuators,
            "measurements": [],
        },
        "motionSynthesis": {"points": [], "poses": [], "constraints": []},
        "geometry": {
            "points": [], "lines": [], "circles": [],
            "polygons": [], "splines": [], "distances": [], "angles": [],
        },
        "shapes": [],
        "images": [],
        "settings": {"linearUnit": "in", "angularUnit": "rad"},
    }


# ---------------------------------------------------------------------------
# Clipboard handling
# ---------------------------------------------------------------------------

def _win32_clipboard_set(text):
    """Set the Windows clipboard via the Win32 API (CF_UNICODETEXT).

    We can't use ``clip.exe`` because piping ``text.encode('utf-16')`` into
    it leaves the UTF-16 BOM (``\\xff\\xfe``) as a literal ``U+FEFF``
    character at the start of whatever pastes — and MotionGen's strict
    JSON parser rejects a BOM-prefixed string. Calling ``SetClipboardData``
    directly with CF_UNICODETEXT lets us hand Windows the pure text bytes
    with no leading BOM, exactly the way Notepad / browsers do it.

    Returns True on success, False (silently) on any API failure so the
    caller can fall back to other methods.
    """
    try:
        import ctypes
        from ctypes import wintypes
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        CF_UNICODETEXT = 13
        GMEM_MOVEABLE = 0x0002

        kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
        kernel32.GlobalAlloc.restype = wintypes.HGLOBAL
        kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
        kernel32.GlobalLock.restype = wintypes.LPVOID
        kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
        kernel32.GlobalFree.argtypes = [wintypes.HGLOBAL]
        user32.OpenClipboard.argtypes = [wintypes.HWND]
        user32.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]
        user32.SetClipboardData.restype = wintypes.HANDLE

        # Encode WITHOUT BOM, with a trailing UTF-16 null terminator.
        data = text.encode('utf-16-le') + b'\x00\x00'
        h_mem = kernel32.GlobalAlloc(GMEM_MOVEABLE, len(data))
        if not h_mem:
            return False
        p_mem = kernel32.GlobalLock(h_mem)
        if not p_mem:
            kernel32.GlobalFree(h_mem)
            return False
        ctypes.memmove(p_mem, data, len(data))
        kernel32.GlobalUnlock(h_mem)

        if not user32.OpenClipboard(0):
            kernel32.GlobalFree(h_mem)
            return False
        try:
            user32.EmptyClipboard()
            if not user32.SetClipboardData(CF_UNICODETEXT, h_mem):
                kernel32.GlobalFree(h_mem)
                return False
            # On success the clipboard owns h_mem and will free it later.
        finally:
            user32.CloseClipboard()
        return True
    except Exception:
        return False


def copy_to_clipboard(text):
    """Best-effort clipboard write. Returns the platform/tool used, or None.

    On Windows we use the Win32 API directly (no BOM in the clipboard
    payload). On macOS / Linux we shell out to the standard tools. The
    cross-platform ``pyperclip`` is the final fallback if installed.
    """
    # Windows — Win32 API, BOM-free.
    if sys.platform.startswith('win'):
        if _win32_clipboard_set(text):
            return 'windows:Win32 SetClipboardData (CF_UNICODETEXT)'
    # macOS
    if sys.platform == 'darwin' and shutil.which('pbcopy'):
        subprocess.run(['pbcopy'], input=text.encode('utf-8'), check=True)
        return 'macos:pbcopy'
    # Linux — try xclip then xsel
    if shutil.which('xclip'):
        subprocess.run(['xclip', '-selection', 'clipboard'],
                       input=text.encode('utf-8'), check=True)
        return 'linux:xclip'
    if shutil.which('xsel'):
        subprocess.run(['xsel', '--clipboard', '--input'],
                       input=text.encode('utf-8'), check=True)
        return 'linux:xsel'
    # pyperclip fallback (cross-platform if installed)
    try:
        import pyperclip  # type: ignore
        pyperclip.copy(text)
        return 'pyperclip'
    except Exception:
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(
        description="Convert a saved linkage config (.txt) to MotionGen JSON."
    )
    p.add_argument('config', type=str, help="Path to config .txt from the visualizer.")
    p.add_argument('--out', type=str, default=None,
                   help="Optional path to also write the JSON to a file.")
    p.add_argument('--no-clip', action='store_true',
                   help="Skip copying to clipboard; just print.")
    p.add_argument('--minified', action='store_true',
                   help="Single-line JSON (MotionGen accepts both).")
    args = p.parse_args(argv)

    if not os.path.exists(args.config):
        print(f"error: config not found: {args.config}", file=sys.stderr)
        return 2

    JJ, PSlice, motor, fixed_nodes, path_node = load_config(args.config)
    payload = build_motiongen_json(JJ, PSlice, motor, fixed_nodes, path_node)
    text = (json.dumps(payload, separators=(",", ":")) if args.minified
            else json.dumps(payload, indent=2))

    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"wrote {args.out}")

    if not args.no_clip:
        used = copy_to_clipboard(text)
        if used:
            n_act = len(payload['mechanism']['actuators'])
            print(f"copied to clipboard via {used} "
                  f"({len(text):,} chars, {n_act} actuator{'s' if n_act != 1 else ''}). "
                  f"Paste it into MotionGen's Load-From-JSON dialog.")
        else:
            print("(could not copy to clipboard — printing JSON below; "
                  "install pyperclip or xclip for auto-copy)")
            print(text)
    else:
        print(text)
    return 0


if __name__ == '__main__':
    sys.exit(main())
