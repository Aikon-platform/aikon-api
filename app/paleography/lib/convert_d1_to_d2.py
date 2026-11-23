#!/usr/bin/env python3
"""Converter D1 -> D2 pour M4M. D1: <d1>/<folio>/<ligne>.png + .txt ; D2: annotation.json + images/<folio>/."""
import argparse, json, random, shutil, sys
from collections import Counter
from pathlib import Path

def doc_key(stem):
    return "_".join(stem.split("_")[:2])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d1", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    # PATCH 1 : --script devient optionnel (ecriture par defaut) ; --script-map donne l'ecriture par folio
    ap.add_argument("--script", default=None, help="ecriture par defaut (tous les folios sans entree dans --script-map)")
    ap.add_argument("--script-map", type=Path, default=None, help="JSON {folio: ecriture}")
    ap.add_argument("--val-ratio", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-train-lines", type=int, default=4)
    a = ap.parse_args()
    if a.out.exists():
        print(f"[ERREUR] {a.out} existe deja", file=sys.stderr); return 1
    if not 0.0 <= a.val_ratio < 1.0:
        print("[ERREUR] --val-ratio dans [0,1)", file=sys.stderr); return 1
    # PATCH 3 : code 1 propre si --d1 absent (avant : exception iterdir)
    if not a.d1.is_dir():
        print(f"[ERREUR] {a.d1} n'est pas un dossier", file=sys.stderr); return 1
    # PATCH 1 : chargement de la map folio -> ecriture
    smap = {}
    if a.script_map:
        smap = json.loads(a.script_map.read_text(encoding="utf-8"))
        if not isinstance(smap, dict):
            print("[ERREUR] --script-map doit etre un objet JSON {folio: ecriture}", file=sys.stderr); return 1
    folders = sorted(p for p in a.d1.iterdir() if p.is_dir())
    if not folders:
        print(f"[ERREUR] aucun sous-dossier dans {a.d1}", file=sys.stderr); return 1
    # PATCH 2 : le rng global est supprime (il partageait son etat entre folios)
    ann, errors, warnings = {}, [], []
    for folder in folders:
        # PATCH 1 : ecriture du folio ; sans ecriture connue, le folio est saute (code retour 2)
        script = smap.get(folder.name, a.script)
        if script is None:
            script = ""  # pas d'ecriture fournie: champ vide (M4M classe alors en "document")
        lines = []
        for img in sorted(folder.iterdir()):
            suf = img.suffix.lower()
            if suf in {".tif", ".tiff"}:
                errors.append(f"{img}: format non pris en charge (tif/tiff)"); continue
            if suf not in {".png", ".jpg", ".jpeg"}:
                continue
            txt = img.with_suffix(".txt")
            if not txt.exists():
                errors.append(f"{img}: pas de .txt"); continue
            if not img.stem.startswith(folder.name):
                errors.append(f"{img.name}: dossier pas prefixe"); continue
            if doc_key(img.stem) != folder.name:
                errors.append(f"{img.name}: cle-doc != dossier"); continue
            label = txt.read_text(encoding="utf-8-sig").strip()
            if not label:
                errors.append(f"{txt}: vide"); continue
            if "\n" in label:
                errors.append(f"{txt}: multi-ligne"); continue
            if img.name in ann:
                errors.append(f"{img.name}: doublon"); continue
            lines.append((img, label))
        # PATCH 2 : seed derive de (--seed, nom du folio) => le split d'un folio ne depend plus des autres.
        # Une str est seedee de facon deterministe (pas affectee par PYTHONHASHSEED).
        rng = random.Random(f"{a.seed}:{folder.name}")
        order = list(range(len(lines))); rng.shuffle(order)
        n_val = int(len(lines) * a.val_ratio); val_idx = set(order[:n_val])
        for i, (img, label) in enumerate(lines):
            dst = a.out / "images" / folder.name / img.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, dst)
            ann[img.name] = {"label": label, "split": "val" if i in val_idx else "train", "script": script}
        if lines and len(lines) - n_val < a.min_train_lines:
            warnings.append(f"{folder.name}: {len(lines)-n_val} lignes train (< {a.min_train_lines})")
    if not ann:
        print("[ERREUR] aucune ligne valide", file=sys.stderr)
        for e in errors: print("   -", e, file=sys.stderr)
        return 1
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "annotation.json").write_text(json.dumps(ann, ensure_ascii=False, indent=2), encoding="utf-8")
    docs = Counter(doc_key(Path(k).stem) for k in ann)
    print(f"[OK] {len(ann)} lignes ; splits {dict(Counter(v['split'] for v in ann.values()))} ; {len(docs)} folio(s)")
    for w in warnings: print("[WARN]", w)
    for e in errors: print("[ERREUR-LIGNE]", e)
    return 0 if not errors else 2

if __name__ == "__main__":
    sys.exit(main())
