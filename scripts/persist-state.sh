#!/usr/bin/env bash
# Persist bot state to a dedicated 'state' branch (survives workflow runs).
set -euo pipefail

BRANCH="${STATE_BRANCH:-state}"
FILES=(state.json trans_cache.db)

case "${1:-}" in
  restore)
    if ! git fetch origin "$BRANCH" --depth=1 2>/dev/null; then
      echo "No '$BRANCH' branch yet — first run."
      exit 0
    fi
    for f in "${FILES[@]}"; do
      if git cat-file -e "origin/$BRANCH:$f" 2>/dev/null; then
        git show "origin/$BRANCH:$f" > "$f"
        echo "Restored $f ($(stat -c%s "$f") bytes)"
      fi
    done
    ;;

  save)
    git config user.name  "github-actions[bot]"
    git config user.email "github-actions[bot]@users.noreply.github.com"

    WT=$(mktemp -d)
    trap 'git worktree remove -f "$WT" 2>/dev/null || rm -rf "$WT"' EXIT

    git fetch origin "$BRANCH" --depth=1 2>/dev/null || true

    if git rev-parse --verify "origin/$BRANCH" >/dev/null 2>&1; then
      git worktree add -B "$BRANCH" "$WT" "origin/$BRANCH"
    else
      git worktree add --detach "$WT" HEAD
      (
        cd "$WT"
        git checkout --orphan "$BRANCH"
        git rm -rf . >/dev/null 2>&1 || true
        touch .gitkeep
        git add .gitkeep
        git commit -m "init: state branch [skip ci]"
      )
    fi

    for f in "${FILES[@]}"; do
      [[ -f "$f" ]] && cp "$f" "$WT/" || true
    done

    (
      cd "$WT"
      git add -A
      if git diff --cached --quiet; then
        echo "No state changes to save."
      else
        git commit -m "state: $(date -u +%FT%TZ) [skip ci]"
        git push origin "$BRANCH"
        echo "State saved to '$BRANCH'."
      fi
    )
    ;;

  *)
    echo "Usage: $0 {restore|save}" >&2
    exit 2
    ;;
esac