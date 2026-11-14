#!/usr/bin/env bash
#
# Publish a development wheel to the rolling `nightly` prerelease of a repository.
#
#   GITHUB_REPOSITORY=FlashML-org/FreeToken-metal scripts/publish-nightly.sh DIST_DIR
#
# DIST_DIR holds the one wheel scripts/build-wheels.sh built from NIGHTLY_COMMIT (default:
# HEAD). The prerelease is created on first use. Each publish deletes the previous assets,
# uploads the wheel and then engine-macos_arm64.json (the wheel's URL, sha256 and size),
# rewrites the release notes, and moves the `nightly` tag to the commit last, so the tag never
# points at a commit whose wheel is not there. Old assets go first: a reader that takes the
# first matching asset never gets an old wheel next to a new one. GitHub calls are retried.
# Needs gh with write access to the repository; GITHUB_REPOSITORY is required, since this
# deletes assets and force-moves a tag.
set -euo pipefail
shopt -s nullglob

TAG=nightly
MANIFEST=engine-macos_arm64.json

say() { printf '==> %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

# gh with three tries: a transient 5xx in the middle of a publish would otherwise leave the
# release without a wheel until the next build.
gh_retry() {
  local attempt
  for attempt in 1 2 3; do
    if gh "$@"; then
      return 0
    fi
    if [ "$attempt" != 3 ]; then
      printf 'gh %s failed, retrying\n' "$1 $2 $3" >&2
      sleep $((attempt * 5))
    fi
  done
  return 1
}

[ $# -eq 1 ] || die "usage: GITHUB_REPOSITORY=owner/repo scripts/publish-nightly.sh DIST_DIR"
DIST="$1"
repo="${GITHUB_REPOSITORY:-}"
[ -n "$repo" ] || die "set GITHUB_REPOSITORY to the repository to publish to; there is no default"
command -v gh >/dev/null || die "gh is not on PATH"
commit="$(git rev-parse --verify "${NIGHTLY_COMMIT:-HEAD}^{commit}")"

wheels=("$DIST"/*.whl)
[ ${#wheels[@]} -eq 1 ] || die "expected one wheel in $DIST, got ${#wheels[@]}"
wheel="${wheels[0]}"
name="$(basename "$wheel")"
pattern='^freetoken_metal-([^-]+\+g([0-9a-f]{7,}))-(cp3[0-9]+)-cp3[0-9]+-macosx_[0-9_]+_arm64\.whl$'
[[ "$name" =~ $pattern ]] \
  || die "$name is not a stamped freetoken-metal wheel for macOS on Apple Silicon"
version="${BASH_REMATCH[1]}"
stamp="${BASH_REMATCH[2]}"
abi="${BASH_REMATCH[3]}"
[[ "$commit" == "$stamp"* ]] || die "$name was built from $stamp, not from $commit"
short="${commit:0:${#stamp}}"
python="3.${abi#cp3}"

url="https://github.com/$repo/releases/download/$TAG/${name//+/%2B}"
sha256="$(shasum -a 256 "$wheel" | cut -d' ' -f1)"
size="$(wc -c < "$wheel" | tr -d ' ')"
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT
# FreeToken's manifest shape (scripts/publish-wheels.sh there) without kernel_cache and cuda:
# ftmetal ships one wheel and compiles its Metal kernels at run time.
cat > "$scratch/$MANIFEST" <<EOF
{
  "schema": 1,
  "channel": "$TAG",
  "platform": "macos_arm64",
  "version": "$version",
  "commit": "$commit",
  "published": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "python": "$abi",
  "runtime": {
    "name": "$name",
    "url": "$url",
    "sha256": "$sha256",
    "size": $size
  }
}
EOF

notes="Rolling development build of \`main\` at [$short](https://github.com/$repo/commit/$commit)
($version), for Python $python on macOS 15 or later with Apple Silicon. Not a release: every new
build replaces it.

\`\`\`bash
pip install \"freetoken-metal @ $url\"
\`\`\`

sha256 \`$sha256\`. \`$MANIFEST\` names the same wheel for programs."

# Only a 404 means the release does not exist; any other error (auth, network) stops here.
if ! status="$(gh api "repos/$repo/releases/tags/$TAG" --jq .id 2>&1)"; then
  case "$status" in
    *"HTTP 404"*)
      say "creating the $TAG prerelease at $short"
      gh_retry release create "$TAG" --repo "$repo" --prerelease --target "$commit" \
        --title "Nightly" --notes "$notes"
      ;;
    *) die "cannot read the $TAG release of $repo: $status" ;;
  esac
fi

say "deleting the previous assets"
assets="$(gh_retry release view "$TAG" --repo "$repo" --json assets --jq '.assets[].name')"
while IFS= read -r asset; do
  [ -z "$asset" ] || gh_retry release delete-asset "$TAG" "$asset" --repo "$repo" --yes
done <<< "$assets"

say "uploading $name, then $MANIFEST"
gh_retry release upload "$TAG" --repo "$repo" "$wheel"
gh_retry release upload "$TAG" --repo "$repo" "$scratch/$MANIFEST"

say "rewriting the release notes"
gh_retry release edit "$TAG" --repo "$repo" --prerelease --title "Nightly" --notes "$notes" \
  >/dev/null

say "moving the $TAG tag to $short"
if ! moved="$(gh api --method PATCH "repos/$repo/git/refs/tags/$TAG" -f sha="$commit" \
  -F force=true 2>&1)"; then
  case "$moved" in
    *"Reference does not exist"*)
      # The release survived a deleted tag: create the tag again.
      gh_retry api --method POST "repos/$repo/git/refs" -f ref="refs/tags/$TAG" \
        -f sha="$commit" >/dev/null
      ;;
    *)
      gh_retry api --method PATCH "repos/$repo/git/refs/tags/$TAG" -f sha="$commit" \
        -F force=true >/dev/null
      ;;
  esac
fi

say "published $version to https://github.com/$repo/releases/tag/$TAG"
