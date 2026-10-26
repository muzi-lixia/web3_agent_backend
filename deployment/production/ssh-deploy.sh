#!/usr/bin/env bash
# Forced command for the dedicated CI SSH key. Never eval SSH_ORIGINAL_COMMAND.
set -Eeuo pipefail
umask 077
read -r executable component image extra <<< "${SSH_ORIGINAL_COMMAND:-}"
[[ "$executable" == /opt/web3-agent/deploy.sh && -z "${extra:-}" ]] || exit 2
[[ "$component" == backend || "$component" == frontend ]] || exit 2
[[ "$image" =~ ^ghcr\.io/muzi-lixia/web3_agent_${component}@sha256:[a-f0-9]{64}$ ]] || exit 2

# GitHub passes the current job's short-lived packages:read token over SSH stdin.
# Docker credentials exist only for this deployment, not in ~/.docker/config.json.
IFS= read -r registry_user
IFS= read -r registry_token
[[ "$registry_user" =~ ^[A-Za-z0-9_-]+$ && -n "$registry_token" ]] || exit 2
export DOCKER_CONFIG
DOCKER_CONFIG=$(mktemp -d /tmp/web3-agent-registry.XXXXXX)
cleanup() { rm -rf -- "$DOCKER_CONFIG"; }
trap cleanup EXIT
printf '%s' "$registry_token" | docker login ghcr.io -u "$registry_user" --password-stdin
unset registry_token
/opt/web3-agent/deploy.sh "$component" "$image"
