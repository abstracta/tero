#!/bin/bash

set -e

if [[ $DB_URL == "postgresql"* ]]; then
  SERVER="${DB_URL##*@}"
  SERVER="${SERVER%%/*}"
  if [[ $SERVER != *":"* ]]; then
    SERVER="${SERVER}:5432"
  fi
  /usr/src/app/wait-for-it.sh -t 60 "${SERVER}"
fi

if [[ -n "$OPENID_URL" ]]; then
  PROTOCOL="${OPENID_URL%%://*}"
  SERVER="${OPENID_URL##*://}"
  SERVER="${SERVER%%/*}"
  if [[ $SERVER != *":"* ]]; then
    if [[ $PROTOCOL == "https" ]]; then
      SERVER="${SERVER}:443"
    elif [[ $PROTOCOL == "http" ]]; then
      SERVER="${SERVER}:80"
    fi
  fi
  /usr/src/app/wait-for-it.sh -t 60 "${SERVER}"
fi

[ -n "$PRELOADED_TOKENIZER_MODELS" ] && export HF_HUB_OFFLINE=1

exec "$@"
