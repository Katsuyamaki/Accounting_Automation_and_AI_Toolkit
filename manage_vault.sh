#!/usr/bin/env zsh

# Configuration
PROJECT_ID="crafty-isotope-429106-b1"

# Inputs
COMMAND=$1
CLIENT_NAME=$2

if [[ -z "$COMMAND" || -z "$CLIENT_NAME" ]]; then
    echo "Usage: ./manage_vault.sh create [client_name]"
    exit 1
fi

if [[ "$COMMAND" == "create" ]]; then
    # 1. Generate the key
    NEW_KEY=$(openssl rand -base64 32)
    SECRET_NAME="CLIENT_KEY_${CLIENT_NAME}"

    echo "🔐 Creating secret $SECRET_NAME..."

    # 2. Create the secret in Google Cloud
    echo -n "$NEW_KEY" | gcloud secrets create "$SECRET_NAME" \
        --data-file=- \
        --project="$PROJECT_ID" \
        --replication-policy="automatic"

    echo "✅ Success! Secret stored in Cloud Vault."
    echo "------------------------------------------------"
    echo "CLIENT_ID: $CLIENT_NAME"
    echo "API_KEY:   $NEW_KEY"
    echo "------------------------------------------------"
    echo "⚠️  Copy this key NOW. It cannot be recovered in plain text later."
fi
