#!/bin/bash

source /opt/conda/etc/profile.d/conda.sh

conda activate nose-ai

echo "🚀 Starting backend..."
nohup python /nose-ai/backend/main.py > backend.log 2>&1 &

cd /nose-ai/frontend

echo "📦 Installing frontend dependencies..."
yarn

echo "🚀 Starting frontend..."
nohup yarn dev > frontend.log 2>&1 &

echo "✅ Both servers started under Conda env: nose-ai"
