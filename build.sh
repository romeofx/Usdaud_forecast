#!/usr/bin/env bash

# Upgrade essential build tools
pip install --upgrade pip setuptools wheel

# Install dependencies
pip install -r requirements.txt
