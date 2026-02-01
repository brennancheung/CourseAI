#!/bin/bash
set -e

# Get the directory where this script lives
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FILES_DIR="$SCRIPT_DIR/files"

# Target is current directory or first argument
TARGET_PATH="${1:-.}"
cd "$TARGET_PATH"

echo "Installing markdown rendering components..."

# Check prerequisites
if [ ! -f "package.json" ]; then
  echo "Error: package.json not found. Run this from a project root."
  exit 1
fi

if [ ! -d "src/components" ]; then
  echo "Error: src/components/ directory not found."
  exit 1
fi

# Install dependencies
echo "Adding dependencies..."
pnpm add react-markdown remark-gfm remark-math rehype-katex katex react-syntax-highlighter
pnpm add -D @types/react-syntax-highlighter @tailwindcss/typography

# Create components directory
mkdir -p src/components/markdown

# Copy component files
echo "Copying components..."
cp "$FILES_DIR/Markdown.tsx" src/components/markdown/
cp "$FILES_DIR/MarkdownComponents.tsx" src/components/markdown/
cp "$FILES_DIR/CodeBlock.tsx" src/components/markdown/

echo ""
echo "✓ Markdown components installed successfully!"
echo ""
echo "Usage:"
echo "  import { Markdown } from '@/components/markdown/Markdown'"
echo "  <Markdown content={markdownString} />"
