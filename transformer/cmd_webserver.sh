#!/bin/bash
cd "$(dirname "$0")"
echo "Serving at http://localhost:9090/web_infer.html"
python -m http.server 9090
