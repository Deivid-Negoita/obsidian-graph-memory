# Read-only recall service. Build the graph on the host, mount the vault :ro.
#   python graph_memory.py --vault ~/my-vault build
#   docker run -p 8765:8765 -v ~/my-vault:/vault:ro obsidian-graph-memory
FROM python:3.12-slim
COPY graph_memory.py /app/graph_memory.py
USER nobody
EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=3s \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/healthz')"]
ENTRYPOINT ["python", "/app/graph_memory.py", "--vault", "/vault"]
CMD ["serve", "--host", "0.0.0.0"]
