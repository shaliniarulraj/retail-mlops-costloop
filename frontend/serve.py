import http.server
import socketserver
import os

PORT = int(os.environ.get("PORT", 8080))

handler = http.server.SimpleHTTPRequestHandler

os.chdir(os.path.dirname(os.path.abspath(__file__)))

with socketserver.TCPServer(("", PORT), handler) as httpd:
    print(f"Serving at port {PORT}")
    httpd.serve_forever()