# print_routes.py
import os
from importlib import import_module

# Change this if you run uvicorn with a different module
APP_MODULE = os.getenv("FASTAPI_APP_MODULE", "app")
APP_OBJECT = os.getenv("FASTAPI_APP_OBJECT", "app")

app_mod = import_module(APP_MODULE)
app = getattr(app_mod, APP_OBJECT)

print(f"\nRoutes in {APP_MODULE}.{APP_OBJECT}:\n" + "-" * 40)

for route in app.routes:
    methods = ",".join(route.methods) if hasattr(route, "methods") else ""
    print(f"{route.path:30} {methods}")
