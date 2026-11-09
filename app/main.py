import logging

from dotenv import load_dotenv

load_dotenv()  # exposes optional LANGSMITH_* tracing variables to LangChain

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from fastapi.templating import Jinja2Templates  # noqa: E402

from app.api.routes import router  # noqa: E402
from app.core.config import BASE_DIR, get_settings  # noqa: E402
from app.services.audit import init_db  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

settings = get_settings()
init_db()

app = FastAPI(title=settings.app_name, version="1.0.0")
app.include_router(router)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


@app.get("/", include_in_schema=False)
def home(request: Request):
    return templates.TemplateResponse(request, "index.html", {"app_name": settings.app_name})
