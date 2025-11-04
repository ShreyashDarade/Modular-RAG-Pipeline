import uvicorn


def run() -> None:
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    run()
