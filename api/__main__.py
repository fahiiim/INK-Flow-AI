"""Local development launcher for the FastAPI application."""

import uvicorn


if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="10.10.28.89",
        port=8001,
        reload=True,
    )
