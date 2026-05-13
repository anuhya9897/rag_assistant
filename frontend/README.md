# RAG POC – Frontend

Simple web UI to ask questions against the RAG API.

## Usage

1. Start the backend API (see `backend/README.md`).
2. Open the UI:
   - **File:** Open `frontend/index.html` in a browser.
   - **Or serve:** e.g. `npx serve frontend` and open http://localhost:3000

The app calls **http://localhost:8000** by default. To use another API URL, edit `API_BASE` in `js/app.js`.

## Structure

- `index.html` – Single-page layout (form, result tabs, error/loading).
- `css/style.css` – Theming and layout.
- `js/app.js` – Form submit, fetch `/api/ask`, display answer and sources.
