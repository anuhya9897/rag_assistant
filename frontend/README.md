# Frontend — legacy standalone UI

**Alternate** web UI for the Simba RAG API. The **primary** UI is [`../web/`](../web/README.md), served at **`http://localhost:8000/ui/`** when you run `python backend/run.py`.

Use this folder only if you want a separate static host or the older dark-theme layout.

## Usage

1. Start the API (see [`../backend/README.md`](../backend/README.md)).  
2. Open the UI:
   - Open `frontend/index.html` in a browser, or  
   - `npx serve frontend` → [http://localhost:3000](http://localhost:3000)

Default API base: **http://localhost:8000** (`js/app.js` → `API_BASE`).

## Layout

| Path | Role |
|------|------|
| `index.html` | Single-page form, result tabs, loading/error |
| `css/style.css` | Dark theme |
| `js/app.js` | `POST /api/ask`, sources display |

## Limitations vs `web/`

- No streaming UI, reindex panel, or RedMane branding  
- No model **Refresh** / ensure flow  
- Manual `API_BASE` edit for non-default hosts  

See root [README.md](../README.md) for full project documentation.
