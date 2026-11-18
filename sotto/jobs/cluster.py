"""Кластеры похожих вакансий: одно резюме и одно письмо на кластер, а не на каждую вакансию.

Алгоритм простой и предсказуемый:
* вакансия → эмбеддинг (та же локальная модель, что и для поиска);
* если похожа на центр существующего кластера (косинус ≥ ASSIGN) — в него;
* оставшиеся жадно группируются между собой; группа из ≥ MIN_SIZE становится новым кластером,
  которому агент даёт имя и описание; одиночки ждут, пока появятся похожие.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import numpy as np

from ..agents import AgentError, Agents
from ..db import DB

if TYPE_CHECKING:
    from ..search import Search

ASSIGN = 0.62
GROUP = 0.66
MIN_SIZE = 2


def _vec(blob: bytes | None) -> np.ndarray | None:
    return np.frombuffer(blob, dtype=np.float32) if blob else None


async def update_clusters(db: DB, search: "Search", agents: Agents) -> dict[str, int]:
    rows = db.all("SELECT id, title, text FROM vacancies WHERE cluster_id IS NULL AND status != 'hidden'")
    if not rows:
        return {"assigned": 0, "created": 0}
    vecs = await asyncio.to_thread(search.embed, [f"{r['title'] or ''}. {r['text'][:1500]}" for r in rows])
    clusters = [(c["id"], _vec(c["centroid"])) for c in db.all("SELECT id, centroid FROM clusters")]
    assigned = 0
    leftovers: list[int] = []
    for i, r in enumerate(rows):
        best_id, best_sim = None, -1.0
        for cid, cen in clusters:
            if cen is not None:
                sim = float(cen @ vecs[i])
                if sim > best_sim:
                    best_id, best_sim = cid, sim
        if best_id is not None and best_sim >= ASSIGN:
            db.update("vacancies", r["id"], cluster_id=best_id)
            assigned += 1
        else:
            leftovers.append(i)

    # Жадная группировка оставшихся.
    groups: list[list[int]] = []
    for i in leftovers:
        for g in groups:
            centroid = vecs[g].mean(axis=0)
            centroid /= np.linalg.norm(centroid) + 1e-9
            if float(centroid @ vecs[i]) >= GROUP:
                g.append(i)
                break
        else:
            groups.append([i])

    created = 0
    for g in groups:
        if len(g) < MIN_SIZE:
            continue
        sample = "\n\n".join(f"- {rows[i]['title'] or ''}: {rows[i]['text'][:600]}" for i in g[:8])
        prompt = ("Это группа похожих вакансий. Дай ей короткое название (роль + ключевой стек/домен, 2–6 слов) и "
                  "описание: общие требования, уровень, типичные задачи (3–4 строки). Верни ТОЛЬКО JSON "
                  '{"name": "...", "description": "..."}\n\n' + sample)
        try:
            meta, _ = await agents.run_json(prompt, task="название кластера")
            name, desc = str(meta.get("name") or "Кластер"), str(meta.get("description") or "")
        except (AgentError, AttributeError):
            name, desc = rows[g[0]]["title"] or "Кластер", ""
        centroid = vecs[g].mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-9
        cid = db.insert("clusters", name=name[:120], description=desc, centroid=centroid.astype(np.float32).tobytes(),
                        updated_at=time.time())
        for i in g:
            db.update("vacancies", rows[i]["id"], cluster_id=cid)
        created += 1

    await asyncio.to_thread(_recompute_centroids, db, search)
    return {"assigned": assigned, "created": created}


def _recompute_centroids(db: DB, search: "Search") -> None:
    for c in db.all("SELECT id FROM clusters"):
        vs = db.all("SELECT title, text FROM vacancies WHERE cluster_id = ?", [c["id"]])
        if not vs:
            continue
        vecs = search.embed([f"{v['title'] or ''}. {v['text'][:1500]}" for v in vs])
        cen = vecs.mean(axis=0)
        cen /= np.linalg.norm(cen) + 1e-9
        db.update("clusters", c["id"], centroid=cen.astype(np.float32).tobytes())
