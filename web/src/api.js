const j = async (r) => {
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`)
  return r.json()
}

export const api = {
  studies: () => fetch('/api/studies').then(j),
  study: (id) => fetch(`/api/studies/${id}`).then(j),
  audit: () => fetch('/api/audit?limit=300').then(j),
  modelCard: () => fetch('/api/model-card').then(j),
  decide: (id, body) =>
    fetch(`/api/studies/${id}/decision`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }).then(j),
  imageUrl: (id, slice) => `/api/studies/${id}/image.png${slice != null ? `?slice=${slice}` : ''}`,
  overlayUrl: (id, { finding, slice } = {}) => {
    const q = new URLSearchParams()
    if (finding) q.set('finding', finding)
    if (slice != null) q.set('slice', slice)
    return `/api/studies/${id}/overlay.png?${q}`
  },
}

export const pct = (p) => `${(p * 100).toFixed(1)}%`
export const when = (t) => (t ? new Date(t * 1000).toLocaleString() : '—')
