import { useEffect, useState } from 'react'
import { api, when } from '../api'

export default function AuditLog() {
  const [data, setData] = useState(null)
  useEffect(() => { api.audit().then(setData) }, [])
  if (!data) return <div className="page">Loading…</div>
  const v = data.verification
  return (
    <div className="page">
      <h2>Audit log</h2>
      <p className={v.valid ? 'verified' : 'broken'}>
        {v.valid
          ? `✓ Hash chain verified — ${v.entries} entries, head ${v.head.slice(0, 16)}…`
          : `✕ Chain broken at entry ${v.broken_at_id} — the log was modified`}
      </p>
      <p className="muted">Append-only: each entry stores SHA-256(previous hash + entry). Editing or deleting any row breaks the chain.</p>
      <table className="audit">
        <thead><tr><th>#</th><th>Time</th><th>Actor</th><th>Action</th><th>Study</th><th>Details</th><th>Hash</th></tr></thead>
        <tbody>
          {data.entries.map((e) => (
            <tr key={e.id}>
              <td>{e.id}</td><td>{when(e.ts)}</td><td>{e.actor}</td><td>{e.action.replace(/_/g, ' ')}</td>
              <td className="mono">{e.study_id?.slice(0, 8)}</td>
              <td className="details">{e.details.summary || e.details.comment || (e.details.disagree_findings?.length ? `disagree: ${e.details.disagree_findings.join(', ')}` : '') || e.details.modality || ''}</td>
              <td className="mono">{e.hash.slice(0, 10)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
