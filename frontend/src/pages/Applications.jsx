import { useEffect, useState } from "react";
import { api } from "../api/client.js";
import {
  Alert,
  ConfirmDialog,
  LoadingBar,
  PageHeader,
  useAsyncAction,
} from "../components/ui.jsx";
import { text, useLanguage } from "../i18n.jsx";

const STATUS_OPTIONS = ["Interested", "Applied", "Interview", "Rejected", "Offer", "Archived"];

const EMPTY_FORM = {
  company: "",
  role: "",
  link: "",
  status: "Interested",
  applied_date: "",
  resume_version: "tailored_resume.txt",
  cover_letter_version: "cover_letter.txt",
  notes: "",
};

export default function Applications() {
  const { language } = useLanguage();
  const copy = text[language].applications;
  const common = text[language].common;
  const [records, setRecords] = useState([]);
  const [filter, setFilter] = useState("");
  const [form, setForm] = useState(EMPTY_FORM);
  const [deleteTarget, setDeleteTarget] = useState(null);
  const { loading, error, success, run } = useAsyncAction();

  const loadRecords = () =>
    run(async () => {
      const data = await api.getApplications(filter);
      setRecords(data);
      return data;
    });

  useEffect(() => {
    loadRecords();
  }, [filter]);

  const updateForm = (key, value) => {
    setForm((prev) => ({ ...prev, [key]: value }));
  };

  const createRecord = () =>
    run(async () => {
      await api.createApplication(form);
      setForm(EMPTY_FORM);
      await loadRecords();
    }, copy.added);

  const updateStatus = (id, status) =>
    run(async () => {
      await api.updateApplication(id, { status });
      await loadRecords();
    }, copy.updated);

  const deleteRecord = () =>
    run(async () => {
      await api.deleteApplication(deleteTarget.id);
      setDeleteTarget(null);
      await loadRecords();
    }, copy.deleted);

  return (
    <>
      <PageHeader title={copy.title} description={copy.description} />
      <LoadingBar loading={loading} />
      <Alert type="error" message={error} />
      <Alert type="success" message={success} />

      <section className="card">
        <h2 className="card-title">{copy.add}</h2>
        <div className="grid-2">
          <div className="field">
            <label>{copy.company}</label>
            <input value={form.company} onChange={(e) => updateForm("company", e.target.value)} />
          </div>
          <div className="field">
            <label>{copy.role}</label>
            <input value={form.role} onChange={(e) => updateForm("role", e.target.value)} />
          </div>
          <div className="field">
            <label>{common.link}</label>
            <input value={form.link} onChange={(e) => updateForm("link", e.target.value)} />
          </div>
          <div className="field">
            <label>{copy.status}</label>
            <select value={form.status} onChange={(e) => updateForm("status", e.target.value)}>
              {STATUS_OPTIONS.map((status) => <option key={status} value={status}>{status}</option>)}
            </select>
          </div>
          <div className="field">
            <label>{copy.appliedDate}</label>
            <input value={form.applied_date} onChange={(e) => updateForm("applied_date", e.target.value)} placeholder="YYYY-MM-DD" />
          </div>
          <div className="field">
            <label>{copy.notes}</label>
            <input value={form.notes} onChange={(e) => updateForm("notes", e.target.value)} />
          </div>
        </div>
        <div className="btn-row">
          <button type="button" className="btn btn-primary" onClick={createRecord} disabled={loading || !form.company || !form.role}>
            {copy.addRecord}
          </button>
        </div>
      </section>

      <section className="card">
        <div className="section-toolbar">
          <h2 className="card-title">{copy.list}</h2>
          <select value={filter} onChange={(e) => setFilter(e.target.value)}>
            <option value="">{copy.allStatuses}</option>
            {STATUS_OPTIONS.map((status) => <option key={status} value={status}>{status}</option>)}
          </select>
        </div>

        {records.length === 0 ? (
          <p className="empty-state">{copy.noRecords}</p>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>{copy.company}</th>
                  <th>{copy.role}</th>
                  <th>{copy.status}</th>
                  <th>{copy.appliedDate}</th>
                  <th>{copy.actions}</th>
                </tr>
              </thead>
              <tbody>
                {records.map((record) => (
                  <tr key={record.id}>
                    <td>{record.company}</td>
                    <td>{record.role}</td>
                    <td>
                      <select
                        value={record.status || "Interested"}
                        onChange={(e) => updateStatus(record.id, e.target.value)}
                        style={{ border: "none", background: "transparent", fontWeight: 600 }}
                      >
                        {STATUS_OPTIONS.map((status) => <option key={status} value={status}>{status}</option>)}
                      </select>
                    </td>
                    <td>{record.applied_date || common.none}</td>
                    <td>
                      {record.link ? (
                        <a href={record.link} target="_blank" rel="noreferrer" style={{ color: "var(--accent)" }}>
                          {common.link}
                        </a>
                      ) : (
                        common.none
                      )}
                      <button
                        type="button"
                        className="btn btn-danger btn-small"
                        onClick={() => setDeleteTarget(record)}
                        disabled={loading}
                        style={{ marginLeft: record.link ? 12 : 0 }}
                      >
                        {common.delete}
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <ConfirmDialog
        open={Boolean(deleteTarget)}
        title={copy.deleteTitle}
        confirmLabel={common.delete}
        loading={loading}
        onCancel={() => setDeleteTarget(null)}
        onConfirm={deleteRecord}
      >
        <p>
          {copy.deleteBody
            .replace("{company}", deleteTarget?.company || "")
            .replace("{role}", deleteTarget?.role || "")}
        </p>
      </ConfirmDialog>
    </>
  );
}
