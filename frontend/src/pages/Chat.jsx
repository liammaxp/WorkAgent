import { useEffect, useRef, useState } from "react";
import { api } from "../api/client.js";
import {
  Alert,
  LoadingBar,
  PageHeader,
  useAsyncAction,
} from "../components/ui.jsx";
import { text, useLanguage } from "../i18n.jsx";

export default function Chat({ session, setSession }) {
  const { language } = useLanguage();
  const copy = text[language].chat;
  const { message, images, attachmentError, history } = session;
  const imageInputRef = useRef(null);
  const [supportsImages, setSupportsImages] = useState(true);
  const { loading, error, success, run } = useAsyncAction();
  const updateSession = (update) =>
    setSession((current) => ({
      ...current,
      ...(typeof update === "function" ? update(current) : update),
    }));

  const refreshImageSupport = async () => {
    try {
      const data = await api.getStatus();
      setSupportsImages(data.supports_images !== false);
    } catch {
      setSupportsImages(true);
    }
  };

  useEffect(() => {
    refreshImageSupport();
    const onFocus = () => refreshImageSupport();
    window.addEventListener("focus", onFocus);
    return () => window.removeEventListener("focus", onFocus);
  }, []);

  useEffect(() => {
    if (supportsImages || images.length === 0) return;
    updateSession({
      images: [],
      attachmentError: copy.imagesNotSupported,
    });
  }, [supportsImages, images.length, copy.imagesNotSupported]);

  const addImages = async (event) => {
    if (!supportsImages) {
      event.target.value = "";
      updateSession({ attachmentError: copy.imagesNotSupported });
      return;
    }

    const selected = Array.from(event.target.files || []);
    event.target.value = "";
    updateSession({ attachmentError: "" });

    if (images.length + selected.length > 4) {
      updateSession({ attachmentError: copy.tooManyImages });
      return;
    }

    const invalid = selected.find(
      (file) => !["image/jpeg", "image/png", "image/gif", "image/webp"].includes(file.type) || file.size > 10 * 1024 * 1024,
    );
    if (invalid) {
      updateSession({ attachmentError: copy.invalidImage });
      return;
    }

    const added = await Promise.all(
      selected.map(
        (file) =>
          new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve({ name: file.name, mime_type: file.type, data_url: reader.result });
            reader.onerror = () => reject(new Error(copy.imageReadFailed));
            reader.readAsDataURL(file);
          }),
      ),
    ).catch((readError) => {
      updateSession({ attachmentError: readError.message });
      return [];
    });
    updateSession((current) => ({ images: [...current.images, ...added] }));
  };

  const send = () =>
    run(async () => {
      const trimmed = message.trim();
      if (!trimmed && images.length === 0) return null;
      if (!supportsImages && images.length > 0) {
        updateSession({ images: [], attachmentError: copy.imagesNotSupported });
        return null;
      }

      const attachedImages = supportsImages ? images : [];
      const userEntry = { role: "user", text: trimmed || copy.imageOnlyMessage, images: attachedImages };
      updateSession((current) => ({
        history: [...current.history, userEntry],
        message: "",
        images: [],
        attachmentError: "",
      }));

      const data = await api.askAgent(trimmed, attachedImages);
      const agentEntry = { role: "agent", text: data.answer || "" };
      updateSession((current) => ({ history: [...current.history, agentEntry] }));
      return data;
    });

  return (
    <>
      <PageHeader title={copy.title} description={copy.description} />
      <LoadingBar loading={loading} />
      <Alert type="error" message={error} />
      <Alert type="success" message={success} />

      <section className="card" style={{ minHeight: 360 }}>
        {history.length === 0 ? (
          <p className="empty-state">{copy.empty}</p>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            {history.map((entry, index) => (
              <div
                key={index}
                style={{
                  alignSelf: entry.role === "user" ? "flex-end" : "flex-start",
                  maxWidth: "85%",
                  padding: "12px 16px",
                  borderRadius: 12,
                  background: entry.role === "user" ? "var(--accent-soft)" : "var(--surface-muted)",
                  fontSize: 14,
                  lineHeight: 1.7,
                  whiteSpace: "pre-wrap",
                }}
              >
                <strong style={{ display: "block", marginBottom: 4, fontSize: 12, color: "var(--text-muted)" }}>
                  {entry.role === "user" ? copy.you : "Agent"}
                </strong>
                {entry.images?.length > 0 && (
                  <div className="chat-image-grid">
                    {entry.images.map((image, imageIndex) => (
                      <img key={`${image.name}-${imageIndex}`} src={image.data_url} alt={image.name || copy.attachedImage} />
                    ))}
                  </div>
                )}
                {entry.text}
              </div>
            ))}
          </div>
        )}
      </section>

      <section className="card">
        <div className="field">
          <label>{copy.message}</label>
          <textarea
            className="short"
            value={message}
            onChange={(e) => updateSession({ message: e.target.value })}
            placeholder={copy.placeholder}
            onKeyDown={(e) => {
              if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) send();
            }}
          />
        </div>
        <input
          ref={imageInputRef}
          type="file"
          accept="image/jpeg,image/png,image/gif,image/webp"
          multiple
          hidden
          disabled={!supportsImages}
          onChange={addImages}
        />
        {images.length > 0 && (
          <div className="chat-attachment-list">
            {images.map((image, index) => (
              <div className="chat-attachment" key={`${image.name}-${index}`}>
                <img src={image.data_url} alt={image.name || copy.attachedImage} />
                <span title={image.name}>{image.name}</span>
                <button type="button" aria-label={copy.removeImage} onClick={() => updateSession((current) => ({ images: current.images.filter((_, itemIndex) => itemIndex !== index) }))}>
                  x
                </button>
              </div>
            ))}
          </div>
        )}
        {attachmentError && <p className="warning-line">{attachmentError}</p>}
        {!supportsImages && (
          <p className="meta-line">{copy.imagesNotSupported}</p>
        )}
        <div className="btn-row">
          <button
            type="button"
            className="btn btn-secondary"
            title={!supportsImages ? copy.imageUploadDisabled : undefined}
            onClick={() => {
              if (!supportsImages) {
                updateSession({ attachmentError: copy.imagesNotSupported });
                return;
              }
              imageInputRef.current?.click();
            }}
            disabled={loading || !supportsImages || images.length >= 4}
          >
            {supportsImages ? copy.addImage : copy.imageUploadDisabled}
          </button>
          <button
            type="button"
            className="btn btn-primary"
            onClick={send}
            disabled={loading || (!message.trim() && (!supportsImages || images.length === 0))}
          >
            {loading ? copy.thinking : copy.send}
          </button>
        </div>
      </section>
    </>
  );
}
