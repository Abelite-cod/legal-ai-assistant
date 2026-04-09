def build_context(docs, max_chars=2000):
    context = ""
    sources = []

    for doc in docs:
        text = doc.page_content

        if len(context) + len(text) > max_chars:
            break

        context += text + "\n\n"

        meta = doc.metadata or {}
        page = meta.get("page", "Unknown")
        sources.append(f"Page {page}")

    return context.strip(), list(set(sources))