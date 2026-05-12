from contextlib import contextmanager
from app.db import SessionLocal
from app.models.db_models import Conversation, Message


@contextmanager
def get_db():
    """Context manager that guarantees the DB session is always closed."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_or_create_conversation(session_id: str, db):
    conversation = db.query(Conversation).filter_by(session_id=session_id).first()

    if not conversation:
        conversation = Conversation(session_id=session_id)
        db.add(conversation)
        db.commit()
        db.refresh(conversation)

    return conversation


def get_history(session_id: str):
    with get_db() as db:
        conversation = get_or_create_conversation(session_id, db)

        messages = (
            db.query(Message)
            .filter_by(conversation_id=conversation.id)
            .order_by(Message.created_at)
            .all()
        )

        return [{"role": m.role, "content": m.content} for m in messages]


def add_message(session_id: str, role: str, content: str):
    with get_db() as db:
        conversation = get_or_create_conversation(session_id, db)

        message = Message(
            conversation_id=conversation.id,
            role=role,
            content=content
        )

        db.add(message)
        db.commit()


def get_all_sessions():
    """Return a list of all session IDs with message counts."""
    with get_db() as db:
        conversations = db.query(Conversation).order_by(Conversation.created_at.desc()).all()
        result = []
        for conv in conversations:
            count = db.query(Message).filter_by(conversation_id=conv.id).count()
            result.append({
                "session_id": conv.session_id,
                "created_at": conv.created_at.isoformat(),
                "message_count": count
            })
        return result


def delete_session(session_id: str) -> bool:
    """Delete a conversation and all its messages. Returns True if found and deleted."""
    with get_db() as db:
        conversation = db.query(Conversation).filter_by(session_id=session_id).first()
        if not conversation:
            return False
        db.query(Message).filter_by(conversation_id=conversation.id).delete()
        db.delete(conversation)
        db.commit()
        return True
