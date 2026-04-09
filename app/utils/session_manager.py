from app.db import SessionLocal
from app.models.db_models import Conversation, Message

def get_or_create_conversation(session_id: str):
    db = SessionLocal()
    conversation = db.query(Conversation).filter_by(session_id=session_id).first()

    if not conversation:
        conversation = Conversation(session_id=session_id)
        db.add(conversation)
        db.commit()
        db.refresh(conversation)

    return conversation, db


def get_history(session_id: str):
    conversation, db = get_or_create_conversation(session_id)

    messages = (
        db.query(Message)
        .filter_by(conversation_id=conversation.id)
        .order_by(Message.created_at)
        .all()
    )

    history = [{"role": m.role, "content": m.content} for m in messages]

    db.close()
    return history


def add_message(session_id: str, role: str, content: str):
    conversation, db = get_or_create_conversation(session_id)

    message = Message(
        conversation_id=conversation.id,
        role=role,
        content=content
    )

    db.add(message)
    db.commit()
    db.close()