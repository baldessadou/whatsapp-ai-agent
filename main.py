# Agent IA WhatsApp - Système de Commandes (version stable)
# FastAPI + SQLAlchemy + WhatsApp Business Cloud API v22.0
# - Menu interactif + fallback texte
# - Parsing robuste des commandes ("2 pizzas margherita", "1 coca", etc.)
# - Réception des réponses interactives (list_reply)
# - Confirmation avec "confirmer"

import os
import json
import logging
import unicodedata
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy import (
    create_engine, Column, Integer, String, DateTime, Float, Text, ForeignKey
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
import requests

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
@dataclass
class Config:
    WHATSAPP_TOKEN: str = os.getenv("WHATSAPP_TOKEN", "your_whatsapp_token")
    WHATSAPP_PHONE_ID: str = os.getenv("WHATSAPP_PHONE_ID", "your_phone_id")
    WHATSAPP_VERIFY_TOKEN: str = os.getenv("WHATSAPP_VERIFY_TOKEN", "verify_token_123")
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./whatsapp_orders.db")

config = Config()

# -----------------------------------------------------------------------------
# DB
# -----------------------------------------------------------------------------
engine = create_engine(config.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class OrderStatus(Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    PREPARING = "preparing"
    READY = "ready"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


class Customer(Base):
    __tablename__ = "customers"
    id = Column(Integer, primary_key=True, index=True)
    phone_number = Column(String, unique=True, index=True)
    name = Column(String)
    address = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    orders = relationship("Order", back_populates="customer")


class Product(Base):
    __tablename__ = "products"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(Text)
    price = Column(Float)
    category = Column(String)
    available = Column(String, default="true")  # "true" / "false"


class Order(Base):
    __tablename__ = "orders"
    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(Integer, ForeignKey("customers.id"))
    status = Column(String, default=OrderStatus.PENDING.value)
    total_amount = Column(Float)
    items = Column(Text)  # JSON string
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

    customer = relationship("Customer", back_populates="orders")


class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True, index=True)
    phone_number = Column(String, index=True)
    context = Column(Text)  # JSON
    last_interaction = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def normalize(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return s.replace("-", " ").strip()


# -----------------------------------------------------------------------------
# WhatsApp Service
# -----------------------------------------------------------------------------
class WhatsAppService:
    def __init__(self):
        self.token = config.WHATSAPP_TOKEN
        self.phone_id = config.WHATSAPP_PHONE_ID
        # Aligne avec tes tests cURL
        self.base_url = f"https://graph.facebook.com/v22.0/{self.phone_id}"

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    def send_message(self, to: str, message: str) -> bool:
        """Send plain text message."""
        url = f"{self.base_url}/messages"
        data = {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "text",
            "text": {"body": message},
        }
        try:
            r = requests.post(url, json=data, headers=self._headers(), timeout=15)
            ok = r.status_code in (200, 201)
            if ok:
                logging.info(f"WA text ok: {r.text}")
            else:
                logging.error(f"WA text failed {r.status_code}: {r.text}")
            return ok
        except Exception as e:
            logging.error(f"WA text error: {e}")
            return False

    def send_interactive_menu(self, to: str, products: List[Dict]) -> bool:
        """Send List Message (interactive)."""
        url = f"{self.base_url}/messages"

        rows = []
        for p in products[:10]:
            title = p.get("name", "Article")
            desc = (p.get("description") or "").strip()
            price = p.get("price")
            if price is not None:
                desc = (desc + (" - " if desc else "")) + f"€{price}"
            rows.append({
                "id": f"product_{p.get('id', title)}",
                "title": title[:24],
                "description": desc[:72],
            })

        data = {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "interactive",
            "interactive": {
                "type": "list",
                "header": {"type": "text", "text": "🍕 Menu Restaurant"},
                "body": {"text": "Choisissez vos articles :"},
                "footer": {"text": "Tapez 'confirmer' pour valider"},
                "action": {
                    "button": "Voir menu",
                    "sections": [{
                        "title": "Nos produits",
                        "rows": rows
                    }]
                }
            }
        }

        try:
            r = requests.post(url, json=data, headers=self._headers(), timeout=15)
            ok = r.status_code in (200, 201)
            if ok:
                logging.info(f"WA interactive ok: {r.text}")
            else:
                logging.error(f"WA interactive failed {r.status_code}: {r.text}")
            return ok
        except Exception as e:
            logging.error(f"WA interactive error: {e}")
            return False


# -----------------------------------------------------------------------------
# Order Service
# -----------------------------------------------------------------------------
class OrderService:
    def __init__(self, db: Session):
        self.db = db

    def get_or_create_customer(self, phone_number: str) -> Customer:
        c = self.db.query(Customer).filter(Customer.phone_number == phone_number).first()
        if not c:
            c = Customer(phone_number=phone_number)
            self.db.add(c)
            self.db.commit()
            self.db.refresh(c)
        return c

    def create_order(self, phone_number: str, items: List[Dict], notes: str = "") -> Order:
        customer = self.get_or_create_customer(phone_number)
        total = sum(item["price"] * item["quantity"] for item in items)
        order = Order(
            customer_id=customer.id,
            total_amount=total,
            items=json.dumps(items),
            notes=notes,
        )
        self.db.add(order)
        self.db.commit()
        self.db.refresh(order)
        return order

    def get_customer_orders(self, phone: str) -> List[Order]:
        customer = self.get_or_create_customer(phone)
        return self.db.query(Order).filter(Order.customer_id == customer.id).all()


# -----------------------------------------------------------------------------
# Conversation + Intents (rule-based reliable)
# -----------------------------------------------------------------------------
class ConversationService:
    def __init__(self, db: Session):
        self.db = db
        self.whatsapp = WhatsAppService()
        self.order_service = OrderService(db)

    # ---- context
    def get_conversation_context(self, phone: str) -> Dict:
        conv = self.db.query(Conversation).filter(Conversation.phone_number == phone).first()
        if conv and conv.context:
            return json.loads(conv.context)
        return {"state": "new", "current_order": []}

    def update_conversation_context(self, phone: str, context: Dict):
        conv = self.db.query(Conversation).filter(Conversation.phone_number == phone).first()
        if not conv:
            conv = Conversation(phone_number=phone)
            self.db.add(conv)
        conv.context = json.dumps(context)
        conv.last_interaction = datetime.utcnow()
        self.db.commit()

    # ---- product lookup helpers
    def _all_available_products(self) -> List[Product]:
        return self.db.query(Product).filter(Product.available == "true").all()

    def _product_index(self) -> Dict[str, Product]:
        """
        Mappe tokens normalisés -> produit (ex: 'margherita', 'pepperoni', 'carbonara', 'cesar', 'coca', 'eau')
        + nom complet normalisé
        """
        idx: Dict[str, Product] = {}
        for p in self._all_available_products():
            key = normalize(p.name)
            idx[key] = p
            # quelques alias utiles
            if "margherita" in key:
                idx["margherita"] = p
                idx["pizza margherita"] = p
            if "pepperoni" in key:
                idx["pepperoni"] = p
                idx["pizza pepperoni"] = p
            if "carbonara" in key:
                idx["carbonara"] = p
                idx["pasta carbonara"] = p
                idx["pates carbonara"] = p
            if "cesar" in key or "césar" in key:
                idx["cesar"] = p
                idx["salade cesar"] = p
            if "coca" in key:
                idx["coca"] = p
                idx["coca cola"] = p
            if normalize("eau") in key:
                idx["eau"] = p
        return idx

    # ---- rule-based intent & entity extraction
    def _detect_intent(self, msg: str) -> str:
        m = normalize(msg)
        if any(w in m for w in ["bonjour", "salut", "hello", "coucou"]):
            return "greeting"
        if "menu" in m:
            return "menu"
        if "confirmer" in m or "valider" in m:
            return "confirm"
        # tentative de parsing d'articles
        items = self._parse_items(m)
        if items:
            return "order"
        return "other"

    def _parse_items(self, msg_norm: str) -> List[Dict]:
        """
        Parse des phrases du style:
        - "2 pizzas margherita et 1 coca"
        - "1 pepperoni, 2 coca"
        Retour: [{"name": "Pizza Margherita", "price": 12.0, "quantity": 2}, ...]
        """
        idx = self._product_index()
        # tokens/candidats à chercher
        cand = list(idx.keys())
        found: List[Dict] = []

        # simple heuristique: on coupe par virgules/et/+
        parts = []
        for sep in [",", " et ", " + "]:
            if sep in msg_norm:
                msg_norm = msg_norm.replace(sep, "|")
        parts = [p.strip() for p in msg_norm.split("|") if p.strip()]

        if not parts:
            parts = [msg_norm]

        for chunk in parts:
            # quantité
            qty = 1
            tokens = chunk.split()
            if tokens and tokens[0].isdigit():
                qty = max(1, int(tokens[0]))
                chunk = " ".join(tokens[1:]).strip()

            # tente de trouver un produit par plus longue clé
            best_key = ""
            for key in cand:
                if key and key in chunk:
                    if len(key) > len(best_key):
                        best_key = key
            if best_key:
                p = idx[best_key]
                found.append({"name": p.name, "price": float(p.price), "quantity": qty})

        return found

    # ---- main processing
    def process_incoming_message(self, phone: str, message: str) -> str:
        context = self.get_conversation_context(phone)
        intent = self._detect_intent(message)

        logging.info(f"[intent={intent}] from={phone} msg={message!r} ctx={context}")

        if intent == "greeting":
            response = ("🍕 Bonjour! Bienvenue chez Barita Resto.\n"
                        "Tapez *menu* pour voir nos plats ou dites-moi directement votre commande "
                        "(ex: *2 margherita et 1 coca*).")
            context["state"] = "menu_or_order"

        elif intent == "menu":
            # envoi interactif + fallback texte
            products = self._all_available_products()
            products_dict = [{
                "id": p.id, "name": p.name, "description": p.description, "price": p.price
            } for p in products]
            ok = self.whatsapp.send_interactive_menu(phone, products_dict)
            if not ok:
                lines = ["🍕 *Notre menu*"]
                for p in products[:10]:
                    lines.append(f"• {p.name} — €{p.price:.2f}")
                lines.append("\nRépondez par ex. : 2 margherita, 1 coca")
                self.whatsapp.send_message(phone, "\n".join(lines))
            response = "📋 Menu envoyé ! Vous pouvez aussi me dire directement ce que vous voulez."
            context["state"] = "menu_shown"

        elif intent == "order":
            items = self._parse_items(normalize(message))
            if items:
                # cumule dans le panier
                context.setdefault("current_order", [])
                context["current_order"].extend(items)
                total = sum(i["price"] * i["quantity"] for i in context["current_order"])
                order_lines = [
                    f"• {i['quantity']}× {i['name']} — €{i['price'] * i['quantity']:.2f}"
                    for i in context["current_order"]
                ]
                response = ("✅ Ajouté à votre commande !\n\n"
                            "📋 *Récapitulatif*:\n" + "\n".join(order_lines) +
                            f"\n\n💰 *Total*: €{total:.2f}\n"
                            "Tapez *confirmer* pour valider, ou continuez à ajouter des articles.")
                context["state"] = "order_building"
            else:
                response = ("Je n'ai pas bien compris les articles. Donnez un format comme : "
                            "*2 margherita et 1 coca*.")

        elif intent == "confirm":
            cart = context.get("current_order", [])
            if cart:
                order = self.order_service.create_order(phone, cart)
                response = (f"🎉 Commande confirmée ! Numéro: #{order.id}\n"
                            "⏰ Temps de préparation: 25–30 minutes\n"
                            f"💰 Total: €{order.total_amount:.2f}\n"
                            "Vous recevrez une notification quand c'est prêt !")
                context["state"] = "order_confirmed"
                context["current_order"] = []
                context["last_order_id"] = order.id
            else:
                response = "Votre panier est vide. Ajoutez des articles avant de confirmer !"

        else:
            response = ("Je n'ai pas compris. Tapez *menu* pour voir nos options, "
                        "ou envoyez une commande du type *2 margherita et 1 coca*.")

        self.update_conversation_context(phone, context)
        return response

    # ---- interactive reply handler
    def process_interactive_reply(self, phone: str, list_reply_id: str, title: str) -> str:
        """
        list_reply_id reçu sous la forme 'product_<id>' – on ajoute 1 unité.
        """
        context = self.get_conversation_context(phone)
        product_id = None
        if list_reply_id.startswith("product_"):
            try:
                product_id = int(list_reply_id.split("_", 1)[1])
            except Exception:
                product_id = None

        if product_id is not None:
            p = self.db.query(Product).filter(Product.id == product_id).first()
            if p:
                item = {"name": p.name, "price": float(p.price), "quantity": 1}
                context.setdefault("current_order", [])
                context["current_order"].append(item)

                total = sum(i["price"] * i["quantity"] for i in context["current_order"])
                order_lines = [
                    f"• {i['quantity']}× {i['name']} — €{i['price'] * i['quantity']:.2f}"
                    for i in context["current_order"]
                ]
                response = ("✅ Ajouté au panier depuis le menu.\n\n"
                            "📋 *Récapitulatif*:\n" + "\n".join(order_lines) +
                            f"\n\n💰 *Total*: €{total:.2f}\n"
                            "Tapez *confirmer* pour valider, ou continuez à ajouter des articles.")
                context["state"] = "order_building"
                self.update_conversation_context(phone, context)
                return response

        # fallback si id non reconnu
        return ("Je n'ai pas pu ajouter cet élément. Réessayez depuis le *menu* "
                "ou envoyez un message du type *1 margherita*.")


# -----------------------------------------------------------------------------
# API
# -----------------------------------------------------------------------------
app = FastAPI(title="WhatsApp AI Agent - Système de Commandes")

@app.get("/")
async def root():
    return {"message": "WhatsApp AI Agent actif!", "status": "running"}


@app.get("/webhook")
async def verify_webhook(request: Request):
    verify_token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")
    if verify_token == config.WHATSAPP_VERIFY_TOKEN:
        return int(challenge)
    raise HTTPException(status_code=403, detail="Token invalide")


@app.post("/webhook")
async def handle_webhook(request: Request, db: Session = Depends(get_db)):
    """
    Gère:
    - messages texte
    - réponses interactives (list_reply)
    """
    try:
        body = await request.json()
        logging.info(f"INCOMING: {json.dumps(body)[:1200]}")

        entry = body.get("entry", [])
        if not entry:
            return JSONResponse(content={"status": "ignored"})

        changes = entry[0].get("changes", [])
        if not changes:
            return JSONResponse(content={"status": "ignored"})

        value = changes[0].get("value", {})
        messages = value.get("messages", [])
        if not messages:
            return JSONResponse(content={"status": "ignored"})

        conv = ConversationService(db)

        for msg in messages:
            phone_number = msg.get("from")
            msg_type = msg.get("type")

            if msg_type == "text":
                text = msg.get("text", {}).get("body", "")
                if text:
                    reply = conv.process_incoming_message(phone_number, text)
                    WhatsAppService().send_message(phone_number, reply)

            elif msg_type == "interactive":
                interactive = msg.get("interactive", {})
                list_reply = interactive.get("list_reply")
                if list_reply:
                    lr_id = list_reply.get("id", "")
                    title = list_reply.get("title", "")
                    reply = conv.process_interactive_reply(phone_number, lr_id, title)
                    WhatsAppService().send_message(phone_number, reply)

        return JSONResponse(content={"status": "success"})

    except Exception as e:
        logging.error(f"Erreur webhook: {e}", exc_info=True)
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/admin/products")
async def create_product(
    name: str, description: str, price: float, category: str, db: Session = Depends(get_db)
):
    product = Product(name=name, description=description, price=price, category=category)
    db.add(product)
    db.commit()
    db.refresh(product)
    return {"message": "Produit créé", "product_id": product.id}


@app.get("/admin/orders")
async def get_orders(db: Session = Depends(get_db)):
    orders = db.query(Order).all()
    out = []
    for o in orders:
        out.append({
            "id": o.id,
            "customer_phone": o.customer.phone_number if o.customer else None,
            "total": o.total_amount,
            "status": o.status,
            "created_at": o.created_at.isoformat(),
        })
    return out


@app.put("/admin/orders/{order_id}/status")
async def update_order_status(order_id: int, status: str, db: Session = Depends(get_db)):
    order = db.query(Order).filter(Order.id == order_id).first()
    if not order:
        raise HTTPException(status_code=404, detail="Commande introuvable")

    order.status = status
    order.updated_at = datetime.utcnow()
    db.commit()

    whatsapp = WhatsAppService()
    status_messages = {
        "confirmed": "✅ Votre commande a été confirmée!",
        "preparing": "👨‍🍳 Votre commande est en préparation...",
        "ready": "🎉 Votre commande est prête! Vous pouvez venir la récupérer.",
        "delivered": "📦 Commande livrée! Merci et à bientôt!",
        "cancelled": "❌ Votre commande a été annulée. Contactez-nous pour plus d'infos.",
    }
    if status in status_messages:
        # envoyer au client si connu
        if order.customer:
            whatsapp.send_message(order.customer.phone_number, f"Commande #{order.id}: {status_messages[status]}")

    return {"message": "Statut mis à jour"}


# -----------------------------------------------------------------------------
# Init + run
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)

def init_sample_data():
    db = SessionLocal()
    try:
        if db.query(Product).count() == 0:
            products = [
                Product(name="Pizza Margherita", description="Tomate, mozzarella, basilic", price=12.0, category="Pizza"),
                Product(name="Pizza Pepperoni", description="Tomate, mozzarella, pepperoni", price=14.0, category="Pizza"),
                Product(name="Pasta Carbonara", description="Pâtes, lardons, crème, parmesan", price=10.0, category="Pasta"),
                Product(name="Salade César", description="Salade, poulet, parmesan, croûtons", price=8.0, category="Salade"),
                Product(name="Coca-Cola", description="Boisson gazeuse 33cl", price=3.0, category="Boisson"),
                Product(name="Eau", description="Eau minérale 50cl", price=2.0, category="Boisson"),
            ]
            for p in products:
                db.add(p)
            db.commit()
            logging.info("✅ Données de test initialisées.")
    finally:
        db.close()


if __name__ == "__main__":
    init_sample_data()
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
